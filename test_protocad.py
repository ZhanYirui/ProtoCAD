import argparse
import random
import sys
from pathlib import Path

PROTOCAD_ROOT = Path(__file__).resolve().parent
if str(PROTOCAD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOCAD_ROOT))

import numpy as np
import torch
from lightning import Fabric
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from transformers import AutoTokenizer

from src.dataset import PassagesDataset
from src.protocad_epoch_shell import ProtoCADEpochShellModel
from utils.Deepfake_utils import load_deepfake
from utils.M4_utils import load_M4
from utils.OUTFOX_utils import load_OUTFOX
from utils.Turing_utils import load_Turing
from utils.raid_utils import load_raid
from utils.utils import best_threshold_by_f1


def collate_fn(batch):
    text, label, write_model, write_model_set = default_collate(batch)
    encoded_batch = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    return encoded_batch, label, write_model, write_model_set


def load_dataset_splits(opt):
    if opt.dataset == "deepfake":
        dataset = load_deepfake(opt.path)
        mode = "deepfake"
    elif opt.dataset == "TuringBench":
        dataset = load_Turing(file_folder=opt.path)
        mode = "Turing"
    elif opt.dataset == "OUTFOX":
        dataset = load_OUTFOX(opt.path, opt.attack)
        mode = "OUTFOX"
    elif opt.dataset == "M4":
        dataset = load_M4(opt.path)
        mode = "M4"
    elif opt.dataset == "raid":
        dataset = load_raid()
        mode = "raid"
    else:
        raise ValueError(f"Unsupported dataset: {opt.dataset}")

    test_dataset = PassagesDataset(dataset[opt.test_dataset_name], mode=mode)
    geometry_dataset = None
    if opt.geometry_dataset_name:
        geometry_dataset = PassagesDataset(dataset[opt.geometry_dataset_name], mode=mode)
    return test_dataset, geometry_dataset


def infer(test_loader, fabric, model):
    if fabric.global_rank == 0:
        test_loader = tqdm(test_loader, total=len(test_loader), desc="Test")

    model.eval()
    distances = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in test_loader:
            encoded_batch, label, _, _ = batch
            encoded_batch = {k: v.to(fabric.device) for k, v in encoded_batch.items()}
            label = label.to(fabric.device)

            loss, _, _, _, _, batch_distances = model(encoded_batch, label)
            distances.append(batch_distances.detach().cpu())
            labels.append(label.detach().cpu())
            losses.append(loss.detach().cpu())

    return distances, labels, losses


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_checkpoint(model, opt, fabric, geometry_loader):
    checkpoint = torch.load(opt.model_path, map_location="cpu")
    saved_threshold = None

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        saved_threshold = checkpoint.get("threshold")
    else:
        model.get_encoder().load_state_dict(checkpoint, strict=False)
        if geometry_loader is None:
            raise ValueError(
                "Encoder-only checkpoints need --geometry_dataset_name so the prototype and radii can be estimated."
            )
        model.estimate_epoch_geometry(geometry_loader, fabric.device, show_progress=fabric.global_rank == 0)

    return saved_threshold


def compute_results(label_np, distance_np, threshold):
    y_pred = np.where(distance_np > threshold, 1, 0)
    fpr, tpr, _ = roc_curve(label_np, distance_np)
    roc_auc = auc(fpr, tpr)

    precision_curve, recall_curve, _ = precision_recall_curve(label_np, distance_np)
    pr_auc = auc(recall_curve, precision_curve)
    average_pr_auc = average_precision_score(label_np, distance_np)

    target_fpr = 0.05
    tpr_at_fpr_5 = np.interp(target_fpr, fpr, tpr)

    target_tpr = 0.95
    fpr_at_tpr_95 = np.interp(target_tpr, tpr, fpr)

    acc = accuracy_score(label_np, y_pred)
    precision = precision_score(label_np, y_pred)
    recall = recall_score(label_np, y_pred)
    f1 = f1_score(label_np, y_pred)
    return roc_auc, pr_auc, average_pr_auc, tpr_at_fpr_5, fpr_at_tpr_95, acc, precision, recall, f1


def test(opt):
    fabric = Fabric(accelerator="cuda", devices=opt.device_num)
    fabric.launch()

    test_dataset, geometry_dataset = load_dataset_splits(opt)
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    geometry_loader = None
    if geometry_dataset is not None:
        geometry_loader = DataLoader(
            geometry_dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    model = ProtoCADEpochShellModel(opt).to(fabric.device)
    saved_threshold = load_checkpoint(model, opt, fabric, geometry_loader)
    test_loader = fabric.setup_dataloaders(test_loader)
    model = fabric.setup(model)

    distance_list, label_list, loss_list = infer(test_loader, fabric, model)
    fabric.barrier()

    if fabric.global_rank == 0:
        distance_np = torch.cat(distance_list).view(-1).numpy()
        label_np = torch.cat(label_list).view(-1).numpy()
        test_loss = torch.stack(loss_list).mean().item() if loss_list else float("nan")

        if opt.use_saved_threshold and saved_threshold is not None:
            threshold = float(saved_threshold)
            threshold_f1 = float("nan")
        else:
            threshold, threshold_f1 = best_threshold_by_f1(label_np, distance_np)

        roc_auc, pr_auc, average_pr_auc, tpr_at_fpr_5, fpr_at_tpr_95, acc, precision, recall, f1 = compute_results(
            label_np, distance_np, threshold
        )
        print(
            f"Test, Loss: {test_loss}, AUC: {roc_auc}, pr_auc: {pr_auc}, "
            f"average_pr_auc: {average_pr_auc}, tpr_at_fpr_5: {tpr_at_fpr_5}, "
            f"fpr_at_tpr_95: {fpr_at_tpr_95}, Threshold: {threshold}, "
            f"ThresholdF1: {threshold_f1}, Acc:{acc}, Precision:{precision}, Recall:{recall}, F1:{f1}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--projection_size", type=int, default=768, help="Pretrained model output dim")
    parser.add_argument("--temperature", type=float, default=0.07, help="contrastive loss temperature")

    parser.add_argument("--lambda_con", type=float, default=1.0, help="contrastive loss weight")
    parser.add_argument("--lambda_shell", type=float, default=1.0, help="shell loss weight")
    parser.add_argument("--q_m", type=float, default=0.9, help="quantile for machine radius")
    parser.add_argument("--q_h", type=float, default=0.1, help="quantile for human radius before max with r_m")

    parser.add_argument("--dataset", type=str, default="deepfake", help="deepfake, OUTFOX, TuringBench, M4, raid")
    parser.add_argument("--path", type=str, default="./data/Deepfake/cross_domains_cross_models")
    parser.add_argument("--test_dataset_name", type=str, default="test", help="train, valid, test, test_ood")
    parser.add_argument(
        "--geometry_dataset_name",
        type=str,
        default="",
        help="Optional split used to estimate geometry for encoder-only checkpoints.",
    )
    parser.add_argument("--attack", type=str, default="none", help="Attack type only for OUTFOX dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model_protocad_best.pth or encoder checkpoint")
    parser.add_argument("--model_name", type=str, default="princeton-nlp/unsup-simcse-roberta-base")
    parser.add_argument("--use_saved_threshold", action="store_true", help="Use threshold stored in protocad checkpoint")
    parser.add_argument("--resum", type=bool, default=False)
    parser.add_argument("--pth_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    opt = parser.parse_args()

    set_seed(opt.seed)
    tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
    test(opt)
