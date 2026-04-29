import sys
from pathlib import Path

PROTOCAD_ROOT = Path(__file__).resolve().parent
if str(PROTOCAD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOCAD_ROOT))

import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
import yaml
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from src.dataset import PassagesDataset
from src.protocad_epoch_shell import ProtoCADEpochShellModel
from utils.Deepfake_utils import load_deepfake
from utils.M4_utils import load_M4
from utils.OUTFOX_utils import load_OUTFOX
from utils.Turing_utils import load_Turing
from utils.raid_utils import load_raid
from utils.utils import best_threshold_by_f1, compute_metrics


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


def unwrap_model(model):
    raw_model = getattr(model, "_forward_module", model)
    raw_model = getattr(raw_model, "module", raw_model)
    return raw_model


def gather_variable_length_1d(fabric, values, dtype):
    local_tensor = torch.tensor(values, device=fabric.device, dtype=dtype)
    local_length = torch.tensor([local_tensor.numel()], device=fabric.device, dtype=torch.long)

    if fabric.world_size == 1:
        return local_tensor.unsqueeze(0), local_length

    gathered_lengths = fabric.all_gather(local_length).flatten()
    max_length = int(gathered_lengths.max().item()) if gathered_lengths.numel() > 0 else 0
    if local_tensor.numel() < max_length:
        padding = torch.zeros(max_length - local_tensor.numel(), device=fabric.device, dtype=dtype)
        local_tensor = torch.cat([local_tensor, padding], dim=0)

    return fabric.all_gather(local_tensor), gathered_lengths


def build_dataset(opt):
    if opt.dataset == "deepfake":
        dataset = load_deepfake(opt.path)
        train_dataset = PassagesDataset(dataset[opt.database_name], mode="deepfake")
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name], mode="deepfake")
    elif opt.dataset == "TuringBench":
        dataset = load_Turing(file_folder=opt.path)
        train_dataset = PassagesDataset(dataset[opt.database_name], mode="Turing")
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name], mode="Turing")
    elif opt.dataset == "OUTFOX":
        dataset = load_OUTFOX(opt.path)
        train_dataset = PassagesDataset(dataset[opt.database_name], mode="OUTFOX")
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name], mode="OUTFOX")
    elif opt.dataset == "M4":
        dataset = load_M4(opt.path)
        train_split = dataset[opt.database_name]
        if opt.database_name.startswith("train"):
            dev_name = opt.database_name.replace("train", "dev")
            if dev_name in dataset:
                train_split = train_split + dataset[dev_name]
        train_dataset = PassagesDataset(train_split, mode="M4")
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name], mode="M4")
    elif opt.dataset == "raid":
        dataset = load_raid()
        train_dataset = PassagesDataset(dataset[opt.database_name], mode="raid")
        val_dataset = PassagesDataset(dataset[opt.test_dataset_name], mode="raid")
    else:
        raise ValueError(f"Unsupported dataset: {opt.dataset}")
    return train_dataset, val_dataset


def select_threshold(labels, distances):
    labels = np.asarray(labels, dtype=int)
    distances = np.asarray(distances, dtype=float)
    if distances.size == 0:
        raise ValueError("Validation set is empty; cannot select threshold.")
    if np.unique(labels).size < 2:
        return float(np.median(distances))
    best_threshold, _ = best_threshold_by_f1(labels, distances)
    return float(best_threshold)


def evaluate_threshold(labels, distances, threshold):
    preds = np.where(np.asarray(distances) > threshold, 1, 0)
    labels_str = [str(int(item)) for item in labels]
    preds_str = [str(int(item)) for item in preds]
    human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = compute_metrics(labels_str, preds_str)
    return {
        "human_rec": human_rec,
        "machine_rec": machine_rec,
        "avg_rec": avg_rec,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def estimate_and_broadcast_geometry(model, fabric, loader):
    raw_model = unwrap_model(model)
    if fabric.global_rank == 0:
        raw_model.estimate_epoch_geometry(loader, fabric.device, show_progress=True)
    if fabric.world_size > 1:
        fabric.barrier()
        torch.distributed.broadcast(raw_model.center, src=0)
        torch.distributed.broadcast(raw_model.radius_m, src=0)
        torch.distributed.broadcast(raw_model.radius_h, src=0)
    fabric.barrier()


def train(opt):
    torch.set_float32_matmul_precision("medium")
    if opt.device_num > 1:
        fabric = Fabric(
            accelerator="cuda",
            precision="bf16-mixed",
            devices=opt.device_num,
            strategy=DDPStrategy(find_unused_parameters=True),
        )
    else:
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=opt.device_num)
    fabric.launch()

    train_dataset, val_dataset = build_dataset(opt)
    geometry_loader = DataLoader(
        train_dataset,
        batch_size=opt.per_gpu_eval_batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.per_gpu_batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.per_gpu_eval_batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    model = ProtoCADEpochShellModel(opt).to(fabric.device).train()
    if opt.freeze_embedding_layer:
        for name, param in model.model.named_parameters():
            if "emb" in name:
                param.requires_grad = False

    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    if fabric.global_rank == 0:
        for num in range(10000):
            candidate = os.path.join(opt.savedir, f"{opt.name}_v{num}")
            if not os.path.exists(candidate):
                opt.savedir = candidate
                os.makedirs(opt.savedir)
                break
        runs_dir = os.path.join(opt.savedir, "runs")
        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)
        writer = SummaryWriter(runs_dir)
        with open(os.path.join(opt.savedir, "config.yaml"), "w") as file:
            yaml.dump(vars(opt), file, sort_keys=False)

    num_batches_per_epoch = len(train_loader)
    warmup_steps = opt.warmup_steps
    lr = opt.lr
    total_steps = max(opt.total_epoch * num_batches_per_epoch - warmup_steps, 1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr,
        betas=(opt.beta1, opt.beta2),
        eps=opt.eps,
        weight_decay=opt.weight_decay,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=lr / 10)
    model, optimizer = fabric.setup(model, optimizer)

    best_f1 = float("-inf")
    for epoch in range(opt.total_epoch):
        estimate_and_broadcast_geometry(model, fabric, geometry_loader)

        model.train()
        avg_loss = 0.0
        pbar = enumerate(train_loader)
        if fabric.global_rank == 0:
            pbar = tqdm(pbar, total=len(train_loader))
            print(
                ("\n" + "%11s" * 9)
                % ("Epoch", "GPU_mem", "Loss", "AvgLoss", "L_shell", "L_m", "L_h", "L_con", "r_m")
            )

        for i, batch in pbar:
            optimizer.zero_grad()
            current_step = epoch * num_batches_per_epoch + i
            if current_step < warmup_steps:
                current_lr = lr * current_step / max(warmup_steps, 1)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
            current_lr = optimizer.param_groups[0]["lr"]

            encoded_batch, label, _, _ = batch
            encoded_batch = {k: v.to(fabric.device) for k, v in encoded_batch.items()}
            label = label.to(fabric.device)

            loss, loss_shell, loss_m, loss_h, loss_con, _ = model(encoded_batch, label)
            avg_loss = (avg_loss * i + loss.item()) / (i + 1)
            fabric.backward(loss)
            optimizer.step()
            if current_step >= warmup_steps:
                schedule.step()

            if fabric.global_rank == 0:
                mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 7)
                    % (
                        f"{epoch + 1}/{opt.total_epoch}",
                        mem,
                        loss.item(),
                        avg_loss,
                        loss_shell.item(),
                        loss_m.item(),
                        loss_h.item(),
                        loss_con.item(),
                        unwrap_model(model).radius_m.item(),
                    )
                )
                if current_step % 10 == 0:
                    writer.add_scalar("lr", current_lr, current_step)
                    writer.add_scalar("train/loss", loss.item(), current_step)
                    writer.add_scalar("train/avg_loss", avg_loss, current_step)
                    writer.add_scalar("train/loss_shell", loss_shell.item(), current_step)
                    writer.add_scalar("train/loss_machine_shell", loss_m.item(), current_step)
                    writer.add_scalar("train/loss_human_shell", loss_h.item(), current_step)
                    writer.add_scalar("train/loss_con", loss_con.item(), current_step)

        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_distances = []
            val_labels = []
            pbar = enumerate(val_loader)
            if fabric.global_rank == 0:
                pbar = tqdm(pbar, total=len(val_loader))
                print(("\n" + "%11s" * 5) % ("Epoch", "GPU_mem", "ValLoss", "r_m", "r_h"))

            for i, batch in pbar:
                encoded_batch, label, _, _ = batch
                encoded_batch = {k: v.to(fabric.device) for k, v in encoded_batch.items()}
                label = label.to(fabric.device)

                loss, _, _, _, _, distances = model(encoded_batch, label)
                val_loss = (val_loss * i + loss.item()) / (i + 1)
                val_distances.extend(distances.detach().cpu().tolist())
                val_labels.extend(label.detach().cpu().tolist())

                if fabric.global_rank == 0:
                    mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * 3)
                        % (
                            f"{epoch + 1}/{opt.total_epoch}",
                            mem,
                            val_loss,
                            unwrap_model(model).radius_m.item(),
                            unwrap_model(model).radius_h.item(),
                        )
                    )

        gathered_distances, gathered_distance_lengths = gather_variable_length_1d(fabric, val_distances, torch.float32)
        gathered_labels, gathered_label_lengths = gather_variable_length_1d(fabric, val_labels, torch.long)

        if fabric.global_rank == 0:
            all_distances = []
            for rank_idx, length in enumerate(gathered_distance_lengths.tolist()):
                all_distances.extend(gathered_distances[rank_idx, :length].cpu().tolist())
            all_labels = []
            for rank_idx, length in enumerate(gathered_label_lengths.tolist()):
                all_labels.extend(gathered_labels[rank_idx, :length].cpu().tolist())

            all_distances = np.asarray(all_distances)
            all_labels = np.asarray(all_labels)
            threshold = select_threshold(all_labels, all_distances)
            threshold_metrics = evaluate_threshold(all_labels, all_distances, threshold)

            try:
                auc = roc_auc_score(all_labels, all_distances)
            except ValueError:
                auc = float("nan")
            try:
                pr_auc = average_precision_score(all_labels, all_distances)
            except ValueError:
                pr_auc = float("nan")

            print(
                "Validation "
                f"AUC: {auc}, PRAUC: {pr_auc}, Threshold: {threshold}, "
                f"HumanRec: {threshold_metrics['human_rec']}, "
                f"MachineRec: {threshold_metrics['machine_rec']}, "
                f"AvgRec: {threshold_metrics['avg_rec']}, "
                f"Acc: {threshold_metrics['acc']}, "
                f"Precision: {threshold_metrics['precision']}, "
                f"Recall: {threshold_metrics['recall']}, "
                f"F1: {threshold_metrics['f1']}"
            )

            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/auc", auc, epoch)
            writer.add_scalar("val/pr_auc", pr_auc, epoch)
            writer.add_scalar("val/threshold", threshold, epoch)
            writer.add_scalar("val/human_rec", threshold_metrics["human_rec"], epoch)
            writer.add_scalar("val/machine_rec", threshold_metrics["machine_rec"], epoch)
            writer.add_scalar("val/avg_rec", threshold_metrics["avg_rec"], epoch)
            writer.add_scalar("val/acc", threshold_metrics["acc"], epoch)
            writer.add_scalar("val/precision", threshold_metrics["precision"], epoch)
            writer.add_scalar("val/recall", threshold_metrics["recall"], epoch)
            writer.add_scalar("val/f1", threshold_metrics["f1"], epoch)
            writer.add_scalar("val/radius_m", unwrap_model(model).radius_m.item(), epoch)
            writer.add_scalar("val/radius_h", unwrap_model(model).radius_h.item(), epoch)

            raw_model = unwrap_model(model)
            checkpoint = {
                "model_state_dict": raw_model.state_dict(),
                "threshold": threshold,
                "metric_name": "f1",
                "metric_value": threshold_metrics["f1"],
                "radius_m": raw_model.radius_m.item(),
                "radius_h": raw_model.radius_h.item(),
            }

            if threshold_metrics["f1"] > best_f1:
                best_f1 = threshold_metrics["f1"]
                torch.save(raw_model.get_encoder().state_dict(), os.path.join(opt.savedir, "model_best.pth"))
                torch.save(checkpoint, os.path.join(opt.savedir, "model_protocad_best.pth"))
                print(f"Save model to {os.path.join(opt.savedir, 'model_best.pth')}", flush=True)

            torch.save(raw_model.get_encoder().state_dict(), os.path.join(opt.savedir, "model_last.pth"))
            torch.save(checkpoint, os.path.join(opt.savedir, "model_protocad_last.pth"))
            print(f"Save model to {os.path.join(opt.savedir, 'model_last.pth')}", flush=True)

        fabric.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=1, help="GPU number to use")
    parser.add_argument("--projection_size", type=int, default=768, help="Pretrained model output dim")
    parser.add_argument("--temperature", type=float, default=0.07, help="contrastive loss temperature")
    parser.add_argument("--num_workers", type=int, default=0, help="num_workers for dataloader")
    parser.add_argument("--per_gpu_batch_size", default=16, type=int, help="Batch size per GPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, help="Batch size per GPU for evaluation.")

    parser.add_argument("--dataset", type=str, default="deepfake", help="deepfake, OUTFOX, TuringBench, M4, raid")
    parser.add_argument("--path", type=str, default="./data/Deepfake/cross_domains_cross_models")
    parser.add_argument("--database_name", type=str, default="train", help="train, valid, test, test_ood")
    parser.add_argument("--test_dataset_name", type=str, default="valid", help="train, valid, test, test_ood")

    parser.add_argument("--lambda_con", type=float, default=1.0, help="contrastive loss weight")
    parser.add_argument("--lambda_shell", type=float, default=1.0, help="shell loss weight")
    parser.add_argument("--q_m", type=float, default=0.9, help="quantile for machine radius")
    parser.add_argument("--q_h", type=float, default=0.1, help="quantile for human radius before max with r_m")

    parser.add_argument("--total_epoch", type=int, default=10, help="Total number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1")
    parser.add_argument("--beta2", type=float, default=0.98, help="beta2")
    parser.add_argument("--eps", type=float, default=1e-6, help="eps")
    parser.add_argument("--savedir", type=str, default="./runs")
    parser.add_argument("--name", type=str, default="protocad")

    parser.add_argument("--resum", type=bool, default=False)
    parser.add_argument("--pth_path", type=str, default="", help="resume embedding model path")
    parser.add_argument("--model_name", type=str, default="princeton-nlp/unsup-simcse-roberta-base")
    parser.add_argument("--freeze_embedding_layer", action="store_true", help="freeze embedding layer")
    opt = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
    train(opt)
