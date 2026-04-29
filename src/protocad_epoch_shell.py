import torch
import torch.nn as nn
import torch.nn.functional as F

from src.text_embedding import TextEmbeddingModel


class ProtoCADEpochShellModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = TextEmbeddingModel(opt.model_name)
        if opt.resum and opt.pth_path:
            state_dict = torch.load(opt.pth_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)

        self.temperature = opt.temperature
        self.lambda_con = opt.lambda_con
        self.lambda_shell = opt.lambda_shell
        self.q_m = opt.q_m
        self.q_h = opt.q_h

        self.register_buffer("center", torch.zeros(opt.projection_size))
        self.register_buffer("radius_m", torch.tensor(0.0))
        self.register_buffer("radius_h", torch.tensor(0.0))

    def get_encoder(self):
        return self.model

    def encode(self, encoded_batch):
        return self.model(encoded_batch)

    @torch.no_grad()
    def estimate_epoch_geometry(self, train_loader, device):
        machine_embeddings = []
        human_embeddings = []

        for encoded_batch, label, _, _ in train_loader:
            encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
            label = label.to(device)
            embeddings = self.encode(encoded_batch)

            machine_mask = label == 0
            human_mask = label == 1
            if machine_mask.any():
                machine_embeddings.append(embeddings[machine_mask])
            if human_mask.any():
                human_embeddings.append(embeddings[human_mask])

        if not machine_embeddings:
            raise ValueError("No machine samples were found for geometry estimation.")
        if not human_embeddings:
            raise ValueError("No human samples were found for geometry estimation.")

        machine_embeddings = torch.cat(machine_embeddings, dim=0)
        human_embeddings = torch.cat(human_embeddings, dim=0)

        center = machine_embeddings.sum(dim=0)
        center = F.normalize(center, dim=0)

        machine_distances = torch.norm(machine_embeddings - center.unsqueeze(0), dim=1)
        human_distances = torch.norm(human_embeddings - center.unsqueeze(0), dim=1)

        radius_m = torch.quantile(machine_distances, self.q_m)
        radius_h = torch.maximum(torch.quantile(human_distances, self.q_h), radius_m)

        self.center.copy_(center)
        self.radius_m.copy_(radius_m)
        self.radius_h.copy_(radius_h)

    def _shell_loss(self, embeddings, labels):
        center = self.center.unsqueeze(0).to(embeddings.device)
        distances = torch.norm(embeddings - center, dim=1)

        machine_mask = labels == 0
        human_mask = labels == 1

        loss_m = embeddings.new_tensor(0.0)
        loss_h = embeddings.new_tensor(0.0)

        if machine_mask.any():
            loss_m = F.relu(distances[machine_mask] - self.radius_m.to(embeddings.device)).pow(2).mean()
        if human_mask.any():
            loss_h = F.relu(self.radius_h.to(embeddings.device) - distances[human_mask]).pow(2).mean()

        return loss_m + loss_h, loss_m, loss_h

    def _supcon_term(self, anchor, positives, negatives):
        if positives.size(0) == 0 or negatives.size(0) == 0:
            return anchor.new_tensor(0.0)

        pos_logits = torch.matmul(positives, anchor) / self.temperature
        neg_logits = torch.matmul(negatives, anchor) / self.temperature
        all_logits = torch.cat([pos_logits, neg_logits], dim=0)
        denominator = torch.logsumexp(all_logits, dim=0)
        return -(pos_logits - denominator).mean()

    def _contrastive_loss(self, embeddings, labels):
        machine_mask = labels == 0
        human_mask = labels == 1

        machine_embeddings = embeddings[machine_mask]
        human_embeddings = embeddings[human_mask]
        prototype = self.center.to(embeddings.device).unsqueeze(0)

        if machine_embeddings.size(0) == 0 or human_embeddings.size(0) == 0:
            return embeddings.new_tensor(0.0)

        losses = []
        for idx in range(machine_embeddings.size(0)):
            anchor = machine_embeddings[idx]
            other_machine = torch.cat([machine_embeddings[:idx], machine_embeddings[idx + 1 :]], dim=0)
            positives = torch.cat([other_machine, prototype], dim=0)
            losses.append(self._supcon_term(anchor, positives, human_embeddings))

        losses.append(self._supcon_term(prototype.squeeze(0), machine_embeddings, human_embeddings))
        return torch.stack(losses).mean()

    def forward(self, encoded_batch, labels=None):
        embeddings = self.encode(encoded_batch)
        if labels is None:
            return torch.norm(embeddings - self.center.to(embeddings.device), dim=1)

        loss_shell, loss_m, loss_h = self._shell_loss(embeddings, labels)
        loss_con = self._contrastive_loss(embeddings, labels)
        loss = self.lambda_shell * loss_shell + self.lambda_con * loss_con
        distances = torch.norm(embeddings - self.center.to(embeddings.device), dim=1)
        return loss, loss_shell, loss_m, loss_h, loss_con, distances
