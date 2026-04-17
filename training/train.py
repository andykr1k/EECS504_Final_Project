import argparse
import logging
from pathlib import Path
from typing import List, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
from tqdm import tqdm
import wandb
import numpy as np
import random
from collections import defaultdict
from pathvalidate import sanitize_filename
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

from sam3_reid_dataset import Sam3ReIDDataset, VideoSlicePKBatchSampler, DinoDataLoaderWrapper, ApplyBackgroundMask, RandomSubjectZoom

# ==========================================
# 1. Losses
# ==========================================

class CircleLoss(nn.Module):
    def __init__(self, m: float = 0.25, gamma: float = 256):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
        """
        sp: Positive similarities (N_pos,)
        sn: Negative similarities (N_neg,)
        """
        # Calculate weighting factors alpha
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        # Calculate margins delta
        delta_p = 1 - self.m
        delta_n = self.m

        # Calculate logits
        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        # LogSumExp approximation for multi-positive and multi-negative
        loss = self.soft_plus(
            torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)
        )
        return loss

def compute_circle_loss(embeddings: torch.Tensor, labels: torch.Tensor, criterion: CircleLoss) -> torch.Tensor:
    """
    Given a batch of embeddings and labels, computes pairwise similarities and applies Circle Loss.
    """
    # Normalize embeddings to calculate cosine similarity via dot product
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix (B, B)
    sim_matrix = torch.matmul(embeddings, embeddings.t())
    
    # Create masks for positives and negatives
    labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    identity_mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    
    pos_mask = labels_matrix & ~identity_mask # Same label, but not the exact same image
    neg_mask = ~labels_matrix                 # Different labels
    
    # Iterate over anchors to compute loss per anchor
    losses = []
    for i in range(embeddings.size(0)):
        sp = sim_matrix[i][pos_mask[i]]
        sn = sim_matrix[i][neg_mask[i]]
        
        # Only compute if we have both positives and negatives for this anchor
        if sp.numel() > 0 and sn.numel() > 0:
            losses.append(criterion(sp, sn))
            
    if not losses:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
    return torch.mean(torch.stack(losses))

def compute_batch_metrics(embeddings: torch.Tensor, labels: torch.Tensor, m: float = 0.25) -> dict:
    """
    Computes within-batch evaluation metrics.
    """
    # 1. Compute cosine similarity matrix
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.t())
    
    B = labels.size(0)
    labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    identity_mask = torch.eye(B, dtype=torch.bool, device=labels.device)
    
    pos_mask = labels_matrix & ~identity_mask
    neg_mask = ~labels_matrix
    
    # Extract positive and negative similarities
    sp = sim_matrix[pos_mask]
    sn = sim_matrix[neg_mask]
    
    # --- METRICS: Margins & Distributions ---
    delta_p = 1 - m
    delta_n = m
    
    pct_pos_margin = (sp > delta_p).float().mean().item() if sp.numel() > 0 else 0.0
    pct_neg_margin = (sn < delta_n).float().mean().item() if sn.numel() > 0 else 0.0
    
    mean_pos_sim = sp.mean().item() if sp.numel() > 0 else 0.0
    mean_neg_sim = sn.mean().item() if sn.numel() > 0 else 0.0
    
    # --- METRICS: Retrieval (Top-K & mAP) ---
    # Ignore self-similarity by setting the diagonal to -infinity
    sim_matrix.fill_diagonal_(float('-inf'))
    
    # Sort similarities descending to get retrieval rankings
    sorted_sim, sorted_indices = torch.sort(sim_matrix, dim=1, descending=True)
    sorted_labels = labels[sorted_indices]
    
    # Boolean mask of matches (True if retrieved ID matches query ID)
    matches = sorted_labels == labels.unsqueeze(1)
    
    # Top-1 Accuracy: Is the #1 closest embedding a match?
    top1 = matches[:, 0].float().mean().item()
    
    # Top-5 Accuracy: Is there *any* match in the top 5 closest embeddings?
    k = min(5, B - 1)
    top5 = matches[:, :k].any(dim=1).float().mean().item()
    
    # Batch mAP (Mean Average Precision)
    num_positives = pos_mask.sum(dim=1)
    aps = []
    for i in range(B):
        if num_positives[i] == 0:
            continue
        
        # Get indices of positive matches in the sorted array
        query_matches = matches[i]
        pos_indices = torch.nonzero(query_matches).squeeze(1)
        
        # Rank of each positive match (1-based indexing)
        ranks = pos_indices + 1
        
        # Precision at each positive rank
        # e.g. if matches are at rank 1, 3, 4 -> precisions are 1/1, 2/3, 3/4
        arange = torch.arange(1, len(ranks) + 1, device=ranks.device)
        precisions = arange / ranks.float()
        
        ap = precisions.sum() / num_positives[i].float()
        aps.append(ap.item())
        
    mAP = sum(aps) / len(aps) if aps else 0.0
    
    return {
        "pct_pos_margin": pct_pos_margin,
        "pct_neg_margin": pct_neg_margin,
        "mean_pos_sim": mean_pos_sim,
        "mean_neg_sim": mean_neg_sim,
        "top1": top1,
        "top5": top5,
        "mAP": mAP
    }

def log_umap_visualization(embeddings: torch.Tensor, labels: torch.Tensor, epoch: int):
    """
    Reduces embeddings to 2D using PCA then UMAP, and logs a scatter plot to W&B.
    """
    # Move to CPU and convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # 1. PCA down to 50 dimensions (or max possible if less than 50 samples/dims)
    n_pca_components = min(50, embeddings_np.shape[0], embeddings_np.shape[1])
    if embeddings_np.shape[1] > n_pca_components:
        pca = PCA(n_components=n_pca_components)
        embeddings_reduced = pca.fit_transform(embeddings_np)
    else:
        embeddings_reduced = embeddings_np
        
    # 2. UMAP down to 2 dimensions
    # random_state ensures reproducibility across epochs if the data order is the same
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings_reduced)
    
    # 3. Plotting
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels_np)
    
    # Use a colormap with enough discrete colors (tab20 has 20 distinct colors)
    cmap = plt.get_cmap('tab20')
    
    for i, label in enumerate(unique_labels):
        idx = labels_np == label
        # Pick a color based on the label index, cycling if there are >20 classes
        color = cmap(i % 20)
        plt.scatter(
            embeddings_2d[idx, 0], 
            embeddings_2d[idx, 1], 
            label=f"ID: {label}", 
            color=color, 
            alpha=0.7, 
            s=30, # Marker size
            edgecolors='w',
            linewidth=0.5
        )
        
    plt.title(f"UMAP Projection of Validation Embeddings (Epoch {epoch})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    
    # Only show legend if we have a reasonable number of identities (e.g., <= 20)
    # Otherwise, it takes up the whole plot.
    if len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=1.5)
        
    plt.tight_layout()
    
    # 4. Log to wandb
    wandb.log({f"Val Embeddings UMAP": wandb.Image(plt)}, commit=False)
    
    # Free memory
    plt.close()

# ==========================================
# 2. Model Architecture
# ==========================================

def build_mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int) -> nn.Module:
    """Helper to build a configurable MLP"""
    if num_layers == 0:
        return nn.Linear(in_dim, out_dim)
        
    layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
    layers.append(nn.Linear(hidden_dim, out_dim))
    
    return nn.Sequential(*layers)

class ReIDModel(nn.Module):
    def __init__(
        self, 
        dino_dim: int = 384, 
        proj_dim: int = 256, 
        contrastive_dim: int = 256,
        hidden_dim: int = 512, 
        num_layers: int = 2,
        pool_type: Literal['average', 'attention'] = 'attention',
        proj_type: Literal['identity', 'mlp'] = 'mlp'
    ):
        super().__init__()
        self.pool_type = pool_type
        self.proj_type = proj_type
        
        # 1. Embedding Projection
        self.input_dim = dino_dim
        if self.proj_type == 'mlp':
            self.emb_proj = build_mlp(dino_dim, proj_dim, hidden_dim, num_layers)
            self.pool_in_dim = proj_dim
        else:
            self.emb_proj = nn.Identity()
            self.pool_in_dim = dino_dim
            
        # 2. Pooling
        if self.pool_type == 'attention':
            self.attention_net = build_mlp(self.pool_in_dim, 1, hidden_dim, num_layers=1)
            
        # 3. Contrastive Projection
        self.contrastive_proj = build_mlp(self.pool_in_dim, contrastive_dim, hidden_dim, num_layers)

    def forward(self, embeddings_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings_list: A list of length B. Each element is a tensor of shape (N_i, D)
                             where N_i is the variable number of embeddings for that image.
        Returns:
            contrastive_embeddings: Tensor of shape (B, contrastive_dim)
        """
        device = embeddings_list[0].device
        B = len(embeddings_list)
        
        # Get sequence lengths to create attention/pooling mask
        lengths = torch.tensor([e.size(0) for e in embeddings_list], device=device)
        max_len = lengths.max()
        
        # Pad the sequences: (B, max_len, D)
        padded_embs = pad_sequence(embeddings_list, batch_first=True)
        
        # Create boolean mask: (B, max_len). True where elements are valid.
        mask = torch.arange(max_len, device=device).expand(B, max_len) < lengths.unsqueeze(1)
        
        # 1. Project individual embeddings
        projected_embs = self.emb_proj(padded_embs) # (B, max_len, pool_in_dim)
        
        # 2. Pool embeddings across the N dimension
        if self.pool_type == 'attention':
            # Compute attention weights: (B, max_len, 1)
            attn_logits = self.attention_net(projected_embs).squeeze(-1) # (B, max_len)
            # Mask out padding with -inf so softmax evaluates to 0
            attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_logits, dim=1) # (B, max_len)
            
            # Weighted sum: (B, pool_in_dim)
            pooled_emb = torch.bmm(attn_weights.unsqueeze(1), projected_embs).squeeze(1)
            
        elif self.pool_type == 'average':
            # Zero out padding explicitly just to be safe
            projected_embs = projected_embs.masked_fill(~mask.unsqueeze(-1), 0.0)
            # Sum and divide by actual length
            summed = projected_embs.sum(dim=1) # (B, pool_in_dim)
            pooled_emb = summed / lengths.unsqueeze(-1).float() # (B, pool_in_dim)
            
        # 3. Final projection into contrastive space
        final_embs = self.contrastive_proj(pooled_emb) # (B, contrastive_dim)
        return final_embs

class ReIDTransformerModel(nn.Module):
    def __init__(
        self, 
        dino_dim: int = 384, 
        transformer_dim: int = 256, 
        contrastive_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 1. Project variable DINO dims to a consistent Transformer dimension
        self.input_proj = nn.Linear(dino_dim, transformer_dim)
        
        # 2. Learnable [CLS] token for pooling the sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True # Crucial: Expects input as (Batch, Seq, Feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final Projection to Contrastive Space
        self.contrastive_proj = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, contrastive_dim)
        )

    def forward(self, embeddings_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings_list: A list of length B. Each element is (N_i, D).
        Returns:
            contrastive_embeddings: Tensor of shape (B, contrastive_dim)
        """
        device = embeddings_list[0].device
        B = len(embeddings_list)
        
        # Get sequence lengths
        lengths = torch.tensor([e.size(0) for e in embeddings_list], device=device)
        max_len = lengths.max()
        
        # Pad sequences: (B, max_len, D)
        padded_embs = pad_sequence(embeddings_list, batch_first=True)
        
        # Project inputs
        x = self.input_proj(padded_embs) # (B, max_len, transformer_dim)
        
        # Prepend [CLS] token to every sequence in the batch
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, transformer_dim)
        x = torch.cat((cls_tokens, x), dim=1)         # (B, max_len + 1, transformer_dim)
        
        # Create padding mask for PyTorch's Transformer
        # PyTorch expects True for positions that are PADDING and should be IGNORED
        mask = torch.arange(max_len, device=device).expand(B, max_len) >= lengths.unsqueeze(1)
        
        # The [CLS] token at index 0 is always valid, so we prepend False
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
        padding_mask = torch.cat((cls_mask, mask), dim=1) # (B, max_len + 1)
        
        # Pass through Transformer
        out = self.transformer(x, src_key_padding_mask=padding_mask) # (B, max_len + 1, transformer_dim)
        
        # Extract the state of the [CLS] token
        cls_out = out[:, 0, :] # (B, transformer_dim)
        
        # Final projection into contrastive space
        final_embs = self.contrastive_proj(cls_out) # (B, contrastive_dim)
        
        return final_embs

# ==========================================
# 3. Training Loop
# ==========================================

def get_dataloaders(args):
    # Common transforms
    # transform = v2.Compose([
    #     v2.RandomHorizontalFlip(p=0.5),
    #     v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # ])
    transform = v2.Compose([
        # --- 1. Safe Spatial Transforms ---
        v2.RandomHorizontalFlip(p=0.5),

        v2.RandomApply([RandomSubjectZoom(scale_range=(1.0, 2.0))], p=0.25),
        
        # Slight rotation (±5 degrees), translation (±5%), and scaling (95% to 105%)
        # The mask will perfectly track with these changes.
        v2.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),

        # --- 2. Color and Lighting (Photometric) ---
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # v2.RandomGrayscale(p=0.1),
        
        # Randomly apply Gaussian Blur to 10% of images to simulate poor focus
        v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))], p=0.1),

        v2.RandomApply([ApplyBackgroundMask(bg_val=0.0)], p=0.5),
    ])
    only_mask_transform = v2.Compose([
        v2.RandomApply([ApplyBackgroundMask(bg_val=0.0)], p=1),
    ])

    train_dataset = Sam3ReIDDataset(root_dir=args.train_dir, transform=None)
    val_dataset = Sam3ReIDDataset(root_dir=args.val_dir, transform=None)

    train_sampler = VideoSlicePKBatchSampler(
        dataset=train_dataset, num_identities=args.P, instances_per_identity=args.K
    )
    val_sampler = VideoSlicePKBatchSampler(
        dataset=val_dataset, num_identities=args.P, instances_per_identity=args.K,
        seed=args.seed, epoch_deterministic=True
    )

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    dino_train = DinoDataLoaderWrapper(train_loader, transform=transform, device=args.device, checkpoint=args.dino_checkpoint)
    # Share DINO harness to save VRAM
    dino_val = DinoDataLoaderWrapper(val_loader, transform=None, device=args.device, dino_harness=dino_train.dino_harness)

    return dino_train, dino_val

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, max_batches=None):
    model.train()
    total_loss = 0.0
    
    num_batches = min(len(dataloader), max_batches) if max_batches is not None else len(dataloader)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Train", total=num_batches)
    for batch_idx, batch in enumerate(pbar):
        if batch_idx >= num_batches:
            break

        labels = torch.stack(batch["class_id"]).to(device)
        embeddings_list = [emb.to(device) for emb in batch["embeddings"]]
        
        optimizer.zero_grad()
        
        # Forward pass
        final_embeddings = model(embeddings_list)
        
        # Compute loss
        loss = compute_circle_loss(final_embeddings, labels, criterion)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        wandb.log({"train_step_loss": loss.item()})
        
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, args, max_batches=None):
    model.eval()
    total_loss = 0.0
    metrics_accumulator = defaultdict(float)
    
    # --- NEW: Accumulators for visualization ---
    all_embeddings = []
    all_labels = []
    
    num_batches = min(len(dataloader), max_batches) if max_batches is not None else len(dataloader)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Val", total=num_batches)
    
    for batch_idx, batch in enumerate(pbar):
        if batch_idx >= num_batches:
            break

        labels = torch.stack(batch["class_id"]).to(device)
        embeddings_list = [emb.to(device) for emb in batch["embeddings"]]
        
        final_embeddings = model(embeddings_list)
        loss = compute_circle_loss(final_embeddings, labels, criterion)
        total_loss += loss.item()
        
        # --- NEW: Save embeddings and labels for this batch ---
        all_embeddings.append(final_embeddings.detach().cpu())
        all_labels.append(labels.detach().cpu())
        
        batch_metrics = compute_batch_metrics(final_embeddings, labels, m=args.circle_margin)
        for k, v in batch_metrics.items():
            metrics_accumulator[k] += v
            
        pbar.set_postfix({
            'val_loss': loss.item(), 
            'top1': f"{batch_metrics['top1']:.2f}"
        })
        
    avg_loss = total_loss / num_batches
    
    # Average the accumulated metrics
    epoch_metrics = {f"val_{k}": v / num_batches for k, v in metrics_accumulator.items()}
    epoch_metrics["val_epoch_loss"] = avg_loss
    
    wandb.log(epoch_metrics)
    
    # --- NEW: Trigger UMAP Visualization ---
    if all_embeddings:
        full_embeddings_tensor = torch.cat(all_embeddings, dim=0)
        full_labels_tensor = torch.cat(all_labels, dim=0)
        log_umap_visualization(full_embeddings_tensor, full_labels_tensor, epoch)
        
    return avg_loss

def main(args):
    wandb.init(project="sam3-reid", config=vars(args), name=args.run_name)
    
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # Build Dataloaders
    train_loader, val_loader = get_dataloaders(args)

    # Initialize Model
    if args.model_type == "mlp":
        logging.info("Initializing MLP-based ReIDModel...")
        model = ReIDModel(
            dino_dim=args.dino_dim,
            proj_dim=args.proj_dim,
            contrastive_dim=args.contrastive_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            pool_type=args.pool_type,
            proj_type=args.proj_type
        ).to(device)
    elif args.model_type == "transformer":
        logging.info("Initializing Transformer-based ReIDModel...")
        model = ReIDTransformerModel(
            dino_dim=args.dino_dim,
            transformer_dim=args.transformer_dim,
            contrastive_dim=args.contrastive_dim,
            num_heads=args.transformer_heads,
            num_layers=args.transformer_layers,
            dropout=args.transformer_dropout
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    
    wandb.watch(model)

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CircleLoss(m=args.circle_margin, gamma=args.circle_gamma)

    # Training Loop
    initial_val_loss = validate(model, val_loader, criterion, device, epoch=0, args=args, max_batches=args.val_max_batches)
    wandb.log({
        "epoch": 0,
        "val_loss": initial_val_loss,
        "lr": scheduler.get_last_lr()[0]
    })
    
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device, epoch=epoch, args=args, max_batches=args.val_max_batches)
        # val_loss = 0
        scheduler.step()
        
        wandb.log({
            "epoch": epoch, 
            "train_loss": train_loss, 
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0]
        })
        
        # Simple checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = Path(args.checkpoint_dir) / "best_model.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved new best model with val loss: {best_val_loss:.4f}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ReID Embedding Model")
    
    # Data params
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation dataset")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    # Sampler PK params
    parser.add_argument("--P", type=int, default=16, help="Identities per batch")
    parser.add_argument("--K", type=int, default=4, help="Instances per identity")
    
    # --- Global Model Params ---
    parser.add_argument("--model_type", type=str, choices=["mlp", "transformer"], default="transformer", help="Which aggregation architecture to use")
    parser.add_argument("--dino_checkpoint", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")    # facebook/dinov3-vitl16-pretrain-lvd1689m
    parser.add_argument("--dino_dim", type=int, default=384, help="DINO embedding dimension")
    parser.add_argument("--contrastive_dim", type=int, default=256, help="Final contrastive space dimension")
    
    # --- MLP Specific Params ---
    parser.add_argument("--proj_dim", type=int, default=256, help="Intermediate projection dimension (MLP only)")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dim for MLPs (MLP only)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of hidden layers in MLPs (MLP only)")
    parser.add_argument("--pool_type", type=str, choices=["average", "attention"], default="attention", help="Pooling method (MLP only)")
    parser.add_argument("--proj_type", type=str, choices=["identity", "mlp"], default="mlp", help="Projection type (MLP only)")
    
    # --- Transformer Specific Params ---
    parser.add_argument("--transformer_dim", type=int, default=256, help="Internal dimension of the transformer")
    parser.add_argument("--transformer_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--transformer_layers", type=int, default=2, help="Number of transformer encoder layers")
    parser.add_argument("--transformer_dropout", type=float, default=0.1, help="Dropout probability in transformer")
    
    # Loss & Opt params
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--circle_margin", type=float, default=0.25)
    parser.add_argument("--circle_gamma", type=float, default=256.0)

    parser.add_argument("--val_max_batches", type=int, default=100, help="Maximum number of batches to use for validation")
    parser.add_argument("--train_max_batches", type=int, default=None, help="Maximum number of batches to use for training")
    
    # Misc
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = input("Enter a run name: ")
        assert args.run_name, "Run name cannot be empty"

    if args.checkpoint_dir is None:
        safe_run_name = sanitize_filename(args.run_name)
        safe_run_name = safe_run_name.replace(" ", "_")
        args.checkpoint_dir = f"./checkpoints/{safe_run_name}"

    # Set all our seeds
    print(f"Using seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args)