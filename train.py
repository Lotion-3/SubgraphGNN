"""
TrimNN training script — autoresearcher framework.

Trains a GNN to predict subgraph match counts (TrimNN task).
Ground truth is VF2 exact counting on the spatial cell-type graph.
Metric: val_mse — MSE between predicted and exact counts (lower is better).

Usage: python train.py
This file is the only file the agent should modify.
"""

import os
import gc
import math
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TIME_BUDGET, N_LABELS, MAX_GRAPH_NODES, MAX_PATTERN_NODES,
    generate_data, split_data, make_dataloader, evaluate_val_mse,
)

# ---------------------------------------------------------------------------
# Hyperparameters  (agent: edit freely)
# ---------------------------------------------------------------------------

HIDDEN_DIM    = 128      # GNN hidden dimension
NUM_LAYERS    = 3        # number of GNN message-passing layers
DROPOUT       = 0.0      # no dropout — model benefits from full capacity
LR            = 1e-3     # learning rate
WEIGHT_DECAY  = 1e-5     # Adam weight decay
BATCH_SIZE    = 256      # training batch size
MAX_GRAD_NORM = 8.0      # gradient clipping
WARMUP_STEPS  = 50       # linear LR warmup

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GNNLayer(nn.Module):
    """One message-passing step: aggregate neighbours, then MLP."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim)
        self.norm   = nn.LayerNorm(dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        # x: (B, N, D)  adj_norm: (B, N, N) row-normalised
        msg = torch.bmm(adj_norm, x)
        h   = torch.cat([x, msg], dim=-1)
        return x + self.drop(self.norm(F.gelu(self.linear(h))))   # residual


class SubgraphGNN(nn.Module):
    """
    Encodes a (pattern, graph) pair with shared GNN weights, then predicts
    the number of times the pattern appears as a subgraph.

    Inputs (all batched):
        p_adj  (B, Np, Np)  — pattern adjacency (padded)
        p_feat (B, Np, L)   — pattern node one-hot features
        p_mask (B, Np)      — True for real pattern nodes
        g_adj  (B, Ng, Ng)  — graph adjacency (padded)
        g_feat (B, Ng, L)   — graph node one-hot features
        g_mask (B, Ng)      — True for real graph nodes
    Output:
        pred (B, 1)         — predicted occurrence count (non-negative)
    """

    def __init__(self, n_labels, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed = nn.Linear(n_labels, hidden_dim, bias=False)
        self.layers = nn.ModuleList(
            [GNNLayer(hidden_dim, dropout) for _ in range(num_layers)]
        )
        # Bidirectional cross-attention with pre-norm
        self.cross_attn_p2g = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4,
            dropout=dropout, batch_first=True,
        )
        self.cross_attn_g2p = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4,
            dropout=dropout, batch_first=True,
        )
        self.norm_p   = nn.LayerNorm(hidden_dim)
        self.norm_g   = nn.LayerNorm(hidden_dim)
        self.gate_g   = nn.Linear(hidden_dim, hidden_dim)   # vector gate per graph node
        nn.init.constant_(self.gate_g.bias, -2.0)          # start sparse (gates near 0.12)
        self.tri_embed = nn.Linear(1, hidden_dim, bias=False)  # map structural triangle count → H
        self.predict  = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),   # p_max + g_max + g_gated_sum
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),   # smooth non-negative output
        )

    @staticmethod
    def _norm_adj(adj):
        """Symmetric row normalisation, avoiding division by zero."""
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return adj / deg

    def _encode(self, adj, feat):
        adj_n = self._norm_adj(adj)
        x = F.gelu(self.embed(feat))
        for layer in self.layers:
            x = layer(x, adj_n)
        return x

    def forward(self, p_adj, p_feat, p_mask, g_adj, g_feat, g_mask):
        p_enc = self._encode(p_adj, p_feat)   # (B, Np, H)
        g_enc = self._encode(g_adj, g_feat)   # (B, Ng, H)
        # Augment graph enc with structural triangle count per node
        A2_g  = torch.bmm(g_adj, g_adj)                              # (B, Ng, Ng) 2-hop paths
        tri_c = (A2_g * g_adj).sum(dim=-1, keepdim=True) / 2        # (B, Ng, 1) triangles per node
        g_enc = g_enc + self.tri_embed(torch.log1p(tri_c))             # (B, Ng, H) log-scale embed

        p_mask_f = p_mask.unsqueeze(-1).float()  # (B, Np, 1)
        g_mask_f = g_mask.unsqueeze(-1).float()

        g_key_mask = ~g_mask
        p_key_mask = ~p_mask

        # Pre-norm + residual cross-attention
        p_attn, _ = self.cross_attn_p2g(
            self.norm_p(p_enc), self.norm_g(g_enc), self.norm_g(g_enc),
            key_padding_mask=g_key_mask,
        )
        p_out = p_enc + p_attn                        # (B, Np, H)

        g_attn, _ = self.cross_attn_g2p(
            self.norm_g(g_enc), self.norm_p(p_enc), self.norm_p(p_enc),
            key_padding_mask=p_key_mask,
        )
        g_out = g_enc + g_attn                        # (B, Ng, H)

        p_pool    = (p_out * p_mask_f).max(dim=1).values            # (B, H)
        g_pool    = (g_out * g_mask_f).max(dim=1).values            # (B, H)
        gate      = torch.sigmoid(self.gate_g(g_out))               # (B, Ng, H) pattern-cond.
        g_gated   = (gate * g_enc * g_mask_f).sum(dim=1)            # (B, H) structural count

        return self.predict(torch.cat([p_pool, g_pool, g_gated], dim=-1))  # (B, 1)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
device = torch.device("cpu")

print("Loading data ...")
all_data = generate_data()
train_data, val_data = split_data(all_data)
print(f"Train: {len(train_data)}  Val: {len(val_data)}")

train_loader = make_dataloader(train_data, BATCH_SIZE, shuffle=True)
val_loader   = make_dataloader(val_data,   BATCH_SIZE, shuffle=False)

model     = SubgraphGNN(N_LABELS, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    t = step - WARMUP_STEPS
    total = 3800 - WARMUP_STEPS   # single cosine cycle over full budget
    # Single cycle: 1.0 → 0.01 (no restart, smooth continuous descent)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(t / total, 1.0)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training  = time.time()
total_train_time  = 0.0
step              = 0
smooth_loss       = None
train_iter        = iter(train_loader)

while True:
    model.train()
    t0 = time.time()

    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    p_adj, p_feat, p_mask, g_adj, g_feat, g_mask, counts = [
        x.to(device) for x in batch
    ]

    pred = model(p_adj, p_feat, p_mask, g_adj, g_feat, g_mask)
    loss = F.mse_loss(pred.view(-1), counts.view(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    scheduler.step()

    dt = time.time() - t0

    # Skip first few steps from timing (JIT / cache warm-up)
    if step >= 5:
        total_train_time += dt

    loss_f = loss.item()
    ema    = 0.95
    smooth_loss = (loss_f if smooth_loss is None
                   else ema * smooth_loss + (1 - ema) * loss_f)

    progress  = min(total_train_time / TIME_BUDGET, 1.0)
    remaining = max(0.0, TIME_BUDGET - total_train_time)
    print(
        f"\rstep {step:05d} ({100*progress:.1f}%) | "
        f"loss: {smooth_loss:.4f} | "
        f"dt: {dt*1000:.0f}ms | "
        f"remaining: {remaining:.0f}s   ",
        end="", flush=True,
    )

    if step == 0:
        gc.collect()

    step += 1

    if step > 5 and total_train_time >= TIME_BUDGET:
        break

print()   # newline after \r

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
val_mse = evaluate_val_mse(model, val_loader, device)

t_end        = time.time()
total_seconds = t_end - t_start

print("---")
print(f"val_mse:          {val_mse:.6f}")
print(f"training_seconds: {total_train_time:.1f}")
print(f"total_seconds:    {total_seconds:.1f}")
print(f"peak_ram_mb:      0.0")
print(f"num_steps:        {step}")
print(f"num_params_M:     {n_params / 1e6:.2f}")
print(f"hidden_dim:       {HIDDEN_DIM}")
print(f"num_layers:       {NUM_LAYERS}")
