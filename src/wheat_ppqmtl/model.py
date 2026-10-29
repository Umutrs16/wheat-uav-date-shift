from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.features(x).flatten(1))


class PhysicsEncoder(nn.Module):
    def __init__(self, in_dim: int = 5, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.act(self.fc3(x))


class PhenologyEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPQMTL(nn.Module):
    def __init__(
        self,
        pheno_dim: int,
        use_cnn: bool = False,
        use_phenology: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_cnn = use_cnn
        self.use_phenology = use_phenology
        self.cnn = SmallCNN(dropout) if use_cnn else None
        self.physics = PhysicsEncoder()
        self.phenology = PhenologyEncoder(pheno_dim) if use_phenology else None
        fused = 64
        if use_cnn:
            fused += 128
        if use_phenology:
            fused += 32
        self.fuse = nn.Sequential(
            nn.Linear(fused, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lcc = nn.Linear(64, 2)
        self.lai = nn.Linear(64, 2)
        self.ccc = nn.Linear(64, 2)

    def forward(self, x_phys: torch.Tensor, x_pheno: torch.Tensor | None = None, patch: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        parts = [self.physics(x_phys)]
        if self.use_cnn:
            if patch is None:
                raise ValueError("CNN branch requested but no patch was provided")
            parts.append(self.cnn(patch))
        if self.use_phenology:
            if x_pheno is None:
                raise ValueError("Phenology branch requested but no phenology features were provided")
            parts.append(self.phenology(x_pheno))
        hidden = self.fuse(torch.cat(parts, dim=-1))
        lcc = self.lcc(hidden)
        lai = self.lai(hidden)
        ccc = self.ccc(hidden)
        return {
            "mu_lcc": lcc[:, 0],
            "logvar_lcc": lcc[:, 1],
            "mu_lai": lai[:, 0],
            "logvar_lai": lai[:, 1],
            "mu_ccc": ccc[:, 0],
            "logvar_ccc": ccc[:, 1],
        }


class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
