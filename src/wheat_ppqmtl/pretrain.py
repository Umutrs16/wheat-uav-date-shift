from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wheat_ppqmtl.io_utils import read_table, write_json
from wheat_ppqmtl.model import PhysicsEncoder
from wheat_ppqmtl.paths import configure_stdio, configs
from wheat_ppqmtl.scaling import RobustScale


def _run() -> dict:
    cfg = configs()
    table = read_table(cfg["root"] / "data" / "synthetic" / "prosail.csv")
    features = ["B", "G", "R", "RE", "NIR"]
    targets = ["cab", "lai", "ccc"]
    train = table[table["sim_split"] == "train"]
    val = table[table["sim_split"] == "val"]
    x_scaler = RobustScale().fit(train[features].to_numpy(float))
    y_scaler = RobustScale().fit(train[targets].to_numpy(float))
    x_train = torch.tensor(x_scaler.transform(train[features].to_numpy(float)), dtype=torch.float32)
    y_train = torch.tensor(y_scaler.transform(train[targets].to_numpy(float)), dtype=torch.float32)
    x_val = torch.tensor(x_scaler.transform(val[features].to_numpy(float)), dtype=torch.float32)
    y_val = torch.tensor(y_scaler.transform(val[targets].to_numpy(float)), dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = PhysicsEncoder(in_dim=len(features)).to(device)
    head = nn.Linear(64, 3).to(device)
    pre_cfg = cfg["prosail"]["pretrain"]
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=float(pre_cfg["lr"]),
        weight_decay=float(pre_cfg["weight_decay"]),
    )
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=int(pre_cfg["batch_size"]), shuffle=True)
    best = None
    best_loss = float("inf")
    patience = 8
    stale = 0
    for epoch in range(int(pre_cfg["epochs"])):
        encoder.train()
        head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(head(encoder(xb)), yb)
            loss.backward()
            optimizer.step()
        encoder.eval()
        head.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.mse_loss(head(encoder(x_val.to(device))), y_val.to(device)).cpu())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
        if (epoch + 1) % 5 == 0:
            print(f"pretrain epoch {epoch + 1} val_mse={val_loss:.4f}")
    encoder.load_state_dict(best)
    dest = cfg["root"] / "models" / "physics_encoder.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder": encoder.state_dict(),
        "input_scaler": x_scaler.to_dict(),
        "target_scaler": y_scaler.to_dict(),
        "features": features,
        "targets": targets,
        "val_mse_standardized": best_loss,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
    }
    torch.save(payload, dest)
    # Report original-unit RMSE on the synthetic validation split. This is not a field result.
    with torch.no_grad():
        pred_z = head(encoder(x_val.to(device))).cpu().numpy()
    pred = y_scaler.inverse(pred_z)
    observed = val[targets].to_numpy(float)
    rmse = np.sqrt(np.mean((pred - observed) ** 2, axis=0))
    report = {
        "checkpoint": str(dest),
        "val_mse_standardized": best_loss,
        "val_rmse_original": {name: float(value) for name, value in zip(targets, rmse)},
        "n_simulated": int(len(table)),
    }
    write_json(cfg["root"] / "models" / "physics_pretrain_report.json", report)
    print(report["val_rmse_original"])
    return report


def main() -> None:
    configure_stdio()
    argparse.ArgumentParser(description="Pretrain the spectral physics encoder on PROSAIL samples.").parse_args()
    _run()


if __name__ == "__main__":
    main()
