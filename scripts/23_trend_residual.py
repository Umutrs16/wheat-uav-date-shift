"""Additive date trend plus a within-date spectral residual.

The phenology branch sees only day-of-year harmonics. The residual branch sees
only the five reflectance bands. A centering penalty pushes the residual mean
on each training date toward zero. Early stopping uses the tune split.
Seeds, learning rate and patience match the existing training configuration.
This fit does not overwrite the primary spectral checkpoints.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

import _path  # noqa: F401
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.metrics import cluster_bootstrap_ci, regression_metrics
from wheat_ppqmtl.paths import configure_stdio, configs


class TrendResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.trend = nn.Linear(2, 2)
        self.residual = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 2),
        )

    def forward(self, bands: torch.Tensor, doy: torch.Tensor):
        resid = self.residual(bands)
        return self.trend(doy) + resid, resid


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _fit_fold(train, tune, seed: int, device):
    torch.manual_seed(seed)
    band_cols = ["B", "G", "R", "RE", "NIR"]
    doy_cols = ["doy_sin", "doy_cos"]
    lai_train = np.isclose(pd.to_numeric(train["lai_qc"], errors="coerce").to_numpy(), 1.0)
    band_scaler = StandardScaler().fit(train[band_cols].to_numpy(float))
    doy_scaler = StandardScaler().fit(train[doy_cols].to_numpy(float))
    y_lcc = StandardScaler().fit(train[["lcc"]].to_numpy(float))
    lai_values = train.loc[lai_train, ["lai"]].to_numpy(float)
    y_lai = StandardScaler().fit(lai_values) if len(lai_values) >= 8 else None

    def pack(frame):
        bands = torch.tensor(band_scaler.transform(frame[band_cols].to_numpy(float)), dtype=torch.float32, device=device)
        doy = torch.tensor(doy_scaler.transform(frame[doy_cols].to_numpy(float)), dtype=torch.float32, device=device)
        lcc = torch.tensor(y_lcc.transform(frame[["lcc"]].to_numpy(float)).ravel(), dtype=torch.float32, device=device)
        lai_mask = torch.tensor(np.isclose(pd.to_numeric(frame["lai_qc"], errors="coerce"), 1.0), dtype=torch.bool, device=device)
        lai = np.zeros(len(frame), dtype=np.float32)
        if y_lai is not None and lai_mask.any():
            raw = frame["lai"].to_numpy(float)
            raw = np.where(np.isfinite(raw), raw, 0.0)
            lai = y_lai.transform(raw.reshape(-1, 1)).ravel().astype(np.float32)
        return bands, doy, lcc, lai_mask, torch.tensor(lai, dtype=torch.float32, device=device), frame["date_key"].to_numpy()

    bands, doy, lcc, lai_mask, lai, dates = pack(train)
    model = TrendResidual().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_state, best_score, stale = None, float("inf"), 0
    index = {day: np.flatnonzero(dates == day) for day in pd.unique(dates)}
    for _ in range(300):
        model.train()
        optimizer.zero_grad()
        pred, resid = model(bands, doy)
        loss = torch.mean((pred[:, 0] - lcc) ** 2)
        if y_lai is not None and bool(lai_mask.any()):
            loss = loss + torch.mean((pred[lai_mask, 1] - lai[lai_mask]) ** 2)
        center = []
        for idx in index.values():
            sel = torch.tensor(idx, dtype=torch.long, device=device)
            center.append(resid[sel, 0].mean() ** 2)
            lai_sel = lai_mask[sel]
            if y_lai is not None and bool(lai_sel.any()):
                center.append(resid[sel, 1][lai_sel].mean() ** 2)
        loss = loss + torch.stack(center).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            tb, td, tlcc, _, _, _ = pack(tune)
            tpred, _ = model(tb, td)
            pred_lcc = y_lcc.inverse_transform(tpred[:, 0].cpu().numpy().reshape(-1, 1)).ravel()
            score = regression_metrics(tune["lcc"], pred_lcc)["rmse"]
        if score < best_score - 1e-4:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 30:
                break
    model.load_state_dict(best_state)
    bundle = {"band_scaler": band_scaler, "doy_scaler": doy_scaler, "y_lcc": y_lcc, "y_lai": y_lai, "device": device}
    return model, bundle


def _predict_frame(model, bundle, frame):
    device = bundle["device"]
    model.eval()
    bands = torch.tensor(bundle["band_scaler"].transform(frame[["B", "G", "R", "RE", "NIR"]].to_numpy(float)), dtype=torch.float32, device=device)
    doy = torch.tensor(bundle["doy_scaler"].transform(frame[["doy_sin", "doy_cos"]].to_numpy(float)), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred, _ = model(bands, doy)
    lcc = bundle["y_lcc"].inverse_transform(pred[:, 0].cpu().numpy().reshape(-1, 1)).ravel()
    if bundle["y_lai"] is None:
        lai = np.full(len(frame), np.nan)
    else:
        lai = bundle["y_lai"].inverse_transform(pred[:, 1].cpu().numpy().reshape(-1, 1)).ravel()
    return lcc, lai


def _summarize(frame: pd.DataFrame, target: str) -> dict:
    use = frame[frame["target"] == target]
    ens = use.groupby("sample_id", as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"), date_key=("date_key", "first"), block_id=("block_id", "first")
    )
    y = ens["y_true"].to_numpy(float)
    pred = ens["y_pred"].to_numpy(float)
    dates = ens["date_key"].to_numpy()
    y_dm = np.empty_like(y)
    p_dm = np.empty_like(pred)
    for day in pd.unique(dates):
        mask = dates == day
        y_dm[mask] = y[mask] - y[mask].mean()
        p_dm[mask] = pred[mask] - pred[mask].mean()
    stats = regression_metrics(y, pred)
    ci = cluster_bootstrap_ci(y, pred, ens["block_id"].to_numpy(), n_boot=1000, seed=42)
    stats["rmse_ci95"] = ci["rmse_ci95"]
    stats["within_rmse"] = float(np.sqrt(np.mean((y_dm - p_dm) ** 2)))
    return stats


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    features = read_table(root / "data" / "features" / "features.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    summary = {}
    for scheme in ("grouped_random", "spatial_block", "lodo"):
        splits = read_table(root / "splits" / f"{scheme}.csv")
        rows = []
        for fold in sorted(splits["outer_fold"].unique()):
            merged = splits[splits["outer_fold"] == fold].merge(features, on="sample_id", suffixes=("", "_f"))
            train = merged[merged["role"] == "train"]
            tune = merged[merged["role"] == "tune"]
            test = merged[merged["role"] == "test"]
            if len(train) < 8 or len(tune) < 2 or len(test) == 0:
                continue
            for seed in (42, 43, 44, 45, 46):
                print(f"{scheme} fold {int(fold)} seed {seed}", flush=True)
                model, bundle = _fit_fold(train, tune, seed, device)
                lcc, lai = _predict_frame(model, bundle, test)
                qc = np.isclose(pd.to_numeric(test["lai_qc"], errors="coerce").to_numpy(), 1.0)
                for sample_id, date_key, block_id, y_lcc, y_lai, p_lcc, p_lai, keep_lai in zip(
                    test["sample_id"], test["date_key"], test["block_id"], test["lcc"], test["lai"], lcc, lai, qc
                ):
                    rows.append({"scheme": scheme, "outer_fold": int(fold), "seed": seed, "sample_id": sample_id,
                                 "date_key": date_key, "block_id": block_id, "target": "lcc", "y_true": float(y_lcc), "y_pred": float(p_lcc)})
                    if keep_lai and np.isfinite(y_lai):
                        rows.append({"scheme": scheme, "outer_fold": int(fold), "seed": seed, "sample_id": sample_id,
                                     "date_key": date_key, "block_id": block_id, "target": "lai", "y_true": float(y_lai), "y_pred": float(p_lai)})
        frame = pd.DataFrame(rows)
        frame.to_csv(root / "outputs" / "predictions" / f"trend_residual__{scheme}.csv", index=False)
        summary[scheme] = {"lcc": _summarize(frame, "lcc"), "lai": _summarize(frame, "lai")}
    dest = root / "outputs" / "metrics" / "trend_residual.json"
    dest.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
