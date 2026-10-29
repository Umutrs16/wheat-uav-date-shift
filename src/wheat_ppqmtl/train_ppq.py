from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from wheat_ppqmtl.indices import PHENOLOGY_COLUMNS, SPECTRAL, VI_COLUMNS
from wheat_ppqmtl.io_utils import read_json, read_table
from wheat_ppqmtl.losses import date_level_lai_loss, multitask_loss
from wheat_ppqmtl.metrics import regression_metrics
from wheat_ppqmtl.model import PPQMTL
from wheat_ppqmtl.paths import configure_stdio, configs
from wheat_ppqmtl.scaling import RobustScale


class FoldSet(Dataset):
    def __init__(self, phys, pheno, y, mask, qc, patch, augment, phys_center, phys_scale, patch_mean, patch_std):
        self.phys = np.asarray(phys, dtype=np.float32)
        self.pheno = None if pheno is None else np.asarray(pheno, dtype=np.float32)
        self.y = {key: np.asarray(value, dtype=np.float32) for key, value in y.items()}
        self.mask = {key: np.asarray(value, dtype=np.float32) for key, value in mask.items()}
        self.qc = np.asarray(qc, dtype=np.float32)
        self.patch = None if patch is None else np.asarray(patch, dtype=np.float32)
        self.augment = augment
        self.phys_center = np.asarray(phys_center, dtype=np.float32)
        self.phys_scale = np.asarray(phys_scale, dtype=np.float32)
        self.patch_mean = None if patch_mean is None else np.asarray(patch_mean, dtype=np.float32)
        self.patch_std = None if patch_std is None else np.asarray(patch_std, dtype=np.float32)

    def __len__(self):
        return len(self.phys)

    def __getitem__(self, index):
        raw = self.phys[index] * self.phys_scale + self.phys_center
        if self.augment:
            raw = raw.copy()
            raw[:5] = raw[:5] * (1.0 + np.random.uniform(-0.02, 0.02, size=5))
            raw[:5] = raw[:5] + np.random.normal(0.0, 0.005, size=5)
        phys = (raw - self.phys_center) / self.phys_scale
        item = {
            "phys": torch.tensor(phys, dtype=torch.float32),
            "y_lcc": torch.tensor(self.y["lcc"][index]),
            "y_lai": torch.tensor(self.y["lai"][index]),
            "y_ccc": torch.tensor(self.y["ccc"][index]),
            "mask_lcc": torch.tensor(self.mask["lcc"][index]),
            "mask_lai": torch.tensor(self.mask["lai"][index]),
            "qc_weight": torch.tensor(self.qc[index]),
        }
        if self.pheno is not None:
            item["pheno"] = torch.tensor(self.pheno[index], dtype=torch.float32)
        if self.patch is not None:
            patch = self.patch[index].copy()
            if self.augment:
                if np.random.rand() < 0.5:
                    patch = patch[:, :, ::-1]
                if np.random.rand() < 0.5:
                    patch = patch[:, ::-1, :]
                patch = np.rot90(patch, np.random.randint(0, 4), axes=(1, 2)).copy()
                patch = patch * (1.0 + np.random.uniform(-0.02, 0.02)) + np.random.normal(0.0, 0.005, size=patch.shape)
            mean = self.patch_mean[:, None, None]
            std = self.patch_std[:, None, None]
            item["patch"] = torch.tensor((patch - mean) / std, dtype=torch.float32)
        return item


def _predictor_columns(variant: dict, frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    mode = variant.get("pheno_mode", "all")
    if mode == "none":
        pheno = []
    elif mode == "vi":
        pheno = list(VI_COLUMNS)
    elif mode == "doy":
        pheno = ["doy_sin", "doy_cos"]
    elif mode == "all_sza":
        pheno = list(PHENOLOGY_COLUMNS) + ["cos_sza"]
    else:
        pheno = list(PHENOLOGY_COLUMNS)
    return list(SPECTRAL), [column for column in pheno if column in frame.columns]


def _independent_lai(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[np.isclose(pd.to_numeric(frame["lai_qc"], errors="coerce"), 1.0), "lai"].to_numpy(float)


def _qc_weight(values: np.ndarray, weights: dict, enabled: bool) -> np.ndarray:
    out = np.ones(len(values), dtype=np.float32)
    if not enabled:
        return out
    for i, value in enumerate(values):
        if np.isclose(value, 1.0):
            out[i] = float(weights["1"])
        elif np.isclose(value, 2.0):
            out[i] = float(weights["2"])
    return out


def _load_patches(frame: pd.DataFrame, index: pd.DataFrame) -> np.ndarray:
    paths = frame if "path" in frame.columns else frame.merge(index, on="sample_id", how="left")
    kept = paths["kept"].map(lambda value: str(value).lower() in {"1", "true"})
    if paths["path"].isna().any() or (~kept).any():
        raise RuntimeError("A training row is missing a kept image patch")
    return np.stack([np.load(path)["reflectance"] for path in paths["path"]]).astype(np.float32)


def _score(y_true, y_pred, mask, scale) -> float:
    if mask.sum() < 2:
        return 0.0
    stats = regression_metrics(y_true[mask], y_pred[mask])
    return float(stats["rmse"] / scale)


def _predict_numpy(model, dataset: FoldSet, device, scales: dict) -> dict[str, np.ndarray]:
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    collected = []
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["phys"].to(device),
                batch["pheno"].to(device) if "pheno" in batch else None,
                batch["patch"].to(device) if "patch" in batch else None,
            )
            collected.append({key: value.cpu().numpy() for key, value in pred.items()})
    merged = {key: np.concatenate([part[key] for part in collected]) for key in collected[0]}
    out = {}
    for target in ("lcc", "lai", "ccc"):
        out[f"mu_{target}"] = merged[f"mu_{target}"] * scales[target]["scale"] + scales[target]["center"]
        out[f"sigma_{target}"] = np.exp(0.5 * merged[f"logvar_{target}"]) * scales[target]["scale"]
    return out


def _partial_path(root, variant: dict, scheme: str, fold: int, seed: int, tasks: tuple[str, ...]):
    tag = "-".join(tasks)
    return root / "outputs" / "predictions" / "partial_rev2" / f"{variant['id']}__{scheme}__fold{fold}__seed{seed}__{tag}.csv"


def _train_one(variant: dict, scheme: str, fold: int, seed: int, tasks: tuple[str, ...]) -> list[dict]:
    cfg = configs()
    root = cfg["root"]
    partial = _partial_path(root, variant, scheme, fold, seed, tasks)
    if partial.exists():
        print(f"resume {partial.name}")
        return pd.read_csv(partial).to_dict(orient="records")
    features = read_table(root / "data" / "features" / "features.csv")
    splits = read_table(root / "splits" / f"{scheme}.csv")
    fold_df = splits[splits["outer_fold"] == fold].merge(features, on="sample_id", suffixes=("", "_feat"))
    use_cnn = bool(variant["use_cnn"])
    patch_index_path = root / "data" / "patches" / "patch_index.csv"
    if use_cnn and not patch_index_path.exists():
        print(f"skip {variant['id']} because patches are not available")
        return []
    patch_index = read_table(patch_index_path) if use_cnn else None
    phys_cols, pheno_cols = _predictor_columns(variant, fold_df)
    use_phenology = bool(pheno_cols)
    predictor_cols = phys_cols + pheno_cols
    fold_df = fold_df.replace([np.inf, -np.inf], np.nan).dropna(subset=predictor_cols).copy()
    train = fold_df[fold_df["role"] == "train"].copy()
    tune = fold_df[fold_df["role"] == "tune"].copy()
    if use_cnn:
        before = len(train)
        kept_ids = set(patch_index.loc[patch_index["kept"].map(lambda value: str(value).lower() in {"1", "true"}), "sample_id"])
        train = train[train["sample_id"].isin(kept_ids)].copy()
        tune = tune[tune["sample_id"].isin(kept_ids)].copy()
        if len(train) < 16:
            print(f"skip fold {fold}: only {len(train)} patched training samples (from {before})")
            return []

    checkpoint = root / "models" / "physics_encoder.pt"
    if variant["use_prosail_init"]:
        if not checkpoint.exists():
            raise FileNotFoundError("Physics encoder checkpoint is missing. Run 06_pretrain_physics.py.")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if list(saved["features"]) != phys_cols:
            raise RuntimeError("Physics encoder inputs do not match the five reflectance bands")
        phys_scaler = RobustScale.from_dict(saved["input_scaler"])
    else:
        saved = None
        phys_scaler = RobustScale().fit(train[phys_cols].to_numpy(float))
    pheno_scaler = RobustScale().fit(train[pheno_cols].to_numpy(float)) if use_phenology else None
    supervision = variant.get("lai_supervision", "weak")
    target_scalers = {}
    for target in ("lcc", "lai", "ccc"):
        if target == "lai":
            finite = _independent_lai(train)
            finite = finite[np.isfinite(finite)]
            if finite.size < 8:
                # Fixed anchors. Not fitted on a later flight or chosen from its score.
                # Default pair [1, 6] has median 3.5 and IQR 2.5 under this scaler.
                finite = np.asarray(variant.get("lai_anchors", [1.0, 6.0]), dtype=float)
        elif target == "ccc":
            qc = np.isclose(pd.to_numeric(train["lai_qc"], errors="coerce"), 1.0)
            finite = train.loc[qc, "ccc"].to_numpy(float)
            finite = finite[np.isfinite(finite)]
            if finite.size < 8:
                finite = np.array([0.1, 4.0])
        else:
            values = train[target].to_numpy(float)
            finite = values[np.isfinite(values)]
        target_scalers[target] = RobustScale().fit(finite if np.size(finite) else np.array([0.0, 1.0]))

    def pack(frame: pd.DataFrame, augment: bool):
        phys = phys_scaler.transform(frame[phys_cols].to_numpy(float))
        pheno = pheno_scaler.transform(frame[pheno_cols].to_numpy(float)) if pheno_scaler is not None else None
        y, mask = {}, {}
        qc = pd.to_numeric(frame["lai_qc"], errors="coerce").to_numpy(float)
        independent = np.isclose(qc, 1.0)
        for target, scaler in target_scalers.items():
            raw = frame[target].to_numpy(float)
            mask[target] = np.isfinite(raw).astype(np.float32)
            if target in {"lai", "ccc"} and supervision == "weak":
                mask[target] = mask[target] * independent.astype(np.float32)
            filled = np.where(np.isfinite(raw), raw, scaler.center[0])
            y[target] = scaler.transform(filled).astype(np.float32)
        qc = _qc_weight(frame["lai_qc"].to_numpy(float), cfg["data"]["qc_weights"], bool(variant["use_qc"]))
        patch = _load_patches(frame, patch_index) if use_cnn else None
        return phys, pheno, y, mask, qc, patch

    # Patch moments from the training split only.
    patch_mean = patch_std = None
    train_patch = _load_patches(train, patch_index) if use_cnn else None
    if train_patch is not None:
        patch_mean = train_patch.mean(axis=(0, 2, 3))
        patch_std = train_patch.std(axis=(0, 2, 3))
        patch_std = np.where(patch_std < 1e-6, 1.0, patch_std)

    def make_set(frame, augment):
        phys, pheno, y, mask, qc, patch = pack(frame, augment)
        return FoldSet(
            phys, pheno, y, mask, qc, patch, augment,
            phys_scaler.center, phys_scaler.scale, patch_mean, patch_std,
        )

    train_set = make_set(train, True)
    tune_set = make_set(tune, False) if len(tune) else None
    weak_rows = train[np.isclose(pd.to_numeric(train["lai_qc"], errors="coerce"), 2.0)].copy()
    weak_target = None
    weak_set = None
    if supervision == "weak" and "lai" in tasks and len(weak_rows):
        unique = np.unique(np.round(weak_rows["lai"].dropna().to_numpy(float), 6))
        if unique.size != 1:
            raise RuntimeError("The date-level LAI label is not a single recorded value")
        weak_target = float(unique[0])
        weak_set = make_set(weak_rows, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PPQMTL(
        pheno_dim=max(1, len(pheno_cols)),
        use_cnn=use_cnn,
        use_phenology=use_phenology,
        dropout=float(cfg["model"]["fusion"]["dropout"]),
    ).to(device)
    if saved is not None:
        model.physics.load_state_dict(saved["encoder"])
    phys_params = list(model.physics.parameters())
    phys_ids = {id(p) for p in phys_params}
    other = [p for p in model.parameters() if id(p) not in phys_ids]
    physics_lr = float(cfg["model"]["optimizer"]["physics_lr"] if saved is not None else cfg["model"]["optimizer"]["lr"])
    optimizer = torch.optim.AdamW(
        [{"params": other, "lr": float(cfg["model"]["optimizer"]["lr"])}, {"params": phys_params, "lr": physics_lr}],
        weight_decay=float(cfg["model"]["optimizer"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=float(cfg["model"]["scheduler"]["factor"]), patience=int(cfg["model"]["scheduler"]["patience"])
    )
    loader = DataLoader(train_set, batch_size=int(cfg["model"]["train"]["batch_size"]), shuffle=True)
    lambdas = {
        "lcc": float(cfg["model"]["loss"]["lambda_lcc"]),
        "lai": float(cfg["model"]["loss"]["lambda_lai"]),
        "ccc": float(cfg["model"]["loss"]["lambda_ccc"]),
        "consistency": float(cfg["model"]["loss"]["lambda_consistency"]),
    }
    scale_tensors = {
        "ccc_factor": float(cfg["data"]["ccc"]["factor"]),
        "lcc_center": float(target_scalers["lcc"].center[0]),
        "lcc_scale": float(target_scalers["lcc"].scale[0]),
        "lai_center": float(target_scalers["lai"].center[0]),
        "lai_scale": float(target_scalers["lai"].scale[0]),
        "ccc_center": float(target_scalers["ccc"].center[0]),
        "ccc_scale": float(target_scalers["ccc"].scale[0]),
    }
    best_state, best_score, stale = None, float("inf"), 0
    for epoch in range(int(cfg["model"]["train"]["max_epochs"])):
        model.train()
        for batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            for key, value in scale_tensors.items():
                batch[key] = torch.tensor(value, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            pred = model(batch["phys"], batch.get("pheno"), batch.get("patch"))
            loss, _ = multitask_loss(
                pred, batch, lambdas,
                logvar_min=float(cfg["model"]["loss"]["logvar_min"]),
                logvar_max=float(cfg["model"]["loss"]["logvar_max"]),
                use_consistency=bool(variant["use_consistency"]),
                use_qc=bool(variant["use_qc"]),
                hetero=bool(variant["hetero"]),
                tasks=tasks,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["model"]["optimizer"]["grad_clip"]))
            optimizer.step()
        lambda_date = float(variant.get("lambda_date", cfg["model"]["loss"]["lambda_weak"]))
        if weak_set is not None and lambda_date != 0.0:
            model.eval()
            weak_batch = next(iter(DataLoader(weak_set, batch_size=len(weak_set), shuffle=False)))
            weak_batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in weak_batch.items()}
            for key, value in scale_tensors.items():
                weak_batch[key] = torch.tensor(value, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            weak_pred = model(weak_batch["phys"], weak_batch.get("pheno"), weak_batch.get("patch"))
            weak_loss = lambda_date * date_level_lai_loss(
                weak_pred["mu_lai"], weak_batch["lai_center"], weak_batch["lai_scale"], weak_target,
            )
            weak_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["model"]["optimizer"]["grad_clip"]))
            optimizer.step()
        score = 0.0
        if tune_set is not None and len(tune_set):
            pred_tune = _predict_numpy(model, tune_set, device, {
                target: {"center": float(target_scalers[target].center[0]), "scale": float(target_scalers[target].scale[0])}
                for target in ("lcc", "lai", "ccc")
            })
            raw_tune = tune.reset_index(drop=True)
            tune_qc = pd.to_numeric(raw_tune["lai_qc"], errors="coerce").to_numpy(float)
            for target in tasks:
                if target == "ccc" and target not in raw_tune:
                    continue
                observed = raw_tune[target].to_numpy(float)
                mask = np.isfinite(observed)
                if target in {"lai", "ccc"}:
                    mask = mask & np.isclose(tune_qc, 1.0)
                score += _score(observed, pred_tune[f"mu_{target}"], mask, float(target_scalers[target].scale[0]))
        else:
            score = float(loss.detach().cpu())
        scheduler.step(score)
        if score < best_score - 1e-4:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["model"]["train"]["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    tag = "-".join(tasks)
    ckpt = root / "models" / "checkpoints" / f"{variant['id']}__{scheme}__fold{fold}__seed{seed}__{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "variant": variant,
        "scheme": scheme,
        "outer_fold": fold,
        "seed": seed,
        "tasks": tasks,
        "phys_cols": phys_cols,
        "pheno_cols": pheno_cols,
        "phys_scaler": phys_scaler.to_dict(),
        "pheno_scaler": None if pheno_scaler is None else pheno_scaler.to_dict(),
        "target_scalers": {key: value.to_dict() for key, value in target_scalers.items()},
        "patch_mean": None if patch_mean is None else patch_mean.tolist(),
        "patch_std": None if patch_std is None else patch_std.tolist(),
        "use_cnn": use_cnn,
    }, ckpt)
    rows = []
    scale_view = {
        target: {"center": float(target_scalers[target].center[0]), "scale": float(target_scalers[target].scale[0])}
        for target in ("lcc", "lai", "ccc")
    }
    for role in ("calibration", "test"):
        part = fold_df[fold_df["role"] == role].copy()
        if use_cnn:
            part = part[part["sample_id"].isin(kept_ids)].copy()
        if len(part) == 0:
            continue
        pred = _predict_numpy(model, make_set(part, False), device, scale_view)
        part = part.reset_index(drop=True)
        for target in tasks:
            for i, rec in part.iterrows():
                rows.append({
                    "model": variant["id"],
                    "scheme": scheme,
                    "outer_fold": int(fold),
                    "seed": int(seed),
                    "sample_id": rec["sample_id"],
                    "plot_id": int(rec["plot_id"]),
                    "date_key": rec["date_key"],
                    "lai_qc": rec["lai_qc"],
                    "role": role,
                    "target": target,
                    "y_true": rec[target],
                    "y_pred": float(pred[f"mu_{target}"][i]),
                    "y_sigma": float(pred[f"sigma_{target}"][i]) if variant["hetero"] else np.nan,
                })
    print(f"{variant['id']} {scheme} fold {fold} seed {seed} tasks {tag} tune_score={best_score:.4f}", flush=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(partial, index=False)
    return rows


def _seeds(variant: dict, ppq_cfg: dict) -> list[int]:
    key = variant["seeds"]
    if key == "main":
        return [int(v) for v in ppq_cfg["main_seeds"]]
    if key == "ablation":
        return [int(v) for v in ppq_cfg["ablation_seeds"]]
    return [int(v) for v in key]


def run_ppq(only: set[str] | None = None) -> None:
    cfg = configs()
    root = cfg["root"]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for variant in cfg["experiment"]["ppq"]["variants"]:
        if only and variant["id"] not in only:
            continue
        if variant["use_cnn"] and not (root / "data" / "patches" / "patch_index.csv").exists():
            print(f"skip {variant['id']}: no patches")
            continue
        if variant.get("separate_tasks"):
            task_groups = [("lcc",), ("lai",)]
        else:
            task_groups = [tuple(variant.get("tasks", ["lcc", "lai"]))]
        for scheme in variant["schemes"]:
            splits = read_table(root / "splits" / f"{scheme}.csv")
            allowed_folds = None if variant.get("folds") is None else {int(v) for v in variant["folds"]}
            for seed in _seeds(variant, cfg["experiment"]["ppq"]):
                for fold in sorted(splits["outer_fold"].unique()):
                    if allowed_folds is not None and int(fold) not in allowed_folds:
                        continue
                    for tasks in task_groups:
                        rows = _train_one(variant, scheme, int(fold), int(seed), tasks)
                        grouped.setdefault((variant["id"], scheme), []).extend(rows)
        for scheme in variant["schemes"]:
            rows = grouped.get((variant["id"], scheme), [])
            if not rows:
                continue
            dest = root / "outputs" / "predictions" / f"{variant['id']}__{scheme}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(dest, index=False)
            print(f"wrote {dest.name}")


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Train PPQ-MTL variants. The CNN variant waits for patches.")
    parser.add_argument("--only", default="", help="Comma-separated variant ids. Empty trains every variant.")
    args = parser.parse_args()
    only = {item.strip() for item in args.only.split(",") if item.strip()} or None
    meta = read_json(configs()["root"] / "data" / "features" / "feature_meta.json")
    print(f"feature source: {meta.get('spectral_source')}")
    run_ppq(only)


if __name__ == "__main__":
    main()
