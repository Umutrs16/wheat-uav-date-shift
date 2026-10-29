from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.metrics import regression_metrics
from wheat_ppqmtl.model import TabularMLP
from wheat_ppqmtl.paths import configure_stdio, configs


def _xy(frame: pd.DataFrame, columns: list[str], target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    use = frame.dropna(subset=columns + [target]).copy()
    return use[columns].to_numpy(float), use[target].to_numpy(float), use


def _predict_gpr(model, scaler, x):
    mean, std = model.predict(scaler.transform(x), return_std=True)
    return mean, std


def _fit_classical(name: str, x_train, y_train, x_tune, y_tune, seed: int, exp: dict):
    scaler = StandardScaler().fit(x_train)
    xt = scaler.transform(x_train)
    xv = scaler.transform(x_tune) if len(x_tune) else None

    if name == "gpr":
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(x_train.shape[1]),
            length_scale_bounds=(1e-2, 1e2),
            nu=float(exp["baselines"]["gpr"]["nu"]),
        ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 10.0))
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=int(exp["baselines"]["gpr"]["n_restarts_optimizer"]),
            random_state=seed,
        )
        try:
            model.fit(xt, y_train)
        except Exception:
            fallback = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=0.5) + WhiteKernel(0.1)
            model = GaussianProcessRegressor(
                kernel=fallback, normalize_y=True, n_restarts_optimizer=0, random_state=seed
            )
            model.fit(xt, y_train)
        return model, scaler

    if name == "plsr":
        best, best_score = None, float("inf")
        for n_comp in exp["baselines"]["plsr_components"]:
            if n_comp >= x_train.shape[1] or n_comp >= len(x_train):
                continue
            candidate = PLSRegression(n_components=int(n_comp))
            candidate.fit(xt, y_train)
            if xv is None or len(y_tune) == 0:
                return candidate, scaler
            pred = candidate.predict(xv).reshape(-1)
            score = regression_metrics(y_tune, pred)["rmse"]
            if score < best_score:
                best, best_score = candidate, score
        if best is None:
            best = PLSRegression(n_components=2).fit(xt, y_train)
        return best, scaler

    if name == "svr":
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        y_scaled = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
        best, best_score = None, float("inf")
        for c_value in exp["baselines"]["svr_C"]:
            candidate = SVR(C=float(c_value), epsilon=0.1)
            candidate.fit(xt, y_scaled)
            if xv is None or len(y_tune) == 0:
                return candidate, (scaler, y_scaler)
            pred = y_scaler.inverse_transform(candidate.predict(xv).reshape(-1, 1)).ravel()
            score = regression_metrics(y_tune, pred)["rmse"]
            if score < best_score:
                best, best_score = candidate, score
        return best, (scaler, y_scaler)

    if name == "rf":
        spec = exp["baselines"]["rf"]
        model = RandomForestRegressor(
            n_estimators=int(spec["n_estimators"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            max_features="sqrt",
            random_state=seed,
            n_jobs=4,
        )
        model.fit(xt, y_train)
        return model, scaler

    if name == "xgb":
        from xgboost import XGBRegressor
        spec = exp["baselines"]["xgb"]
        model = XGBRegressor(
            n_estimators=int(spec["n_estimators"]),
            max_depth=int(spec["max_depth"]),
            learning_rate=float(spec["learning_rate"]),
            subsample=float(spec["subsample"]),
            colsample_bytree=float(spec["colsample_bytree"]),
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=4,
        )
        model.fit(xt, y_train)
        return model, scaler

    raise ValueError(name)


def _predict(name, model, scaler, x):
    if name == "svr":
        x_scaler, y_scaler = scaler
        pred = model.predict(x_scaler.transform(x)).reshape(-1, 1)
        return y_scaler.inverse_transform(pred).ravel(), None
    if name == "gpr":
        return _predict_gpr(model, scaler, x)
    if name == "plsr":
        return model.predict(scaler.transform(x)).reshape(-1), None
    return model.predict(scaler.transform(x)), None


def _fit_mlp(x_train, y_train, x_tune, y_tune, seed: int, model_cfg: dict):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    xt = torch.tensor(scaler.transform(x_train), dtype=torch.float32)
    yt = torch.tensor(y_scaler.transform(y_train.reshape(-1, 1)).ravel(), dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_cfg["optimizer"]["lr"]), weight_decay=float(model_cfg["optimizer"]["weight_decay"]))
    loader = DataLoader(TensorDataset(xt, yt), batch_size=int(model_cfg["train"]["batch_size"]), shuffle=True)
    best_state, best_score, stale = None, float("inf"), 0
    patience = int(model_cfg["train"]["patience"])
    for _ in range(int(model_cfg["train"]["max_epochs"])):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward()
            optimizer.step()
        if len(x_tune) == 0:
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            break
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(scaler.transform(x_tune), dtype=torch.float32, device=device)).cpu().numpy()
        pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
        score = regression_metrics(y_tune, pred)["rmse"]
        if score < best_score - 1e-4:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, (scaler, y_scaler, device)


def _predict_mlp(model, bundle, x):
    import torch
    scaler, y_scaler, device = bundle
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(scaler.transform(x), dtype=torch.float32, device=device)).cpu().numpy()
    return y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel(), None


def _plot_supported(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    if target != "lai":
        return frame
    qc = pd.to_numeric(frame["lai_qc"], errors="coerce")
    return frame[np.isclose(qc, 1.0)].copy()


def _constant_value(train: pd.DataFrame, target: str) -> float | None:
    supported = _plot_supported(train, target).dropna(subset=[target])
    if target == "lai" and len(supported) < 8:
        qc2 = train[np.isclose(pd.to_numeric(train["lai_qc"], errors="coerce"), 2.0)]
        values = qc2["lai"].dropna().to_numpy(float)
        unique = np.unique(np.round(values, 6)) if values.size else np.array([])
        if unique.size == 1:
            return float(unique[0])
        return None
    if len(supported) < 2:
        return None
    return float(supported[target].mean())


def _append_predictions(rows, model_name, scheme, fold, seed, role, part, target, mean, sigma):
    part = part.reset_index(drop=True)
    for i, rec in part.iterrows():
        rows.append({
            "model": model_name,
            "scheme": scheme,
            "outer_fold": int(fold),
            "seed": int(seed),
            "sample_id": rec["sample_id"],
            "plot_id": int(rec["plot_id"]),
            "date_key": rec["date_key"],
            "lai_qc": rec.get("lai_qc", np.nan),
            "role": role,
            "target": target,
            "y_true": rec[target],
            "y_pred": float(mean[i]) if np.ndim(mean) else float(mean),
            "y_sigma": float(sigma[i]) if sigma is not None else np.nan,
        })


def run_baselines() -> None:
    cfg = configs()
    root = cfg["root"]
    features = read_table(root / "data" / "features" / "features.csv")
    rows = []
    schemes = cfg["experiment"].get("baseline_schemes", cfg["experiment"]["schemes"])
    feature_sets = cfg["experiment"]["baselines"]["feature_sets"]
    for scheme in schemes:
        splits = read_table(root / "splits" / f"{scheme}.csv")
        for feature_name, columns in feature_sets.items():
            for model_name in cfg["experiment"]["baselines"]["models"]:
                model_id = f"{model_name}__{feature_name}"
                print(f"baseline {model_id} {scheme}", flush=True)
                for target in ("lcc", "lai"):
                    for seed in cfg["experiment"]["baselines"]["seeds"]:
                        for fold in sorted(splits["outer_fold"].unique()):
                            fold_df = splits[splits["outer_fold"] == fold]
                            merged = fold_df.merge(features, on="sample_id", suffixes=("", "_feat"))
                            train = _plot_supported(merged[merged["role"] == "train"], target)
                            tune = _plot_supported(merged[merged["role"] == "tune"], target)
                            calibration = _plot_supported(merged[merged["role"] == "calibration"], target)
                            test = _plot_supported(merged[merged["role"] == "test"], target)
                            x_train, y_train, _ = _xy(train, columns, target)
                            x_tune, y_tune, _ = _xy(tune, columns, target)
                            if len(x_train) < 8 or len(x_tune) < 2:
                                print(f"  skip {model_id} {scheme} {target} fold {int(fold)}: not enough independent labels", flush=True)
                                continue
                            if model_name == "mlp":
                                model, bundle = _fit_mlp(x_train, y_train, x_tune, y_tune, int(seed), cfg["model"])
                                predict = lambda x, model=model, bundle=bundle: _predict_mlp(model, bundle, x)
                            else:
                                model, bundle = _fit_classical(model_name, x_train, y_train, x_tune, y_tune, int(seed), cfg["experiment"])
                                predict = lambda x, model=model, bundle=bundle, model_name=model_name: _predict(model_name, model, bundle, x)
                            for role, part in (("calibration", calibration), ("test", test)):
                                if len(part) == 0:
                                    continue
                                usable = part.dropna(subset=columns + [target]).copy()
                                if len(usable) == 0:
                                    continue
                                mean, sigma = predict(usable[columns].to_numpy(float))
                                _append_predictions(rows, model_id, scheme, fold, seed, role, usable, target, mean, sigma)
                dest = root / "outputs" / "predictions" / f"{model_id}__{scheme}.csv"
                dest.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([row for row in rows if row["model"] == model_id and row["scheme"] == scheme]).to_csv(dest, index=False)
                print(f"wrote {dest.name}", flush=True)
        print(f"constant {scheme}", flush=True)
        for target in ("lcc", "lai"):
            for seed in cfg["experiment"]["baselines"]["seeds"]:
                for fold in sorted(splits["outer_fold"].unique()):
                    fold_df = splits[splits["outer_fold"] == fold]
                    merged = fold_df.merge(features, on="sample_id", suffixes=("", "_feat"))
                    value = _constant_value(merged[merged["role"] == "train"], target)
                    if value is None:
                        continue
                    for role in ("calibration", "test"):
                        part = _plot_supported(merged[merged["role"] == role], target).dropna(subset=[target])
                        if len(part) == 0:
                            continue
                        mean = np.full(len(part), value)
                        _append_predictions(rows, "constant", scheme, fold, seed, role, part, target, mean, None)
        dest = root / "outputs" / "predictions" / f"constant__{scheme}.csv"
        pd.DataFrame([row for row in rows if row["model"] == "constant" and row["scheme"] == scheme]).to_csv(dest, index=False)
        print(f"wrote {dest.name}", flush=True)


def main() -> None:
    configure_stdio()
    argparse.ArgumentParser(description="Train the pre-specified tabular baselines on frozen splits.").parse_args()
    run_baselines()


if __name__ == "__main__":
    main()
