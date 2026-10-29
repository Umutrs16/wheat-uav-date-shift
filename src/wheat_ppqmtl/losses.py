from __future__ import annotations

import torch
import torch.nn.functional as F


def hetero_nll(y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.exp(-logvar) * (y - mu) ** 2 + 0.5 * logvar


def date_level_lai_loss(mu_lai: torch.Tensor, center: torch.Tensor, scale: torch.Tensor, target: float) -> torch.Tensor:
    """One date-level residual, divided by the training LAI interquartile range.

    Plot-level likelihoods are in standardized units. The date mean is in m² m⁻²,
    so the residual is divided by the same IQR before it is squared.
    """
    predicted = mu_lai * scale + center
    iqr = torch.clamp(scale, min=1e-6)
    return ((predicted.mean() - float(target)) / iqr) ** 2


def masked_mean(values: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    w = mask if weight is None else mask * weight
    denom = torch.clamp(w.sum(), min=1e-8)
    if float(w.sum()) <= 0:
        return values.sum() * 0.0
    return (values * w).sum() / denom


def multitask_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    lambdas: dict[str, float],
    logvar_min: float = -6.0,
    logvar_max: float = 6.0,
    use_consistency: bool = True,
    use_qc: bool = True,
    hetero: bool = True,
    tasks: tuple[str, ...] = ("lcc", "lai", "ccc"),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Physiological multi-task loss. Targets are in standardized units.

    Consistency is computed from original-unit predictions and then mapped back
    to the CCC standardized space with `ccc_center` and `ccc_scale`.
    """
    parts = {}
    total = pred["mu_lcc"].sum() * 0.0
    ccc_factor = batch["ccc_factor"]

    def head(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        mu = pred[f"mu_{name}"]
        logvar = torch.clamp(pred[f"logvar_{name}"], logvar_min, logvar_max)
        return mu, logvar

    if "lcc" in tasks:
        mu, logvar = head("lcc")
        if hetero:
            values = hetero_nll(batch["y_lcc"], mu, logvar)
        else:
            values = (batch["y_lcc"] - mu) ** 2
        parts["lcc"] = masked_mean(values, batch["mask_lcc"])
        total = total + lambdas["lcc"] * parts["lcc"]

    if "lai" in tasks:
        mu, logvar = head("lai")
        if hetero:
            values = hetero_nll(batch["y_lai"], mu, logvar)
        else:
            values = (batch["y_lai"] - mu) ** 2
        weight = batch["qc_weight"] if use_qc else torch.ones_like(batch["qc_weight"])
        parts["lai"] = masked_mean(values, batch["mask_lai"], weight)
        total = total + lambdas["lai"] * parts["lai"]

    both = batch["mask_lcc"] * batch["mask_lai"]
    if "ccc" in tasks:
        mu, logvar = head("ccc")
        if hetero:
            values = hetero_nll(batch["y_ccc"], mu, logvar)
        else:
            values = (batch["y_ccc"] - mu) ** 2
        parts["ccc"] = masked_mean(values, both)
        total = total + lambdas["ccc"] * parts["ccc"]

    if use_consistency and "lcc" in tasks and "lai" in tasks and "ccc" in tasks:
        lcc_hat = pred["mu_lcc"] * batch["lcc_scale"] + batch["lcc_center"]
        lai_hat = pred["mu_lai"] * batch["lai_scale"] + batch["lai_center"]
        ccc_hat = ccc_factor * lcc_hat * lai_hat
        ccc_hat_z = (ccc_hat - batch["ccc_center"]) / batch["ccc_scale"]
        cons = F.smooth_l1_loss(pred["mu_ccc"], ccc_hat_z, reduction="none")
        parts["consistency"] = masked_mean(cons, both)
        total = total + lambdas["consistency"] * parts["consistency"]

    report = {key: float(value.detach().cpu()) for key, value in parts.items()}
    report["total"] = float(total.detach().cpu())
    return total, report
