"""Combines per-task losses with masked labels and configurable balancing.

Two balancing modes are supported (``model.loss_balancing.mode``):

* ``fixed``:              ``total = age_weight * age_loss + gender_weight * gender_loss``
* ``learned_uncertainty``: homoscedastic-uncertainty weighting (Kendall et al., 2018)
  using trainable log-variances ``s_age``, ``s_gender`` owned by the model::

      total = exp(-s_age) * age_loss + s_age + exp(-s_gender) * gender_loss + s_gender

A task's loss term is entirely omitted from ``total`` (not just down-weighted)
when its labels are absent for every sample in the batch, since a loss of 0
combined with a learned weight would otherwise still contribute a
"regularization" term (e.g. the ``+ s_gender`` bias term) that has no
supervisory meaning when there is no label at all in the batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.losses.quantile_loss import multi_quantile_pinball_loss


@dataclass
class MultiTaskLossOutput:
    total_loss: torch.Tensor
    age_loss: torch.Tensor | None
    gender_loss: torch.Tensor | None
    effective_age_weight: float
    effective_gender_weight: float
    log_var_age: float | None
    log_var_gender: float | None


def compute_multitask_loss(
    age_output: dict[str, torch.Tensor],
    gender_logits: torch.Tensor,
    age_target: torch.Tensor,
    age_mask: torch.Tensor,
    gender_target: torch.Tensor,
    gender_mask: torch.Tensor,
    mode: str,
    fixed_age_weight: float = 1.0,
    fixed_gender_weight: float = 1.0,
    log_var_age: torch.Tensor | None = None,
    log_var_gender: torch.Tensor | None = None,
    gender_class_weights: torch.Tensor | None = None,
) -> MultiTaskLossOutput:
    """Compute the combined training loss for one batch."""
    has_age = bool(age_mask.any().item())
    has_gender = bool(gender_mask.any().item())

    age_loss = None
    if has_age:
        age_loss = multi_quantile_pinball_loss(
            age_output["q10_raw"], age_output["q50_raw"], age_output["q90_raw"], age_target, age_mask
        )

    gender_loss = None
    if has_gender:
        per_sample = F.cross_entropy(
            gender_logits, gender_target, weight=gender_class_weights, reduction="none"
        )
        weights = gender_mask.float()
        gender_loss = (per_sample * weights).sum() / weights.sum()

    device = gender_logits.device
    total = torch.zeros((), device=device)
    eff_age_weight = 0.0
    eff_gender_weight = 0.0
    lv_age_val = None
    lv_gender_val = None

    if mode == "learned_uncertainty":
        if has_age:
            precision_age = torch.exp(-log_var_age)
            total = total + precision_age * age_loss + log_var_age
            eff_age_weight = precision_age.item()
            lv_age_val = log_var_age.item()
        if has_gender:
            precision_gender = torch.exp(-log_var_gender)
            total = total + precision_gender * gender_loss + log_var_gender
            eff_gender_weight = precision_gender.item()
            lv_gender_val = log_var_gender.item()
    else:
        if has_age:
            total = total + fixed_age_weight * age_loss
            eff_age_weight = fixed_age_weight
        if has_gender:
            total = total + fixed_gender_weight * gender_loss
            eff_gender_weight = fixed_gender_weight

    return MultiTaskLossOutput(
        total_loss=total,
        age_loss=age_loss,
        gender_loss=gender_loss,
        effective_age_weight=eff_age_weight,
        effective_gender_weight=eff_gender_weight,
        log_var_age=lv_age_val,
        log_var_gender=lv_gender_val,
    )
