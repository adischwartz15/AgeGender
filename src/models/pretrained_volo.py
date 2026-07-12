"""Supplementary experiment: ImageNet-pretrained VOLO-D1 (face-only) multi-task model.

This is **not** part of the from-scratch controlled architecture ablation
suite (``src/models/multitask_model.py``, Experiments 0/0b/0c/A-D). It
answers a different question -- "how much practical improvement comes from
an externally pretrained modern visual backbone, relative to our best
from-scratch model" -- and is reported independently in Table B (see
``src/evaluation/comparison.py::build_transfer_learning_table``), never
merged into the core ablation table.

Reuses, unmodified, the project's existing task-specific adapters
(``src/models/adapters.py``), task heads (``src/models/heads.py``), and
learned homoscedastic-uncertainty loss balancing
(``src/losses/multitask_loss.py``) -- both ``AgeAdapter``/``GenderAdapter``
and ``AgeQuantileHead``/``GenderClassificationHead`` already accept an
arbitrary ``input_dim``, so nothing in those modules needed to change.

Face-only: this wraps a plain image-classification VOLO backbone from
``timm``, not MiVOLO (which additionally requires a body crop, a person
detector, and face+body cross-attention -- see README for the citation and
why this project only borrows MiVOLO's *motivation* for using VOLO on
faces, not its architecture).

``timm`` is an optional dependency (see ``requirements-transfer.txt``) --
this module is the *only* place in the repository that imports it, and
does so lazily inside functions, so every core (from-scratch) experiment
imports and runs with ``timm`` completely absent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.adapters import AgeAdapter, GenderAdapter, IdentityAdapter
from src.models.heads import AgeQuantileHead, GenderClassificationHead

DEFAULT_MODEL_ID = "volo_d1_224"

# Canonical, human-readable pretrained-source tags this project will ever
# accept for the transfer-learning extension. Deliberately does NOT include
# anything resembling "mivolo", "utkface", or any dataset this project
# evaluates on -- the point of this allow-list is to make it structurally
# impossible to configure a leakage source, not just to trust the caller.
ALLOWED_PRETRAINED_SOURCES = frozenset(
    {
        "imagenet1k",
        "imagenet21k",
        "imagenet21k_ft_imagenet1k",
        "imagenet22k",
        "imagenet22k_ft_imagenet1k",
    }
)

# Maps timm/PIL interpolation name strings (as returned by
# timm.data.resolve_model_data_config) to PIL resample constants used by
# src/data/transforms.py.
_INTERPOLATION_NAME_TO_PIL = {
    "bilinear": 2,  # PIL.Image.BILINEAR
    "bicubic": 3,  # PIL.Image.BICUBIC
    "nearest": 0,  # PIL.Image.NEAREST
    "lanczos": 1,  # PIL.Image.LANCZOS
    "box": 4,  # PIL.Image.BOX
    "hamming": 5,  # PIL.Image.HAMMING
}


class MissingTimmError(ImportError):
    """Raised when the VOLO transfer-learning extension is selected but ``timm`` is not installed."""


class PretrainedSourceNotAllowedError(ValueError):
    """Raised when ``model.volo.pretrained_source`` is outside the ImageNet-only allow-list.

    This is the config-level enforcement of "ImageNet-pretrained weights
    only" (never a MiVOLO/UTKFace checkpoint) -- it cannot be checked by
    inspecting the downloaded weights themselves, so the config value is
    validated against a closed allow-list before any weights are loaded.
    """


class InvalidStageTransitionError(RuntimeError):
    """Raised for a stage transition that doesn't correspond to a real training phase."""


def validate_pretrained_source(source: str) -> None:
    """Raise :class:`PretrainedSourceNotAllowedError` unless ``source`` is an ImageNet source."""
    if source not in ALLOWED_PRETRAINED_SOURCES:
        raise PretrainedSourceNotAllowedError(
            f"model.volo.pretrained_source='{source}' is not in the allow-list "
            f"{sorted(ALLOWED_PRETRAINED_SOURCES)}. This project only ever uses "
            "ImageNet-pretrained weights for the transfer-learning extension -- a "
            "MiVOLO or UTKFace-trained checkpoint would leak the test set."
        )


def _require_timm():
    try:
        import timm
    except ImportError as exc:
        raise MissingTimmError(
            "The VOLO transfer-learning extension requires timm. Install it with "
            "`pip install -r requirements-transfer.txt`."
        ) from exc
    # A broken/partial install (e.g. a stray namespace-package directory
    # left behind by an interrupted install/uninstall) can make `import
    # timm` succeed with a hollow module that has none of the real
    # library's attributes -- treat that identically to "not installed"
    # rather than letting it surface as a confusing AttributeError deeper
    # inside model construction.
    if not hasattr(timm, "create_model"):
        raise MissingTimmError(
            "A 'timm' module was importable but does not look like a real timm "
            "install (missing timm.create_model -- this can happen with a broken "
            "or partial install). Reinstall it with "
            "`pip install -r requirements-transfer.txt`."
        )
    return timm


def _verify_model_id(timm_module, model_id: str) -> None:
    available = timm_module.list_models(pretrained=False)
    if model_id not in available:
        raise ValueError(
            f"Unknown timm model identifier '{model_id}'. This is not present in the "
            f"installed timm version's model registry (checked via timm.list_models()). "
            f"Expected e.g. '{DEFAULT_MODEL_ID}' -- run timm.list_models('volo*') to see "
            "what your installed timm version actually provides."
        )


@dataclass
class VOLOParameterBreakdown:
    """Parameter counts by component, mirroring ``multitask_model.ParameterBreakdown``'s shape."""

    backbone_name: str
    backbone: int
    backbone_trainable: int
    adapters: int
    age_head: int
    gender_head: int
    log_variance: int
    total: int
    trainable_total: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "backbone_name": self.backbone_name,
            "backbone_parameters": self.backbone,
            "backbone_trainable_parameters": self.backbone_trainable,
            "adapter_parameters": self.adapters,
            "age_head_parameters": self.age_head,
            "gender_head_parameters": self.gender_head,
            "log_variance_parameters": self.log_variance,
            "total_parameters": self.total,
            "trainable_parameters": self.trainable_total,
        }


class PretrainedVOLOFaceOnlyMultiTask(nn.Module):
    """ImageNet-pretrained VOLO-D1 backbone + the project's existing adapters/heads/loss balancing.

    Not a subclass of ``MultiTaskFaceModel`` and not registered in
    ``src/models/backbone_factory.py`` -- VOLO's freeze/unfreeze semantics
    (partial "last N stages" unfreezing) and its own resolved
    size/mean/std/interpolation don't fit the generic
    top-level-submodule-name freezing that the from-scratch backbones share
    (see ``MultiTaskFaceModel.set_stage_trainable``), so this class owns its
    own small freeze/parameter-group API instead of stretching that one to
    cover a case it wasn't designed for. It exposes the same ``encode()``
    /``forward()`` output-dict contract as ``MultiTaskFaceModel`` so
    evaluation code (``scripts/evaluate.py``) can treat both interchangeably.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        model_cfg = config["model"]
        volo_cfg = model_cfg.get("volo", {})
        self.model_id: str = volo_cfg.get("model_id", DEFAULT_MODEL_ID)
        pretrained: bool = volo_cfg.get("pretrained", True)
        pretrained_source: str = volo_cfg.get("pretrained_source", "imagenet1k")
        # Pure config validation, checked before timm is even imported --
        # a disallowed pretrained_source is a config-authoring error that
        # should fail immediately, without requiring the optional
        # dependency to be installed just to report it.
        validate_pretrained_source(pretrained_source)
        self.pretrained_source = pretrained_source
        self.pretrained = pretrained

        timm = _require_timm()
        _verify_model_id(timm, self.model_id)

        # num_classes=0 makes timm's own forward() return the pooled
        # pre-logits feature vector directly -- see the dry-run assertions
        # below, which independently verify this via forward_features() +
        # forward_head(..., pre_logits=True) as well, so a silent pooling
        # mismatch cannot slip through undetected.
        #
        # No try/except around this call: if a pretrained-weight download
        # fails (offline, network error, revoked URL), this must raise --
        # never silently fall back to pretrained=False while still being
        # labeled a "pretrained" model.
        self.backbone = timm.create_model(self.model_id, pretrained=pretrained, num_classes=0)

        self.data_config = timm.data.resolve_model_data_config(self.backbone)
        if not self.data_config.get("mean") or not self.data_config.get("std"):
            raise ValueError(
                f"timm.data.resolve_model_data_config('{self.model_id}') did not return "
                "normalization mean/std -- cannot build a correct preprocessing pipeline "
                "for this backbone."
            )
        self.input_size: int = self.data_config["input_size"][-1]
        self.interpolation_name: str = self.data_config.get("interpolation", "bicubic")

        self.embedding_dim = self._discover_and_verify_embedding_dim(timm)

        adapters_cfg = model_cfg.get("adapters", {})
        bottleneck_ratio = adapters_cfg.get("bottleneck_ratio", 4)
        bottleneck_dim = adapters_cfg.get("bottleneck_dim") or max(1, round(self.embedding_dim / bottleneck_ratio))
        self.bottleneck_ratio = bottleneck_ratio
        self.bottleneck_dim = bottleneck_dim
        adapter_dropout = adapters_cfg.get("dropout", 0.1)
        adapters_enabled = adapters_cfg.get("enabled", True)

        if adapters_enabled:
            self.age_adapter: nn.Module = AgeAdapter(self.embedding_dim, bottleneck_dim, adapter_dropout)
            self.gender_adapter: nn.Module = GenderAdapter(self.embedding_dim, bottleneck_dim, adapter_dropout)
        else:
            self.age_adapter = IdentityAdapter()
            self.gender_adapter = IdentityAdapter()
        self.adapters_enabled = adapters_enabled

        age_head_cfg = model_cfg.get("age_head", {})
        gender_head_cfg = model_cfg.get("gender_head", {})
        self.age_head = AgeQuantileHead(
            input_dim=self.embedding_dim,
            hidden_dim=age_head_cfg.get("hidden_dim", 128),
            dropout=age_head_cfg.get("dropout", 0.1),
            age_min=age_head_cfg.get("age_min", 0),
            age_max=age_head_cfg.get("age_max", 120),
        )
        self.gender_head = GenderClassificationHead(
            input_dim=self.embedding_dim,
            hidden_dim=gender_head_cfg.get("hidden_dim", 128),
            dropout=gender_head_cfg.get("dropout", 0.1),
            num_classes=gender_head_cfg.get("num_classes", 2),
        )

        loss_balancing_cfg = model_cfg.get("loss_balancing", {})
        self.loss_balancing_mode = loss_balancing_cfg.get("mode", "learned_uncertainty")
        if self.loss_balancing_mode == "learned_uncertainty":
            init_cfg = loss_balancing_cfg.get("learned_uncertainty", {})
            self.log_var_age = nn.Parameter(torch.tensor(float(init_cfg.get("init_log_var_age", 0.0))))
            self.log_var_gender = nn.Parameter(torch.tensor(float(init_cfg.get("init_log_var_gender", 0.0))))
        else:
            self.log_var_age = None
            self.log_var_gender = None

        # Starts fully trainable (matches timm.create_model's default
        # requires_grad=True on every parameter); Stage 1 of
        # src/training/transfer_trainer.py calls freeze_backbone()
        # explicitly before training begins.

    def _discover_and_verify_embedding_dim(self, timm_module) -> int:
        """Dry-run forward pass: derive and cross-verify the pooled embedding dimension.

        Never hardcodes 512 (VOLO-D1's embedding dim is smaller than the
        from-scratch ResNet-18 backbone's -- 384, discovered here, not
        assumed). Cross-checks three independent signals' *shapes* and
        raises loudly on any mismatch, rather than silently trusting one
        of them:

        1. ``self.backbone.num_features`` (timm's own declared attribute).
        2. The actual ``forward(x)`` output shape (built with
           ``num_classes=0``; this is the pooled vector used everywhere
           else in this class, e.g. ``encode()``).
        3. The ``forward_features(x)`` -> ``forward_head(x, pre_logits=True)``
           path explicitly, to positively confirm ``forward_features``
           returns an *unpooled token tensor* (``[B, N, D]``, not already
           ``[B, D]``) and that ``forward_head`` is what performs the
           pooling -- exactly the failure mode this dry run exists to catch.

        Only the *shapes* of (2) and (3) are required to agree, not their
        values: for VOLO specifically, its auxiliary class-attention /
        token-labeling design means ``forward(x)`` and
        ``forward_head(forward_features(x), pre_logits=True)`` are
        verified (empirically, against a real timm install) to *not* be
        numerically identical, even though both are valid ``[B, D]``
        pooled representations. ``forward(x)`` (path 2) is treated as
        canonical since it is every timm model's guaranteed public
        contract; path 3 exists only to prove ``forward_features`` isn't
        already flat.
        """
        declared_dim = getattr(self.backbone, "num_features", None)
        if declared_dim is None:
            raise ValueError(
                f"timm model '{self.model_id}' has no 'num_features' attribute; cannot "
                "determine its embedding dimension programmatically."
            )

        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, 3, self.input_size, self.input_size)

            forward_out = self.backbone(dummy)
            if forward_out.ndim != 2:
                raise ValueError(
                    f"timm model '{self.model_id}' forward(x) with num_classes=0 returned "
                    f"a tensor of shape {tuple(forward_out.shape)}, expected exactly "
                    "[batch, embedding_dim] (a pooled vector). A token/spatial tensor here "
                    "would silently corrupt every downstream adapter/head."
                )

            features = self.backbone.forward_features(dummy)
            pooled = self.backbone.forward_head(features, pre_logits=True)
            if pooled.ndim != 2:
                raise ValueError(
                    f"timm model '{self.model_id}' forward_head(forward_features(x), "
                    f"pre_logits=True) returned shape {tuple(pooled.shape)}, expected "
                    "[batch, embedding_dim]."
                )

        if forward_out.shape[1] != declared_dim or pooled.shape[1] != declared_dim:
            raise ValueError(
                f"Embedding dimension mismatch for '{self.model_id}': "
                f"num_features={declared_dim}, forward(x).shape[1]={forward_out.shape[1]}, "
                f"forward_head(pre_logits=True).shape[1]={pooled.shape[1]}. These must all "
                "agree before adapters/heads can be safely wired up."
            )
        if features.ndim != 3:
            raise ValueError(
                f"'{self.model_id}': forward_features(x) returned shape {tuple(features.shape)} "
                "(expected a 3-D [batch, tokens, dim] token tensor) -- this dry run exists "
                "specifically to catch a model whose forward_features already returns a "
                "pooled [B, D] vector, which would need different handling."
            )

        self.backbone.train()
        return int(declared_dim)

    # -- encode/forward: same output-dict contract as MultiTaskFaceModel ------------

    def encode(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        z_shared = self.backbone(images)
        assert z_shared.ndim == 2 and z_shared.shape[1] == self.embedding_dim, (
            f"VOLO backbone output shape {tuple(z_shared.shape)} does not match the "
            f"discovered embedding_dim={self.embedding_dim}."
        )
        z_age = self.age_adapter(z_shared)
        z_gender = self.gender_adapter(z_shared)
        return {"shared_embedding": z_shared, "age_embedding": z_age, "gender_embedding": z_gender}

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.encode(images)
        age_output = self.age_head(embeddings["age_embedding"])
        gender_logits = self.gender_head(embeddings["gender_embedding"])
        return {**embeddings, "age_output": age_output, "gender_logits": gender_logits}

    # -- freeze/unfreeze + parameter groups ------------------------------------------

    def freeze_backbone(self) -> None:
        """Stage 1: freeze every backbone parameter. Adapters/heads/balancing are untouched."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze every backbone parameter (full fine-tune)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_stages(self, n: int) -> None:
        """Unfreeze only the last ``n`` stages of ``self.backbone.network`` (timm VOLO's stage list).

        Everything else in the backbone stays frozen. Always also unfreezes
        the final norm layer (``self.backbone.norm``), since it sits after
        the last stage and would otherwise stay frozen even when the stage
        feeding it is trainable.
        """
        if n < 1:
            raise InvalidStageTransitionError(f"unfreeze_last_stages(n={n}) requires n >= 1.")
        network = getattr(self.backbone, "network", None)
        if network is None:
            raise InvalidStageTransitionError(
                f"'{self.model_id}' has no 'network' attribute (the expected timm VOLO "
                "stage-list module). Cannot resolve 'last N stages' for this model -- "
                "inspect the installed timm version's VOLO module layout and update "
                "unfreeze_last_stages() accordingly."
            )
        stages = list(network)
        if n > len(stages):
            raise InvalidStageTransitionError(
                f"unfreeze_last_stages(n={n}) requested more stages than exist ({len(stages)})."
            )
        self.freeze_backbone()
        for stage in stages[-n:]:
            for param in stage.parameters():
                param.requires_grad = True
        norm = getattr(self.backbone, "norm", None)
        if norm is not None:
            for param in norm.parameters():
                param.requires_grad = True

    def get_parameter_groups(
        self, backbone_lr: float, adapter_lr: float, head_lr: float, balance_lr: float, weight_decay: float,
    ) -> list[dict]:
        """Separate optimizer param groups for backbone / adapters / heads / balancing params.

        Only includes a group if it has at least one trainable parameter
        (e.g. the backbone group is omitted entirely while fully frozen in
        Stage 1), mirroring the empty-group-skipping convention already
        used by ``src/training/trainer.py::_build_optimizer``.
        """
        groups = []

        def _add(params, lr):
            trainable = [p for p in params if p.requires_grad]
            if trainable:
                groups.append({"params": trainable, "lr": lr, "weight_decay": weight_decay})

        _add(self.backbone.parameters(), backbone_lr)
        _add(list(self.age_adapter.parameters()) + list(self.gender_adapter.parameters()), adapter_lr)
        _add(list(self.age_head.parameters()) + list(self.gender_head.parameters()), head_lr)
        if self.log_var_age is not None:
            _add([self.log_var_age, self.log_var_gender], balance_lr)
        if not groups:
            raise InvalidStageTransitionError("get_parameter_groups() produced zero trainable parameter groups.")
        return groups

    # -- introspection ------------------------------------------------------------

    def parameter_breakdown(self) -> VOLOParameterBreakdown:
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        adapter_params = 0
        if hasattr(self.age_adapter, "num_parameters"):
            adapter_params += self.age_adapter.num_parameters()
        if hasattr(self.gender_adapter, "num_parameters"):
            adapter_params += self.gender_adapter.num_parameters()
        age_head_params = sum(p.numel() for p in self.age_head.parameters())
        gender_head_params = sum(p.numel() for p in self.gender_head.parameters())
        log_var_params = 0
        if self.log_var_age is not None:
            log_var_params = self.log_var_age.numel() + self.log_var_gender.numel()

        total = backbone_params + adapter_params + age_head_params + gender_head_params + log_var_params
        trainable_total = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return VOLOParameterBreakdown(
            backbone_name=self.model_id,
            backbone=backbone_params,
            backbone_trainable=backbone_trainable,
            adapters=adapter_params,
            age_head=age_head_params,
            gender_head=gender_head_params,
            log_variance=log_var_params,
            total=total,
            trainable_total=trainable_total,
        )

    def build_transforms(self):
        """Build (TrainTransform, EvalTransform) using this backbone's own resolved
        input size/mean/std/interpolation -- never this project's 128px/existing
        normalization defaults (those stay exactly as-is for every core experiment)."""
        from src.data.transforms import EvalTransform, TrainTransform

        interpolation = _INTERPOLATION_NAME_TO_PIL.get(self.interpolation_name, 3)
        mean = tuple(self.data_config["mean"])
        std = tuple(self.data_config["std"])
        train_transform = TrainTransform(self.input_size, mean=mean, std=std, interpolation=interpolation)
        eval_transform = EvalTransform(self.input_size, mean=mean, std=std, interpolation=interpolation)
        return train_transform, eval_transform


def build_pretrained_volo_model(config: dict) -> PretrainedVOLOFaceOnlyMultiTask:
    """Factory mirroring ``multitask_model.build_multitask_model``'s call shape."""
    return PretrainedVOLOFaceOnlyMultiTask(config)
