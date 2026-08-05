"""Referable-glaucoma screening for the openDR fundus pipeline.

Ports the ConvNeXt-Tiny classifier and Grad-CAM explanation from the
``tiagopessoalim/glaucoma`` Hugging Face Space into a module that follows
the same conventions as :mod:`modules.gradcam`:

* Runs entirely on CPU (Raspberry Pi 4 target).
* Falls back to demo mode (random weights) when the checkpoint file is
  absent, instead of crashing — the audit record clearly marks this case.
* Caches the loaded model across calls (``_model_cache``).
* Writes a heatmap overlay + JSON audit record alongside the source image,
  mirroring :func:`modules.gradcam.run_gradcam`'s on-disk contract so
  ``fundus.py`` can drive both pipelines the same way.

The model predicts 11 sigmoid outputs: index 0 is the probability of
referable glaucoma (``RG_THRESHOLD``), and indices 1-10 are 10 morphological
signs (each with its own threshold in ``FEATURE_THRESHOLDS``) used by
ophthalmologists to justify a glaucoma referral (neuroretinal rim
appearance, RNFL defects, disc haemorrhages, etc).

Environment variables
----------------------
OPEN_DR_GLAUCOMA_MODEL_PATH
    Path to the ``.pt`` checkpoint. Defaults to
    ``<OPEN_DR_BASE>/models/convnext_tiny.pt``.
OPEN_DR_GLAUCOMA_CONFIG_PATH
    Path to the ``.json`` config (``model_name`` / ``IMG_SIZE``). Defaults
    to ``<OPEN_DR_BASE>/models/convnext_tiny.json``. When absent, the model
    is assumed to be ``convnext_tiny`` at 896x896 — the resolution the
    shipped checkpoint was actually trained at.

Fetch the checkpoint with ``tools/download_glaucoma_model.py`` — it is not
committed to the repository (~111 MB).
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .heatmap import DEFAULT_OVERLAY_ALPHA, overlay_heatmap

try:
    import timm
    import torch
    import torch.nn as nn

    # Matches modules/gradcam.py: use all available CPU cores for BLAS ops.
    # Harmless if gradcam.py already made this call earlier in the process.
    torch.set_num_threads(os.cpu_count() or 1)

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model / clinical constants (from the reference checkpoint's training run)
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_NAME: str = "convnext_tiny"

# The reference checkpoint's own config.json trains at 896x896, not the
# 224x224 that a casual read of the source Space's fallback default would
# suggest — that fallback was inconsistent with the checkpoint it ships.
_DEFAULT_IMG_SIZE: int = 896

_NUM_CLASSES: int = 11

# Sanity bounds for IMG_SIZE read from the config JSON — guards against a
# corrupted/hand-edited config causing a degenerate cv2.resize (0 or
# negative) or an OOM-inducing allocation (absurdly large) on the Pi.
_MIN_IMG_SIZE: int = 32
_MAX_IMG_SIZE: int = 2048

_IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: Probability threshold above which the referable-glaucoma head is positive.
RG_THRESHOLD: float = 0.508

#: Per-feature activation thresholds, in the same order as FEATURE_LABELS_PT.
FEATURE_THRESHOLDS: tuple[float, ...] = (
    0.508, 0.37, 0.614, 0.676, 0.697,
    0.75, 0.754, 0.171, 0.531, 0.814,
)

#: Morphological sign labels (PT-BR), in model-output order (indices 1-10).
FEATURE_LABELS_PT: dict[str, str] = {
    "appearance neuroretinal rim superiorly": "Anel neuro-retiniano superior",
    "appearance neuroretinal rim inferiorly": "Anel neuro-retiniano inferior",
    "retinal nerve fiber layer defect superiorly": "Defeito CFN superior",
    "retinal nerve fiber layer defect inferiorly": "Defeito CFN inferior",
    "baring of the circumlinear vessel superiorly": "Exposição vaso circunlinear sup.",
    "baring of the circumlinear vessel inferiorly": "Exposição vaso circunlinear inf.",
    "nasalization of the vessel trunk": "Nasalização do tronco vascular",
    "disc hemorrhages": "Hemorragias de disco",
    "laminar dots": "Pontos laminares",
    "large cup": "Escavação aumentada",
}

#: In-process model cache: (resolved_weights_path, model_name) -> (model, img_size).
_model_cache: "dict[tuple[str | None, str], tuple[nn.Module, int]]" = {}


def _default_base_folder() -> Path:
    return Path(os.environ.get("OPEN_DR_BASE", "/home/pi/openDR")).resolve()


def _default_model_path() -> Path:
    return _default_base_folder() / "models" / "convnext_tiny.pt"


def _default_config_path() -> Path:
    return _default_base_folder() / "models" / "convnext_tiny.json"


# ---------------------------------------------------------------------------
# Model construction and loading
# ---------------------------------------------------------------------------


def _resolve_config(config_path: str | None) -> tuple[str, int]:
    """Return ``(model_name, img_size)`` from *config_path*, or the defaults."""
    if config_path and Path(config_path).is_file():
        try:
            with open(config_path, encoding="utf-8") as fh:
                raw_cfg = json.load(fh)
            model_name = raw_cfg.get("model_name", _DEFAULT_MODEL_NAME)
            img_size = int(raw_cfg.get("IMG_SIZE", _DEFAULT_IMG_SIZE))
            if not (_MIN_IMG_SIZE <= img_size <= _MAX_IMG_SIZE):
                raise ValueError(f"IMG_SIZE {img_size} outside allowed range")
            return model_name, img_size
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(
                "Could not parse glaucoma model config %s (%s); using defaults.",
                config_path,
                exc,
            )
    return _DEFAULT_MODEL_NAME, _DEFAULT_IMG_SIZE


def _build_model(model_name: str = _DEFAULT_MODEL_NAME) -> "nn.Module":
    """Return a ConvNeXt with an 11-way sigmoid head (RG + 10 features)."""
    model = timm.create_model(model_name, pretrained=False, num_classes=_NUM_CLASSES)
    model.head.fc = nn.Sequential(model.head.fc, nn.Sigmoid())
    return model


def _load_model(
    weights_path: str | None = None,
    config_path: str | None = None,
    device: "torch.device | None" = None,
) -> "tuple[nn.Module, int]":
    """Load (or initialise) the glaucoma model.

    Mirrors :func:`modules.gradcam._load_model`: results are cached, and a
    missing checkpoint file produces a randomly-initialised model instead of
    raising — the caller can still exercise the full pipeline in demo mode.

    Returns
    -------
    tuple
        ``(model, img_size)`` — *img_size* comes from *config_path* (or the
        built-in default) and is needed by :func:`_preprocess`.
    """
    if device is None:
        device = torch.device("cpu")

    model_name, img_size = _resolve_config(config_path)
    resolved_weights = str(Path(weights_path).resolve()) if weights_path else None
    cache_key = (resolved_weights, model_name)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    model = _build_model(model_name)

    if resolved_weights and Path(resolved_weights).is_file():
        try:
            state_dict = torch.load(resolved_weights, map_location=device, weights_only=True)
        except (pickle.UnpicklingError, TypeError):
            state_dict = torch.load(resolved_weights, map_location=device, weights_only=False)

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # Fallback for checkpoints saved before the head was wrapped in
            # nn.Sequential(fc, Sigmoid()) — remap the bare Linear weights
            # onto fc[0] specifically (the Sequential itself has no
            # .weight/.bias of its own).
            model.load_state_dict(state_dict, strict=False)
            if "head.fc.0.weight" in state_dict and "head.fc.0.bias" in state_dict:
                model.head.fc[0].weight = nn.Parameter(state_dict["head.fc.0.weight"])
                model.head.fc[0].bias = nn.Parameter(state_dict["head.fc.0.bias"])
    else:
        logger.warning(
            "Glaucoma model weights not found at %s; running in demo mode "
            "with random weights. Run tools/download_glaucoma_model.py to "
            "fetch the real checkpoint.",
            resolved_weights,
        )

    model.to(device)
    model.eval()
    _model_cache[cache_key] = (model, img_size)
    return model, img_size


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------


def _crop_black_borders(image_bgr: np.ndarray, threshold: int = 7) -> np.ndarray:
    """Crop the black letterbox borders typical of retinograph exports."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    if not mask.any():
        return image_bgr
    coords = np.ix_(mask.any(axis=1), mask.any(axis=0))
    cropped = np.stack([image_bgr[..., c][coords] for c in range(3)], axis=-1)
    if cropped.size == 0:
        return image_bgr
    return cropped


def _preprocess(image_bgr: np.ndarray, img_size: int) -> "tuple[torch.Tensor, np.ndarray]":
    """Crop, resize and ImageNet-normalise *image_bgr* for the model.

    Replaces the reference implementation's
    ``albumentations.Compose([Normalize(...), ToTensorV2()])`` with plain
    NumPy so the pipeline does not need the ``albumentations`` dependency.

    Returns
    -------
    tuple
        ``(tensor, resized_bgr)`` — *tensor* is the normalised
        ``(1, 3, img_size, img_size)`` model input; *resized_bgr* is the
        cropped/resized (but not normalised) frame, reused later as the
        base image for the Grad-CAM overlay.
    """
    cropped = _crop_black_borders(image_bgr)
    resized_bgr = cv2.resize(
        cropped, (img_size, img_size), interpolation=cv2.INTER_AREA
    )
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    tensor = torch.from_numpy(np.ascontiguousarray(chw, dtype=np.float32)).unsqueeze(0)
    return tensor, resized_bgr


# ---------------------------------------------------------------------------
# Grad-CAM (and classification output, from the same forward pass)
# ---------------------------------------------------------------------------


def _compute_gradcam(model: "nn.Module", tensor: "torch.Tensor") -> "tuple[np.ndarray, np.ndarray]":
    """Run one forward+backward pass, returning both the model output and CAM.

    A separate no-grad forward pass purely for the classification output
    would cost a second full ConvNeXt-Tiny pass (~25-30% more CPU time at
    896x896) for nothing — Grad-CAM already needs a graph-enabled forward
    pass to backprop through, so the classification output is read off that
    same pass instead. Mirrors the single-pass design of
    :func:`modules.gradcam._compute_gradcam`.

    Hooks the last ConvNeXt stage (``model.stages[-1]``) and backpropagates
    the sigmoid output at index 0 (referable-glaucoma probability). Some
    timm ConvNeXt configurations expose stage activations in channels-last
    layout ``(B, H, W, C)`` rather than the usual channels-first
    ``(B, C, H, W)``; comparing ``acts.shape[1]`` against ``acts.shape[-1]``
    disambiguates the two before global-average-pooling the gradients,
    matching the reference implementation this was ported from.

    Returns
    -------
    tuple
        ``(raw_output, cam)`` — *raw_output* is the detached 11-element
        sigmoid output as a NumPy array; *cam* is a ``float32`` array with
        values in ``[0, 1]``, at the spatial resolution of the hooked
        stage's feature map.
    """
    activations: "dict[str, torch.Tensor]" = {}
    gradients: "dict[str, torch.Tensor]" = {}

    def forward_hook(_module, _inp, output):
        activations["value"] = output

    def backward_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0]

    target_layer = model.stages[-1]
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        with torch.enable_grad():
            model.zero_grad()
            output = model(tensor)
            raw_output = output.detach().squeeze(0).cpu().numpy()
            output[0, 0].backward()

        acts = activations["value"].detach()
        grads = gradients["value"].detach()

        if acts.shape[1] <= acts.shape[-1]:
            weights = grads.mean(dim=(1, 2))[0]
            cam = (acts[0] * weights).sum(dim=-1)
        else:
            weights = grads.mean(dim=(2, 3))[0]
            cam = (acts[0] * weights[:, None, None]).sum(0)

        cam_np = cam.cpu().float().numpy()
        cam_np = np.maximum(cam_np, 0)
        cam_max = float(cam_np.max())
        cam_np = cam_np / (cam_max + 1e-8)
        return raw_output, cam_np.astype(np.float32)
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


# ---------------------------------------------------------------------------
# Result analysis
# ---------------------------------------------------------------------------


def _extract_feature_flags(raw_output: np.ndarray) -> dict[str, Any]:
    """Apply RG_THRESHOLD / FEATURE_THRESHOLDS to the raw model output."""
    rg_probability = float(raw_output[0])
    feature_probabilities = raw_output[1:]

    features = {
        label_pt: bool(feature_probabilities[i] > FEATURE_THRESHOLDS[i])
        for i, label_pt in enumerate(FEATURE_LABELS_PT.values())
    }

    return {
        "referable_glaucoma_probability": round(rg_probability, 4),
        "threshold": RG_THRESHOLD,
        "positive": rg_probability > RG_THRESHOLD,
        "features": features,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_glaucoma_screening(
    image: np.ndarray,
    source_path: str,
    model_path: str | None = None,
    config_path: str | None = None,
    overlay_alpha: float = DEFAULT_OVERLAY_ALPHA,
    status_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Run referable-glaucoma screening and Grad-CAM on a fundus image.

    Writes a Grad-CAM overlay (``<stem>_glaucoma_gradcam.jpg``) and a JSON
    audit record (``<stem>_glaucoma.json``) alongside *source_path*, mirroring
    :func:`modules.gradcam.run_gradcam`'s on-disk contract.

    Parameters
    ----------
    image:
        Source BGR fundus image (``uint8``) — the raw capture/upload, not
        pre-processed.
    source_path:
        Filesystem path used to derive output file names. Must already
        exist with a ``.jpg``/``.jpeg``/``.png`` extension.
    model_path:
        Optional path to the ``.pt`` checkpoint. Falls back to
        ``OPEN_DR_GLAUCOMA_MODEL_PATH``, then
        ``<OPEN_DR_BASE>/models/convnext_tiny.pt``.
    config_path:
        Optional path to the ``.json`` config. Falls back to
        ``OPEN_DR_GLAUCOMA_CONFIG_PATH``, then
        ``<OPEN_DR_BASE>/models/convnext_tiny.json``.
    overlay_alpha:
        Heatmap opacity (0-1) used when blending the Grad-CAM overlay.
    status_callback:
        Optional ``callback(step_name, **payload)`` invoked after each
        phase, mirroring :func:`modules.process.grade_with_explanation`'s
        progress-reporting contract: ``"preprocessing"`` (no payload),
        ``"inference"`` (no payload), ``"report"`` (``glaucoma=<audit
        record>``).

    Returns
    -------
    dict
        The audit record that was written to ``<stem>_glaucoma.json``.

    Raises
    ------
    RuntimeError
        If PyTorch or timm is not installed.
    ValueError
        If *source_path* does not point to an existing JPEG/PNG file.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch and timm are required for glaucoma screening. "
            "Install them with: pip install torch torchvision timm"
        )

    p = Path(source_path).resolve()
    if not p.is_file():
        raise ValueError("source_path does not point to an existing image file.")
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(
            f"Unexpected file extension {p.suffix!r} for source_path; "
            "expected .jpg, .jpeg, or .png."
        )

    parent = p.parent
    base_path = str(parent / (p.stem + "_glaucoma_base.jpg"))
    overlay_path = str(parent / (p.stem + "_glaucoma_gradcam.jpg"))
    json_path = str(parent / (p.stem + "_glaucoma.json"))

    resolved_model_path = (
        model_path
        or os.environ.get("OPEN_DR_GLAUCOMA_MODEL_PATH")
        or str(_default_model_path())
    )
    resolved_config_path = (
        config_path
        or os.environ.get("OPEN_DR_GLAUCOMA_CONFIG_PATH")
        or str(_default_config_path())
    )

    device = torch.device("cpu")
    model, img_size = _load_model(resolved_model_path, resolved_config_path, device=device)

    tensor, resized_bgr = _preprocess(image, img_size)
    tensor = tensor.to(device)
    if status_callback is not None:
        status_callback("preprocessing")

    raw_output, cam = _compute_gradcam(model, tensor)
    analysis = _extract_feature_flags(raw_output)
    if status_callback is not None:
        status_callback("inference")

    overlay = overlay_heatmap(resized_bgr, cam, alpha=overlay_alpha)
    # resized_bgr (pre-heatmap) is written alongside the overlay so a UI can
    # show a pixel-aligned before/after comparison — the raw source_path
    # capture has different crop/resolution after _preprocess and would not
    # line up with the overlay.
    cv2.imwrite(base_path, resized_bgr)
    cv2.imwrite(overlay_path, overlay)

    weights_exist = Path(resolved_model_path).is_file()
    audit_record: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_image": source_path,
        "glaucoma_base_image": base_path,
        "glaucoma_gradcam_overlay": overlay_path,
        "glaucoma_audit_json": json_path,
        "model_path": resolved_model_path if weights_exist else "random_weights_demo",
        "referable_glaucoma": {
            "probability": analysis["referable_glaucoma_probability"],
            "threshold": analysis["threshold"],
            "positive": analysis["positive"],
        },
        "features": analysis["features"],
    }

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(audit_record, fh, indent=2, ensure_ascii=False)

    if status_callback is not None:
        status_callback("report", glaucoma=audit_record)

    return audit_record
