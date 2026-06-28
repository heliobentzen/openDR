"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for diabetic-retinopathy
classification on fundus images.

Implements Class Activation Mapping as described in:
  Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization", ICCV 2017.

The module:

* Runs a ResNet-18 classification model (fine-tuned or demo) on a processed
  fundus image.
* Extracts gradients from the final convolutional block to build the
  class-discriminative activation map (CAM).
* Blends the resulting heatmap over the original image to produce a
  colour-coded overlay (``<stem>_gradcam.jpg``).
* Identifies high-activation lesion regions (microaneurysms, exudates,
  haemorrhages) and writes their bounding-box coordinates together with the
  full audit record to ``<stem>_gradcam.json``.

Environment variables
---------------------
OPEN_DR_MODEL_PATH
    Path to a ``.pt`` / ``.pth`` checkpoint.  When absent or the file does
    not exist the module operates in *demo mode* with random weights – useful
    for integration testing without a trained checkpoint.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: ICDR (International Clinical Diabetic Retinopathy) scale labels.
DR_CLASSES: list[str] = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "PDR",
]

#: Fraction of the maximum activation above which a pixel is treated as a
#: potential lesion site.
_LESION_THRESHOLD: float = 0.5

#: Minimum bounding-box area (original-image pixels) required to report a
#: lesion region (guards against noise artefacts).
_MIN_LESION_AREA: int = 100

#: Opacity of the heatmap overlay blended onto the original image.
_OVERLAY_ALPHA: float = 0.45


# ---------------------------------------------------------------------------
# Internal hook helper
# ---------------------------------------------------------------------------


class _GradCAMHook:
    """Capture forward activations and backward gradients from a layer."""

    def __init__(self) -> None:
        self.activations: "torch.Tensor | None" = None
        self.gradients: "torch.Tensor | None" = None

    def forward_hook(
        self,
        _module: "nn.Module",
        _input: tuple,
        output: "torch.Tensor",
    ) -> None:
        self.activations = output.detach()

    def backward_hook(
        self,
        _module: "nn.Module",
        _grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        self.gradients = grad_output[0].detach()


# ---------------------------------------------------------------------------
# Model construction and loading
# ---------------------------------------------------------------------------


def _build_model(num_classes: int = 5) -> "nn.Module":
    """Return a ResNet-18 with a custom head for *num_classes* DR grades."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _load_model(
    model_path: str | None = None,
    num_classes: int = 5,
    device: "torch.device | None" = None,
) -> "nn.Module":
    """Load (or initialise) the classification model.

    If *model_path* points to an existing file the saved weights are
    loaded; otherwise the model is returned with random weights
    (demo / integration-test mode).

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` / ``.pth`` checkpoint.  Bare ``state_dict``
        files and checkpoints wrapped in a ``{"model_state_dict": …}``
        mapping are both supported.
    num_classes:
        Number of output classes (default: 5 for the ICDR scale).
    device:
        Target device.  Defaults to CPU.
    """
    if device is None:
        device = torch.device("cpu")

    model = _build_model(num_classes=num_classes)

    if model_path and Path(model_path).is_file():
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------


def _preprocess(image: np.ndarray) -> "torch.Tensor":
    """Convert a BGR OpenCV image to a normalised 4-D tensor.

    Applies the standard ImageNet normalisation so the model weights are
    compatible with torchvision pre-trained initialisations.

    Parameters
    ----------
    image:
        BGR image as a ``uint8`` NumPy array.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(1, 3, 224, 224)``.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform(rgb).unsqueeze(0)  # (1, C, H, W)


# ---------------------------------------------------------------------------
# Core Grad-CAM computation
# ---------------------------------------------------------------------------


def _compute_gradcam(
    model: "nn.Module",
    tensor: "torch.Tensor",
    target_class: int | None = None,
) -> tuple[np.ndarray, int, float]:
    """Compute the Grad-CAM activation map for *tensor*.

    Registers forward/backward hooks on the last residual block of the
    ResNet-18 backbone (``model.layer4[-1]``), performs a forward pass,
    back-propagates the score of *target_class*, then weights the captured
    feature maps by the global-average-pooled gradients.

    Parameters
    ----------
    model:
        Classification model (ResNet-18 or any model that exposes a
        ``layer4`` attribute containing residual blocks).
    tensor:
        Pre-processed input tensor of shape ``(1, 3, H, W)``.
    target_class:
        Class index for which to explain the prediction.  When ``None``
        the model's top-1 prediction is used.

    Returns
    -------
    tuple
        ``(cam, pred_class, confidence)`` where *cam* is a ``float32``
        array with values in ``[0, 1]`` at the spatial resolution of the
        last convolutional feature map, *pred_class* is the predicted
        class index, and *confidence* is the corresponding softmax score.
    """
    hook = _GradCAMHook()
    target_layer: "nn.Module" = model.layer4[-1]  # type: ignore[index]
    fwd_handle = target_layer.register_forward_hook(hook.forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(hook.backward_hook)

    try:
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, target_class].item())

        model.zero_grad()
        logits[0, target_class].backward()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    # Global-average-pool the gradients to obtain neuron importance weights.
    weights = hook.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = (weights * hook.activations).sum(dim=1).squeeze(0)  # (H, W)
    cam = torch.relu(cam).cpu().numpy()

    # Normalise to [0, 1].
    cam_min, cam_max = float(cam.min()), float(cam.max())
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam.astype(np.float32), target_class, confidence


# ---------------------------------------------------------------------------
# Heatmap overlay
# ---------------------------------------------------------------------------


def _overlay_heatmap(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = _OVERLAY_ALPHA,
) -> np.ndarray:
    """Blend the Grad-CAM heatmap over the original BGR image.

    Parameters
    ----------
    image:
        Original BGR fundus image (``uint8``).
    cam:
        Grad-CAM activation map with values in ``[0, 1]`` at any spatial
        resolution – it is resized to match *image*.
    alpha:
        Opacity of the heatmap layer (0 = transparent, 1 = fully opaque).

    Returns
    -------
    np.ndarray
        BGR ``uint8`` composite image.
    """
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    return cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)


# ---------------------------------------------------------------------------
# Lesion region extraction
# ---------------------------------------------------------------------------


def _extract_lesion_regions(
    cam: np.ndarray,
    image_shape: tuple[int, int],
    threshold: float = _LESION_THRESHOLD,
    min_area: int = _MIN_LESION_AREA,
) -> list[dict[str, Any]]:
    """Detect high-activation lesion regions and return their coordinates.

    Thresholds the Grad-CAM map to produce a binary mask, finds external
    contours, computes bounding boxes, and scales the coordinates back to
    the original image resolution.

    Parameters
    ----------
    cam:
        Normalised Grad-CAM map ``(H_cam, W_cam)`` with values in
        ``[0, 1]``.
    image_shape:
        ``(height, width)`` of the original image used to scale bounding
        boxes from the CAM resolution.
    threshold:
        Activation fraction above which a pixel is classified as a lesion
        candidate.
    min_area:
        Minimum bounding-box area (original pixels) to include a region.

    Returns
    -------
    list of dict
        Each entry contains the keys ``x``, ``y``, ``width``, ``height``,
        ``mean_activation``, and ``area`` in original-image pixel
        coordinates.  The list is sorted by descending ``mean_activation``
        (most significant lesions first).
    """
    orig_h, orig_w = image_shape
    cam_h, cam_w = cam.shape

    scale_x = orig_w / cam_w
    scale_y = orig_h / cam_h

    binary = (cam >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: list[dict[str, Any]] = []
    for cnt in contours:
        cx, cy, bw, bh = cv2.boundingRect(cnt)
        ox = int(round(cx * scale_x))
        oy = int(round(cy * scale_y))
        obw = int(round(bw * scale_x))
        obh = int(round(bh * scale_y))
        area = obw * obh
        if area < min_area:
            continue
        mean_act = float(cam[cy : cy + bh, cx : cx + bw].mean())
        regions.append(
            {
                "x": ox,
                "y": oy,
                "width": obw,
                "height": obh,
                "mean_activation": round(mean_act, 4),
                "area": area,
            }
        )

    regions.sort(key=lambda r: r["mean_activation"], reverse=True)
    return regions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_gradcam(
    image: np.ndarray,
    source_path: str,
    model_path: str | None = None,
    target_class: int | None = None,
    overlay_alpha: float = _OVERLAY_ALPHA,
) -> dict[str, Any]:
    """Run Grad-CAM on a processed fundus image and write the results to disk.

    Generates a colour-coded heatmap overlay saved alongside the source
    image as ``<stem>_gradcam.jpg`` and writes a JSON audit record as
    ``<stem>_gradcam.json``.  The JSON document contains the predicted DR
    grade, confidence score, and the bounding-box coordinates of all
    detected lesion regions.

    If *model_path* is not provided the value of the ``OPEN_DR_MODEL_PATH``
    environment variable is used.  When neither is set the model runs in
    demo mode with random weights.

    Parameters
    ----------
    image:
        Processed BGR fundus image (output of the extraction pipeline).
    source_path:
        Filesystem path of the processed image – used to derive output
        file names and recorded in the audit document.
    model_path:
        Optional path to a ``.pt`` / ``.pth`` checkpoint.  Falls back to
        the ``OPEN_DR_MODEL_PATH`` environment variable, then demo mode.
    target_class:
        Force Grad-CAM to explain a specific DR class index.  Defaults to
        the model's top-1 prediction.
    overlay_alpha:
        Heatmap opacity (0–1) used when blending the overlay.

    Returns
    -------
    dict
        The audit record that was written to ``<stem>_gradcam.json``.

    Raises
    ------
    RuntimeError
        If PyTorch or torchvision is not installed.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch and torchvision are required for Grad-CAM. "
            "Install them with: pip install torch torchvision"
        )

    resolved_model_path = model_path or os.environ.get("OPEN_DR_MODEL_PATH")
    device = torch.device("cpu")
    model = _load_model(resolved_model_path, device=device)
    tensor = _preprocess(image).to(device)

    cam, pred_class, confidence = _compute_gradcam(model, tensor, target_class)

    overlay = _overlay_heatmap(image, cam, alpha=overlay_alpha)
    lesions = _extract_lesion_regions(cam, image.shape[:2])

    p = Path(source_path).resolve()
    overlay_path = str(p.with_name(p.stem + "_gradcam.jpg"))
    json_path = str(p.with_name(p.stem + "_gradcam.json"))

    cv2.imwrite(overlay_path, overlay)

    audit_record: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_image": source_path,
        "gradcam_overlay": overlay_path,
        "gradcam_audit_json": json_path,
        "model_path": resolved_model_path or "random_weights_demo",
        "predicted_dr_grade": {
            "class_index": pred_class,
            "label": DR_CLASSES[pred_class],
            "confidence": round(confidence, 4),
        },
        "lesion_regions": lesions,
    }

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(audit_record, fh, indent=2, ensure_ascii=False)

    return audit_record
