"""Shared Grad-CAM heatmap overlay rendering.

Extracted from :mod:`modules.gradcam` so that any explainability module in
the openDR pipeline (DR Grad-CAM, glaucoma screening, future additions) can
blend a normalised activation map onto its source image without duplicating
the colormap + alpha-blend logic.
"""
from __future__ import annotations

import cv2
import numpy as np

#: Default opacity of the heatmap layer when blended onto the source image.
DEFAULT_OVERLAY_ALPHA: float = 0.45


def overlay_heatmap(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> np.ndarray:
    """Blend a normalised activation map over a BGR image as a JET heatmap.

    Parameters
    ----------
    image:
        Source BGR image (``uint8``).
    cam:
        Activation map with values in ``[0, 1]`` at any spatial resolution
        — it is resized to match *image*.
    alpha:
        Opacity of the heatmap layer (0 = transparent, 1 = fully opaque).

    Returns
    -------
    np.ndarray
        BGR ``uint8`` composite image, same shape as *image*.
    """
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    return cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)
