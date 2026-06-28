"""
Removal of LED glare from retinal images captured by the OWL device.

Author: Dhruv Joshi

Removes the two direct specular reflections produced by the 20D lens in
OWL 1.0 (part of the openDR project, Srujana Center for Innovation,
LV Prasad Eye Institute, Hyderabad, India).
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Final

# ---------------------------------------------------------------------------
# Glare-region constants (pixels in a full-resolution OWL frame)
# ---------------------------------------------------------------------------

# Centre of the glare patch.
_GLARE_X: Final[int] = 1396
_GLARE_Y: Final[int] = 958

# Half-width of the square crop used to detect and inpaint the glare.
_HALF_WIDTH: Final[int] = 100

# Fraction of the maximum pixel value above which a pixel is considered
# saturated (part of the specular highlight).
_SAT_THRESHOLD: Final[float] = 0.9


def remove_glare(im: np.ndarray) -> np.ndarray:
    """Remove direct specular LED reflections from a retinal image.

    Generates a binary mask by thresholding the green channel within a
    square region centred on the known glare location, dilates the mask
    to cover the full reflection area, then inpaints the affected region
    using the Telea algorithm.

    The operation is performed on a cropped *view* of *im* to avoid
    copying the entire full-resolution frame, which matters on
    memory-constrained Raspberry Pi hardware.  The array is modified in
    place.

    Parameters
    ----------
    im:
        Full-resolution BGR retinal image with dtype ``uint8``.  Modified
        in place.

    Returns
    -------
    np.ndarray
        The same array passed in (modified in place) with the specular
        glare region inpainted.
    """
    y0, y1 = _GLARE_Y - _HALF_WIDTH, _GLARE_Y + _HALF_WIDTH
    x0, x1 = _GLARE_X - _HALF_WIDTH, _GLARE_X + _HALF_WIDTH

    # Threshold the green channel of the crop to locate saturated pixels.
    green_crop: np.ndarray = im[y0:y1, x0:x1, 1]
    _ret, temp_mask = cv2.threshold(
        green_crop, _SAT_THRESHOLD * 256, 255, cv2.THRESH_BINARY
    )

    kernel = np.ones((25, 25), dtype=np.uint8)
    temp_mask = cv2.dilate(temp_mask, kernel)

    # Inpaint only the small crop rather than the whole frame.
    im[y0:y1, x0:x1, :] = cv2.inpaint(
        im[y0:y1, x0:x1, :], temp_mask, 1, cv2.INPAINT_TELEA
    )

    return im

