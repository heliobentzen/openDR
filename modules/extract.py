"""
Image extraction utilities for the openDR fundus camera pipeline.

These functions isolate the retinal disc from a raw fundus photograph
captured by the OWL device and prepare it for downstream grading.

Memory notes (Raspberry Pi):
  * Masks are kept as single-channel ``uint8`` arrays – no redundant BGR
    allocation followed by a grayscale conversion.
  * :func:`cv2.bitwise_and` replaces three separate per-channel
    ``np.multiply`` calls, reducing peak memory usage by ~3× per masking
    operation and delegating the work to an optimised C routine.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Final

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Pixel coordinates of the optical-axis centre in a full-resolution OWL frame.
_CENTER: Final[tuple[int, int]] = (1386, 948)

# Radius (px) of the circular region of interest.
_RADIUS: Final[int] = 804

# Brightness threshold used when binarising the grayscale image (empirical).
_THRESHOLD_VALUE: Final[int] = 65

# Square kernel side-length for morphological erosion.
_KERNEL_SIZE: Final[int] = 14

# Number of erosion passes applied to the thresholded image.
_EROSION_ITERATIONS: Final[int] = 5


def extract_circles(image: np.ndarray) -> np.ndarray:
    """Mask all pixels outside the known circular region of interest.

    A filled circular mask is drawn at the fixed centre/radius for the OWL
    optics.  Every pixel outside that circle is set to black ``(0, 0, 0)``
    while pixels inside keep their original BGR values.

    Performance (Raspberry Pi): the mask is a single-channel ``uint8`` array
    applied with :func:`cv2.bitwise_and`.  This avoids allocating a
    three-channel mask, a grayscale conversion, a threshold pass, and three
    separate per-channel multiplication arrays that the previous
    implementation required.

    Parameters
    ----------
    image:
        Source BGR image as a NumPy array with shape ``(H, W, 3)`` and
        dtype ``uint8``.

    Returns
    -------
    np.ndarray
        A new BGR image of the same shape and dtype as *image* with the
        region outside the circle zeroed out.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, _CENTER, _RADIUS, 255, thickness=-1)
    return cv2.bitwise_and(image, image, mask=mask)


def erode_thresh(image: np.ndarray) -> np.ndarray:
    """Convert *image* to a smoothed binary mask via threshold and erosion.

    The image is converted to grayscale, binarised using a fixed intensity
    threshold, eroded to remove small bright artefacts, then Gaussian-blurred
    to produce smoother contour boundaries for downstream ellipse fitting.

    Parameters
    ----------
    image:
        BGR source image (typically the output of :func:`extract_circles`)
        with shape ``(H, W, 3)`` and dtype ``uint8``.

    Returns
    -------
    np.ndarray
        Single-channel ``uint8`` blurred binary image suitable for passing
        to :func:`cv2.findContours`.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((_KERNEL_SIZE, _KERNEL_SIZE), dtype=np.uint8)

    _ret, threshed = cv2.threshold(gray, _THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    threshed = cv2.erode(threshed, kernel, iterations=_EROSION_ITERATIONS)
    threshed = cv2.GaussianBlur(threshed, (21, 21), 0)

    return threshed


def ellipse_fit(image: np.ndarray, cont_img: np.ndarray) -> np.ndarray:
    """Fit an ellipse to the largest contour and mask the source image.

    Finds all external contours in *cont_img*, selects the one with the
    greatest area, fits an ellipse to it, then applies the resulting filled
    ellipse as a single-channel mask to *image*.

    Performance (Raspberry Pi): same single-channel mask + bitwise_and
    strategy as :func:`extract_circles`.

    Parameters
    ----------
    image:
        BGR source image (typically the output of :func:`extract_circles`)
        with shape ``(H, W, 3)`` and dtype ``uint8``.
    cont_img:
        Single-channel binary image (output of :func:`erode_thresh`) used
        for contour detection.

    Returns
    -------
    np.ndarray
        BGR image of the same shape and dtype as *image* with all pixels
        outside the fitted ellipse zeroed out.

    Raises
    ------
    ValueError
        If *cont_img* contains no detectable contours.
    """
    contours, _hierarchy = cv2.findContours(
        cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No contours found in the thresholded image.")

    largest_contour = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(largest_contour)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1)

    return cv2.bitwise_and(image, image, mask=mask)


def extract_fundus(filename: str) -> np.ndarray:
    """Load an image file and return the extracted fundus region.

    Convenience wrapper that chains :func:`extract_circles`,
    :func:`erode_thresh`, and :func:`ellipse_fit` into a single call
    suitable for use in the processing pipeline.

    Parameters
    ----------
    filename:
        Absolute or relative path to the source JPEG/PNG image captured
        by the OWL fundus camera.

    Returns
    -------
    np.ndarray
        BGR image containing only the retinal disc region; background
        pixels are black.

    Raises
    ------
    FileNotFoundError
        If *filename* cannot be read by OpenCV.
    ValueError
        If no fundus contour can be detected in the image.
    """
    test_img = cv2.imread(filename)
    if test_img is None:
        raise FileNotFoundError(f"Cannot read image file: {filename!r}")

    circle = extract_circles(test_img)
    threshed_image = erode_thresh(circle)
    return ellipse_fit(circle, threshed_image)

