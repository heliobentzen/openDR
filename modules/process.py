"""
End-to-end image processing pipeline for the openDR fundus grading system.

This module chains fundus extraction (circle masking + ellipse fitting) with
LED-glare removal and forwards the result to the Theia grading API.  An
optional Grad-CAM explanation step is available via
:func:`grade_with_explanation`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .extract import extract_fundus
from . import gradcam, remove_glare, theia


def grade(filename: str) -> float:
    """Process a raw fundus image and return its diabetic-retinopathy grade.

    The pipeline consists of three steps:

    1. **Extraction** – :func:`~modules.extract.extract_fundus` isolates the
       retinal disc by applying a circular crop followed by an ellipse-fitted
       mask.
    2. **Glare removal** – :func:`~modules.remove_glare.remove_glare`
       inpaints the two direct LED specular reflections from the 20D lens.
    3. **Grading** – :func:`~modules.theia.grade_request` uploads the
       processed image to the Theia API and returns a numeric DR grade.

    The processed image is written to disk as a JPEG (``<stem>_processed.jpg``
    in the same directory as *filename*) so it can be inspected manually
    and sent to the API by file path.

    Parameters
    ----------
    filename:
        Path to the raw fundus JPEG captured by the OWL camera.

    Returns
    -------
    float
        Diabetic-retinopathy grade returned by the Theia API, or ``-1``
        if the API call fails.
    """
    source_path = Path(filename)
    output = str(source_path.with_name(source_path.stem + "_processed.jpg"))
    cv2.imwrite(output, remove_glare.remove_glare(extract_fundus(filename)))
    return theia.grade_request(output)


def grade_with_explanation(
    filename: str,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Process a fundus image, grade it, and generate a Grad-CAM explanation.

    Runs the same extraction / glare-removal pipeline as :func:`grade`, then
    applies Grad-CAM to produce a colour-coded heatmap overlay and a JSON
    audit record that documents the critical lesion regions detected.

    Output files written alongside *filename*:

    * ``<stem>_processed.jpg`` – glare-corrected fundus image.
    * ``<stem>_gradcam.jpg`` – heatmap overlay highlighting lesion regions.
    * ``<stem>_gradcam.json`` – structured audit record (DR grade, confidence,
      bounding-box coordinates of detected lesions).

    Parameters
    ----------
    filename:
        Path to the raw fundus JPEG captured by the OWL camera.
    model_path:
        Optional path to a trained ``.pt`` / ``.pth`` checkpoint used by
        Grad-CAM.  Falls back to the ``OPEN_DR_MODEL_PATH`` environment
        variable; when neither is set the module operates in demo mode with
        random weights.

    Returns
    -------
    dict
        A mapping with keys:

        ``theia_grade``
            Numeric DR grade returned by the Theia API (``-1`` on failure).
        ``gradcam``
            The full Grad-CAM audit record (see
            :func:`~modules.gradcam.run_gradcam`).

    Raises
    ------
    RuntimeError
        If PyTorch / torchvision is not installed (raised by the Grad-CAM
        module).
    """
    source_path = Path(filename)
    processed_path = str(source_path.with_name(source_path.stem + "_processed.jpg"))

    processed_image = remove_glare.remove_glare(extract_fundus(filename))
    cv2.imwrite(processed_path, processed_image)

    theia_grade = theia.grade_request(processed_path)
    try:
        gradcam_record = gradcam.run_gradcam(
            processed_image, processed_path, model_path=model_path
        )
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Grad-CAM explanation step failed: {exc}") from exc

    return {
        "theia_grade": theia_grade,
        "gradcam": gradcam_record,
    }

