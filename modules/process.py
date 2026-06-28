"""
End-to-end image processing pipeline for the openDR fundus grading system.

This module chains fundus extraction (circle masking + ellipse fitting) with
LED-glare removal and forwards the result to the Theia grading API.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from .extract import extract_fundus
from . import remove_glare, theia


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

