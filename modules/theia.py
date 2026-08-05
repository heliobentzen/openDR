"""
Client for the Theia diabetic-retinopathy grading API.

Uploads a processed fundus image to the MIT Media Lab Theia endpoint and
returns the numeric DR grade contained in the JSON response.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

# Network timeout (seconds) for the Theia upload request. Without this,
# requests.post blocks indefinitely if the API hangs or the connection
# stalls, which can exhaust the small inference worker pool that calls
# grade_request (see INFERENCE_WORKER_COUNT in fundus.py, default 2).
_REQUEST_TIMEOUT_S = 30


def grade_request(filename: str) -> float:
    """Upload a processed fundus image to Theia and return the DR grade.

    Reads an API key from ``<OPEN_DR_BASE>/key`` (the default base
    directory is ``/home/pi/openDR``, overridable via the
    ``OPEN_DR_BASE`` environment variable) and POSTs the image to the
    Theia REST endpoint.

    Parameters
    ----------
    filename:
        Path to the processed JPEG image to upload.

    Returns
    -------
    float
        The numeric diabetic-retinopathy grade from the API response, or
        ``-1`` if the key file is missing, the request fails or times out,
        or the response cannot be parsed.
    """
    base_folder = Path(os.environ.get("OPEN_DR_BASE", "/home/pi/openDR")).resolve()
    key_path = base_folder / "key"

    try:
        with key_path.open("r", encoding="utf-8") as keyfile:
            key = keyfile.readline().strip()
    except IOError:
        print("CANNOT FIND KEY FOR THEIA. PLEASE CHECK.")
        return -1

    uri = "https://theia.media.mit.edu/api/v1/uploadImage?key=" + key
    try:
        with open(filename, "rb") as image_file:
            response = requests.post(
                uri, files={"file": image_file}, timeout=_REQUEST_TIMEOUT_S
            )
    except requests.exceptions.RequestException as exc:
        print(f"THEIA REQUEST FAILED: {exc}")
        return -1

    if response.status_code != 200:
        return -1

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        print("THEIA RESPONSE WAS NOT VALID JSON.")
        return -1

    if not isinstance(data, dict):
        print(f"THEIA RESPONSE HAD UNEXPECTED SHAPE: {type(data).__name__}")
        return -1

    grade_value = data.get("grade")
    if isinstance(grade_value, list):
        grade_value = grade_value[0] if grade_value else None

    try:
        return float(grade_value)
    except (TypeError, ValueError):
        print(f"THEIA RESPONSE HAD UNEXPECTED GRADE VALUE: {grade_value!r}")
        return -1

