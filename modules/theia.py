import json
import os
from pathlib import Path

import requests


def grade_request(filename):
    base_folder = Path(os.environ.get("OPEN_DR_BASE", "/home/pi/openDR")).resolve()
    key_path = base_folder / "key"

    try:
        with key_path.open("r", encoding="utf-8") as keyfile:
            key = keyfile.readline().strip()
    except IOError:
        print("CANNOT FIND KEY FOR THEIA. PLEASE CHECK.")
        return -1

    uri = "https://theia.media.mit.edu/api/v1/uploadImage?key=" + key
    with open(filename, "rb") as image_file:
        response = requests.post(uri, files={"file": image_file})

    if response.status_code == 200:
        data = json.loads(response.text)
        grade_value = data.get("grade")
        if isinstance(grade_value, list) and grade_value:
            return float(grade_value[0])
        return float(grade_value)
    return -1

