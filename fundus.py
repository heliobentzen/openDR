##############################################################################
##  OWL v3.0                                                          ########
## ------------------------------------------------------------       ########
##  Authors: Ayush Yadav, Devesh Jain, Ebin Philip, Dhruv Joshi      ########
##############################################################################

import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import pigpio
from flask import Flask, redirect, render_template, request, url_for

from Fundus_Cam import Fundus_Cam


BASE_FOLDER = Path(os.environ.get("OPEN_DR_BASE", "/home/pi/openDR")).resolve()
MODULES_DIR = BASE_FOLDER / "modules"
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))

import process  # noqa: E402
from process import grade  # noqa: E402

grade_val = "Grade"
last_img = "1"
obj_state = False
obj_fc = None
processed_text = ""

app = Flask(__name__)
tokens = ["Flip", "Vid", "Click", "Switch", grade_val, "Shut"]
PATIENT_ID_RE = re.compile(r"^[A-Z0-9_-]{1,64}$")


@app.route("/")
def my_form():
    normalON()
    return render_template("index.html")


@app.route("/", methods=["POST"])
def my_form_post():
    global processed_text
    global obj_state
    global last_img
    global obj_fc

    obj_state = True
    processed_text = sanitize_patient_id(request.form["text"].upper())
    if not processed_text:
        return render_template("index.html")
    make_a_dir(processed_text)
    obj_fc = Fundus_Cam()
    return redirect(url_for("captureSimpleFunc"))


@app.route("/captureSimple", methods=["GET", "POST"])
def captureSimpleFunc():
    global last_img
    global grade_val

    if request.method == "GET":
        return render_template("capture_simple.html", params=tokens, grades={})

    if "d" not in request.form:
        return render_template("capture_simple.html", params=tokens, grades={})

    d = request.form["d"]

    if d == "Click":
        obj_fc.capture()
        decode_image(obj_fc.image)
        return render_template("capture_simple.html", params=tokens, grades={})

    if d == "Flip":
        obj_fc.flip_cam()
        return render_template("capture_simple.html", params=tokens, grades={})

    if d == "Vid":
        obj_fc.continuous_capture()
        if not obj_fc.wait_for_capture(timeout=5):
            return render_template(
                "capture_simple.html",
                params=tokens,
                grades={"grade": "CAPTURE TIMEOUT - RETRY OR CHECK CAMERA"},
            )
        decode_image(obj_fc.images)
        return render_template("capture_simple.html", params=tokens, grades={})

    if d == grade_val:
        if last_img == "1":
            return render_template(
                "capture_simple.html",
                params=tokens,
                grades={"grade": "NO IMAGE SPECIFIED"},
            )

        grade_val = str(grade(last_img))[:4]
        print("the grade is " + grade_val)
        return render_template(
            "capture_simple.html",
            params=tokens,
            grades={"grade": grade_val},
        )

    if d == "Switch":
        if obj_state is True:
            obj_fc.stop_preview()
            obj_fc.stop()
            return redirect(url_for("my_form"))
        return render_template("capture_simple.html", params=tokens, grades={})

    if d == "Shut":
        shut_down()
        return render_template("capture_simple.html", params=tokens, grades={})

    return render_template("capture_simple.html", params=tokens, grades={})


def decode_image(images):
    global last_img

    name_file = BASE_FOLDER / "name"
    with name_file.open("r", encoding="utf-8") as file_r:
        picn = int(file_r.read().strip())

    picn += 1
    with name_file.open("w", encoding="utf-8") as file_w:
        file_w.write(str(picn))

    no = 1
    patient_id = validated_patient_id(processed_text)
    patient_dir = BASE_FOLDER / "images"

    if isinstance(images, list):
        for img in images:
            image_path = patient_dir / f"{patient_id}_{picn}_{no}.jpg"
            image = cv2.imdecode(img, 1)
            cv2.imwrite(str(image_path), image)
            last_img = str(image_path)
            no += 1
    else:
        image_path = patient_dir / f"{patient_id}_{picn}_{no}.jpg"
        image = cv2.imdecode(images, 1)
        cv2.imwrite(str(image_path), image)
        last_img = str(image_path)


def make_a_dir(pr_t):
    validated_patient_id(pr_t)
    directory = BASE_FOLDER / "images"
    directory.mkdir(parents=True, exist_ok=True)


def sanitize_patient_id(value):
    cleaned = re.sub(r"[^A-Z0-9_-]", "", value)
    if PATIENT_ID_RE.fullmatch(cleaned):
        return cleaned
    return ""


def validated_patient_id(value):
    if not PATIENT_ID_RE.fullmatch(value):
        raise ValueError("Invalid patient identifier.")
    return value


orangeyellow = 14
bluegreen = 15
switch = 4

pi = pigpio.pi()
pi.set_mode(orangeyellow, pigpio.OUTPUT)
pi.set_mode(bluegreen, pigpio.OUTPUT)
pi.set_mode(switch, pigpio.INPUT)
pi.set_pull_up_down(switch, pigpio.PUD_UP)


def normalON():
    pi.write(orangeyellow, 0)
    pi.write(bluegreen, 1)


def secondaryON():
    pi.write(orangeyellow, 1)
    pi.write(bluegreen, 0)


def shut_down():
    command = "/usr/bin/sudo /sbin/shutdown now"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]
    print(output.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", threaded=True)
