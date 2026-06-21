##############################################################################
##  OWL v3.0                                                          ########
## ------------------------------------------------------------       ########
##  Authors: Ayush Yadav, Devesh Jain, Ebin Philip, Dhruv Joshi      ########
##############################################################################

import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

import cv2
from flask import Flask, redirect, render_template, request, url_for

from Fundus_Cam import Fundus_Cam
from modules.process import grade

try:
    import pigpio
except ImportError:  # pragma: no cover - depends on Raspberry Pi runtime
    pigpio = None

app = Flask(__name__)
BASE_FOLDER = Path(os.environ.get("OPEN_DR_BASE", "/home/pi/openDR")).resolve()
TOKENS = ["Flip", "Vid", "Click", "Switch", "Grade", "Shut"]
PATIENT_ID_RE = re.compile(r"^[A-Z0-9_-]{1,64}$")

orangeyellow = 14
bluegreen = 15
switch = 4
pi = None


class CameraSessionState:
    def __init__(self):
        self.lock = Lock()
        self.camera = None
        self.last_img = None
        self.patient_id = ""

    def reset(self):
        self.patient_id = ""
        self.last_img = None

    def stop_camera(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera = None


state = CameraSessionState()


@app.route("/")
def my_form():
    normalON()
    return render_template("index.html")


@app.route("/", methods=["POST"])
def my_form_post():
    patient_id = sanitize_patient_id(request.form["text"].upper())
    if not patient_id:
        return render_template("index.html")

    make_a_dir(patient_id)
    with state.lock:
        state.stop_camera()
        state.patient_id = patient_id
        state.last_img = None
        state.camera = Fundus_Cam()
    return redirect(url_for("captureSimpleFunc"))


@app.route("/captureSimple", methods=["GET", "POST"])
def captureSimpleFunc():
    if request.method == "GET":
        return render_capture()

    if "d" not in request.form:
        return render_capture()

    d = request.form["d"]

    if d == "Click":
        with state.lock:
            if state.camera is None or not state.patient_id:
                return render_capture("NO ACTIVE CAPTURE SESSION")
            image = state.camera.capture()
            state.last_img = save_captured_images(state.patient_id, image)
        return render_capture()

    if d == "Flip":
        with state.lock:
            if state.camera is None:
                return render_capture("NO ACTIVE CAPTURE SESSION")
            state.camera.flip_cam()
        return render_capture()

    if d == "Vid":
        with state.lock:
            if state.camera is None or not state.patient_id:
                return render_capture("NO ACTIVE CAPTURE SESSION")
            state.camera.continuous_capture()
            if not state.camera.wait_for_capture(timeout=5):
                return render_capture("CAPTURE TIMEOUT - RETRY OR CHECK CAMERA")
            state.last_img = save_captured_images(state.patient_id, state.camera.images)
        return render_capture()

    if d == "Grade":
        with state.lock:
            last_img = state.last_img
        if last_img is None:
            return render_capture("NO IMAGE SPECIFIED")

        grade_result = str(grade(last_img))[:4]
        print("the grade is " + grade_result)
        return render_capture(grade_result)

    if d == "Switch":
        with state.lock:
            has_camera = state.camera is not None
            state.stop_camera()
            state.reset()
        if has_camera:
            return redirect(url_for("my_form"))
        return render_capture()

    if d == "Shut":
        shut_down()
        return render_capture()

    return render_capture()


def render_capture(grade_message=""):
    return render_template(
        "capture_simple.html",
        params=TOKENS,
        grades={"grade": grade_message},
    )


def save_captured_images(patient_id, images):
    no = 1
    patient_id = validated_patient_id(patient_id)
    patient_dir = BASE_FOLDER / "images"
    last_saved_path = None

    if isinstance(images, list):
        for img in images:
            image_path = build_image_path(patient_dir, patient_id, no)
            image = cv2.imdecode(img, 1)
            cv2.imwrite(str(image_path), image)
            last_saved_path = str(image_path)
            no += 1
    else:
        image_path = build_image_path(patient_dir, patient_id, no)
        image = cv2.imdecode(images, 1)
        cv2.imwrite(str(image_path), image)
        last_saved_path = str(image_path)

    return last_saved_path


def build_image_path(patient_dir, patient_id, capture_number):
    image_identifier = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')}_{uuid4().hex}"
    return patient_dir / f"{patient_id}_{image_identifier}_{capture_number}.jpg"


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


def init_gpio():
    global pi

    if pigpio is None:
        return None

    if pi is None:
        controller = pigpio.pi()
        if not controller.connected:
            app.logger.warning("pigpio daemon is unavailable; GPIO output is disabled.")
            controller.stop()
            return None
        controller.set_mode(orangeyellow, pigpio.OUTPUT)
        controller.set_mode(bluegreen, pigpio.OUTPUT)
        controller.set_mode(switch, pigpio.INPUT)
        controller.set_pull_up_down(switch, pigpio.PUD_UP)
        pi = controller

    return pi


def normalON():
    controller = init_gpio()
    if controller is None:
        return
    controller.write(orangeyellow, 0)
    controller.write(bluegreen, 1)


def secondaryON():
    controller = init_gpio()
    if controller is None:
        return
    controller.write(orangeyellow, 1)
    controller.write(bluegreen, 0)


def shut_down():
    command = "/usr/bin/sudo /sbin/shutdown now"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]
    print(output.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    init_gpio()
    app.run(host="0.0.0.0", threaded=True)
