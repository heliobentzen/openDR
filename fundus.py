##############################################################################
##  OWL v3.0                                                          ########
## ------------------------------------------------------------       ########
##  Authors: Ayush Yadav, Devesh Jain, Ebin Philip, Dhruv Joshi      ########
##############################################################################

import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

import cv2
from flask import Flask, Response, abort, redirect, render_template, request, send_from_directory, url_for

from Fundus_Cam import Fundus_Cam
from modules.process import grade, grade_with_explanation

try:
    import pigpio
except ImportError:  # pragma: no cover - depends on Raspberry Pi runtime
    pigpio = None

app = Flask(__name__)
BASE_FOLDER = Path(os.environ.get("OPEN_DR_BASE", "/home/pi/openDR")).resolve()
TOKENS = ["Flip", "Vid", "Click", "Switch", "Grade", "Explain", "Shut"]
PATIENT_ID_RE = re.compile(r"^[A-Z0-9_-]{1,64}$")
FOCUS_WARNING_MESSAGE = "Posicione o paciente e foque antes de capturar"
MIN_FOCUS_SCORE = 140
DARK_PIXEL_THRESHOLD = 58
MIN_DARK_PIXELS = 120
MIN_DARK_DENSITY = 0.20
MIN_DARK_RATIO = 0.01
MAX_DARK_RATIO = 0.42
PREVIEW_MIN_INTERVAL_S = 0.20

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
preview_rate_lock = Lock()
preview_last_request = {}


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
        if request.form.get("focus_ok") != "1":
            return render_capture(FOCUS_WARNING_MESSAGE)
        with state.lock:
            if state.camera is None or not state.patient_id:
                return render_capture("NO ACTIVE CAPTURE SESSION")
            image = state.camera.capture()
            if not is_eye_in_focus(image):
                return render_capture(FOCUS_WARNING_MESSAGE)
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

    if d == "Explain":
        with state.lock:
            last_img = state.last_img
        if last_img is None:
            return render_capture("NO IMAGE SPECIFIED")

        try:
            result = grade_with_explanation(last_img)
        except RuntimeError as exc:
            return render_capture(f"GRAD-CAM ERROR: {exc}")

        grade_result = str(result["theia_grade"])[:4]
        gradcam_record = result["gradcam"]
        overlay_filename = Path(gradcam_record["gradcam_overlay"]).name
        json_filename = Path(gradcam_record["gradcam_audit_json"]).name
        dr_label = gradcam_record["predicted_dr_grade"]["label"]
        confidence = gradcam_record["predicted_dr_grade"]["confidence"]
        lesion_count = len(gradcam_record["lesion_regions"])
        print(f"Grad-CAM complete: lesions detected={lesion_count}")
        return render_capture(
            grade_result,
            overlay_filename=overlay_filename,
            json_filename=json_filename,
            dr_label=dr_label,
            confidence=confidence,
            lesion_count=lesion_count,
        )

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


@app.route("/preview-frame", methods=["GET"])
def preview_frame():
    client_key = request.remote_addr or "unknown"
    now = time.monotonic()
    with preview_rate_lock:
        last_request = preview_last_request.get(client_key, 0.0)
        if now - last_request < PREVIEW_MIN_INTERVAL_S:
            abort(429)
        preview_last_request[client_key] = now

    with state.lock:
        if state.camera is None:
            abort(404)
        preview = state.camera.capture_preview()
    response = Response(preview.tobytes(), mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


def is_eye_in_focus(image_buffer):
    gray_frame = cv2.imdecode(image_buffer, cv2.IMREAD_GRAYSCALE)
    if gray_frame is None:
        return False

    resized = cv2.resize(gray_frame, (176, 132), interpolation=cv2.INTER_AREA)
    focus_score = cv2.Laplacian(resized, cv2.CV_64F).var()

    _, dark_regions = cv2.threshold(
        resized,
        DARK_PIXEL_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV,
    )
    dark_pixel_count = int(cv2.countNonZero(dark_regions))
    if dark_pixel_count < MIN_DARK_PIXELS:
        return False

    contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    x, y, width, height = cv2.boundingRect(largest_contour)
    bounding_area = max(1, width * height)
    dark_density = contour_area / bounding_area
    dark_ratio = dark_pixel_count / float(resized.shape[0] * resized.shape[1])
    has_eye_contour = (
        dark_density > MIN_DARK_DENSITY
        and dark_ratio > MIN_DARK_RATIO
        and dark_ratio < MAX_DARK_RATIO
    )
    return focus_score >= MIN_FOCUS_SCORE and has_eye_contour


def render_capture(
    grade_message="",
    overlay_filename=None,
    json_filename=None,
    dr_label=None,
    confidence=None,
    lesion_count=None,
):
    return render_template(
        "capture_simple.html",
        params=TOKENS,
        grades={"grade": grade_message},
        focus_warning_message=FOCUS_WARNING_MESSAGE,
        overlay_filename=overlay_filename,
        json_filename=json_filename,
        dr_label=dr_label,
        confidence=confidence,
        lesion_count=lesion_count,
    )


@app.route("/images/<path:filename>")
def serve_image(filename):
    """Serve captured and processed images from the patient images directory.

    The filename is validated to reject path-traversal attempts before
    handing it to :func:`~flask.send_from_directory`.
    """
    # Extract only the bare filename component to prevent traversal outside
    # the images directory (e.g. "../../etc/passwd" → rejected).
    safe_name = Path(filename).name
    if safe_name != filename:
        app.logger.warning("Rejected path-traversal attempt in /images: %s", filename)
        abort(400)
    images_dir = BASE_FOLDER / "images"
    # Confirm the resolved path stays inside the images directory.
    resolved = (images_dir / safe_name).resolve()
    if not str(resolved).startswith(str(images_dir.resolve())):
        app.logger.warning("Rejected out-of-directory request in /images: %s", filename)
        abort(400)
    return send_from_directory(str(images_dir), safe_name)


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
    image_identifier = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S%f')}_{uuid4().hex}"
    )
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
            try:
                controller.stop()
            except Exception:  # pragma: no cover - best-effort cleanup on hardware init
                pass
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
