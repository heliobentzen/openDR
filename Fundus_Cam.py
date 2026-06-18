##############################################################################
#########################  Fundus_Cam.py             #########################
#########################  Primary Author: Ebin      #########################
#########################  Version : 3.0             #########################
#########################  Contributor: Ayush Yadav  #########################
##############################################################################

from threading import Thread

import cv2
import numpy as np
from picamera2 import Picamera2

###############################################################################
### This class provides access to the libcamera stack using Picamera2       ###
###############################################################################


class Fundus_Cam(object):
    def __init__(self, framerate=12, preview=False, max_frames=10):
        self.camera = Picamera2()
        sensor_resolution = self.camera.sensor_resolution
        camera_config = self.camera.create_still_configuration(
            main={"size": sensor_resolution},
            controls={"FrameRate": framerate},
        )
        self.camera.configure(camera_config)
        self.camera.start()

        self.flip_state = False
        self.images = []
        self.stopped = False
        self.image = None
        self.max_frames = max_frames
        self.capture_thread = None

        if preview:
            self.preview()

    def _capture_jpeg_buffer(self):
        frame = self.camera.capture_array()
        if self.flip_state:
            frame = cv2.flip(frame, 0)

        encoded, image_bytes = cv2.imencode(".jpg", frame)
        if not encoded:
            raise RuntimeError(
                "Unable to encode camera frame as JPEG. Check camera connection and frame data."
            )
        return np.frombuffer(image_bytes.tobytes(), dtype=np.uint8)

    def continuous_capture(self):
        self.stopped = False
        self.images = []
        self.capture_thread = Thread(target=self.update, args=())
        self.capture_thread.start()

    def update(self):
        while True:
            self.images.append(self._capture_jpeg_buffer())
            if len(self.images) >= self.max_frames:
                self.stopped = True
                return

    def wait_for_capture(self, timeout=5):
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=timeout)
        return self.stopped

    def flip_cam(self):
        self.flip_state = not self.flip_state

    def capture(self):
        self.image = self._capture_jpeg_buffer()
        return self.image

    def preview(self):
        # Picamera2 preview windows are not managed by this Flask service.
        return

    def stop_preview(self):
        self.stop()

    def stop(self):
        self.stopped = True
        if self.camera.started:
            self.camera.stop()
