# Changelog

## [4.0.0] - 2026-06-28
### Added
- `RetinaCamera` class with lifecycle management, hardware error handling, and CLAHE contrast enhancement.
- Grad-CAM explainability module: visual saliency maps overlaid on retinal images to show DR classification focus areas.
- Live inference progress workflow: grading status streamed to the UI via background executor with clear error and completion states.
- Picamera2 `/preview-frame` endpoint for low-latency JPEG live preview.
- Client-side focus gating in the capture UI: capture button only enabled when sharpness threshold is met; backend revalidates focus before saving.

### Changed
- All HTML templates rebuilt with Tailwind CSS for a modern, responsive interface.
- Start screen and action buttons redesigned with consistent icon-based layout.
- Processing and module files fully annotated with Python 3.11 type hints and docstrings.

### Security
- Sanitised error messages to avoid leaking internal paths to the UI.
- Strengthened `serve_image` path validation to prevent directory traversal.

### Authors
- Original: Ayush Yadav, Ebin Philip, Dhruv Joshi
- Maintained by: @heliobentzen, GitHub Copilot

## [3.0.0] - 2026-06-18
### Changed
- Migrated runtime from Python 2 to Python 3 syntax in core application modules.
- Replaced legacy `picamera` camera integration with `Picamera2` (libcamera backend) for Raspberry Pi OS compatibility.
- Updated installation flow to modern Raspberry Pi OS packages, including OpenCV 4 and libcamera dependencies.
- Updated path handling in processing and Theia modules using `OPEN_DR_BASE` with `/home/pi/openDR` default.

### Documentation
- Revised README for current Raspberry Pi 4, Python 3, OpenCV 4, and libcamera-based setup.
