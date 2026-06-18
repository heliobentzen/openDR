# Changelog

## [3.0.0] - 2026-06-18
### Changed
- Migrated runtime from Python 2 to Python 3 syntax in core application modules.
- Replaced legacy `picamera` camera integration with `Picamera2` (libcamera backend) for Raspberry Pi OS compatibility.
- Updated installation flow to modern Raspberry Pi OS packages, including OpenCV 4 and libcamera dependencies.
- Updated path handling in processing and Theia modules using `OPEN_DR_BASE` with `/home/pi/openDR` default.

### Documentation
- Revised README for current Raspberry Pi 4, Python 3, OpenCV 4, and libcamera-based setup.
