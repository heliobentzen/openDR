## Open Indirect Ophthalmoscope (OIO / OWL)
_Authors: Ayush Yadav, Ebin Philip, Dhruv Joshi_

Built at the Srujana Center for Innovation, LV Prasad Eye Institute, Hyderabad, India.

## LICENSE AND DISCLAIMER
This software is covered by the MIT License (see `license.txt`).

## Version
Current release: **3.0.0**

## Supported Platform
- Raspberry Pi 4
- Latest Raspberry Pi OS (Bullseye/Bookworm, 64-bit recommended)
- libcamera stack (`Picamera2`) for camera access

## Runtime Dependencies
- Python 3.11+
- OpenCV 4.x (`python3-opencv`)
- Picamera2 / libcamera (`python3-picamera2`, `libcamera-apps`)
- Flask, NumPy, Requests, pigpio

## Installation
Clone into `/home/pi/openDR` on the Raspberry Pi:

```bash
cd /home/pi
git clone <repo-url> openDR
cd /home/pi/openDR
bash install.sh
```

## Running the application
```bash
cd /home/pi/openDR
python3 fundus.py
```

## Notes on camera migration
- Legacy `picamera` usage has been replaced with `Picamera2` (libcamera backend).
- Vertical camera flipping is now handled in software before JPEG encoding.

## Folder structure
`images` contains captured patient/session images (patient ID included in filenames).  
`modules` contains image processing and grading modules.  
`static` and `templates` provide Flask UI assets.
