#!/usr/bin/env bash
set -euo pipefail

echo "Installing openDR dependencies for Raspberry Pi OS (Bookworm/Bullseye)..."

sudo apt-get update
sudo apt-get -y upgrade

sudo apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  python3-opencv \
  python3-picamera2 \
  python3-pigpio \
  python3-flask \
  python3-numpy \
  python3-requests \
  pigpio \
  libcamera-apps \
  chromium-browser \
  nodejs \
  npm

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade imutils

# ── Production WSGI server (waitress) — fundus.py must run in this process
# model since the camera/GPIO/inference-executor state is a singleton; a
# pre-fork server (e.g. gunicorn's default worker) would break that.
python3 -m pip install --upgrade waitress

# ── DR Grad-CAM / glaucoma screening (optional but recommended)
# torch/torchvision are NOT installed by this script (Raspberry Pi wheels
# vary by OS/arch — install the build matching your device separately from
# https://pytorch.org/get-started/locally/). Without them, both explainer
# modules fall back to demo mode with random weights instead of failing.
# `timm` is lightweight and safe to install unconditionally here.
python3 -m pip install --upgrade timm

# ── Tailwind CSS (compiled offline; output committed to static/css/tailwind.css)
echo "Building Tailwind CSS..."
cd /home/pi/openDR || { echo "ERROR: /home/pi/openDR not found. Ensure the repository is cloned there."; exit 1; }
npm install --save-dev tailwindcss
npx tailwindcss -i ./static/css/tailwind.src.css -o ./static/css/tailwind.css --minify
echo "Tailwind CSS built successfully."

# ── Glaucoma screening model (~111 MB, not committed to the repo)
echo "Downloading glaucoma screening model..."
python3 tools/download_glaucoma_model.py || echo "WARNING: glaucoma model download failed; the feature will run in demo mode. Re-run tools/download_glaucoma_model.py later."

echo "Done. Ensure libcamera is enabled and run: python3 /home/pi/openDR/fundus.py"
