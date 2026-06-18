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
  chromium-browser

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade imutils

echo "Done. Ensure libcamera is enabled and run: python3 /home/pi/openDR/fundus.py"
