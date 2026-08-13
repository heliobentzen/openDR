# Changelog

## [4.1.0] - 2026-07-16
### Added
- **Glaucoma screening**: new `modules/glaucoma.py` runs a ConvNeXt-Tiny classifier (ported from the `tiagopessoalim/glaucoma` Hugging Face Space) for referable-glaucoma probability plus 10 morphological signs, with its own Grad-CAM explanation. Exposed as a new "Glaucoma" action alongside the existing DR "Relatório" (Explain) button, with its own progress tracker and result panel (verdict, feature table, before/after Grad-CAM comparison slider).
- `tools/download_glaucoma_model.py` fetches the glaucoma checkpoint (~111 MB, not committed to the repo) via a streaming HTTP GET.
- Photo upload (`/upload-image`): analyse an existing fundus photo without the camera, with drag-and-drop, client-side preview, and format/size validation in the capture UI.
- Patient sessions can now start without a working camera (upload-only mode) — a banner and disabled camera controls make this explicit instead of failing outright.
- **Eye laterality (OD/OS)**: the capture UI now has an OD/Direito · OS/Esquerdo toggle; every capture (camera click, video, upload) is tagged with the selected eye, encoded in the saved filename and surfaced in the patient gallery (thumbnail tooltip and detail panel).
- The session-start screen now suggests existing patient ids (`<datalist>` autocomplete, backed by a new `list_known_patient_ids()` scan of the images directory) so a returning patient is matched to their existing history instead of a differently-typed id fragmenting it.

### Changed
- Migrated `fundus.py` from `Fundus_Cam` to `RetinaCamera` (contrast-enhanced capture, typed hardware exceptions, graceful degradation on camera faults).
- Generalised the background inference-job tracker (`fundus.py`) so DR-explain and glaucoma screening share the same job bookkeeping and `/inference-status/<job_id>` endpoint instead of duplicating it.
- Added a process-wide lock so DR-explain and glaucoma screening never run their CPU-heavy CNN passes at the same time on the same device.
- Extracted the Grad-CAM heatmap-overlay blend into a shared `modules/heatmap.py`, reused by both the DR and glaucoma explainers.
- `theia.grade_request` now has a network timeout, so an unresponsive Theia API can no longer hang an inference worker indefinitely.
- The patient capture gallery caches its metadata (invalidated by directory mtime) instead of re-reading every report file on each page.
- Removed unused legacy front-end assets (jQuery/Bootstrap/FontAwesome, orphaned templates and images) that predated the Tailwind UI.
- `fundus.py` is now served by [waitress](https://docs.pylonsproject.org/projects/waitress/) instead of Flask's development server, falling back to the dev server with a warning if `waitress` isn't installed.
- `modules.process.prepare_processed_image` no longer normalises the brightness/contrast/threshold settings twice per request, and skips the full-resolution brightness/contrast pass entirely when the sliders are left at their defaults (the common case).

### Removed
- `Fundus_Cam.py` (superseded by `RetinaCamera`) and the unused `modules.extract.extract_fundus` file-path wrapper (superseded by `extract_fundus_from_image`).

### Fixed
- `OPEN_DR_INFERENCE_WORKERS` / `OPEN_DR_MAX_INFERENCE_JOB_HISTORY` no longer crash the whole app at startup on a non-numeric value; they log a warning and fall back to their defaults instead.
- The session-start form (`POST /`) no longer 500s when submitted without its `text` field.
- `theia.grade_request` now tolerates a malformed Theia response (non-JSON body, unexpected JSON shape, missing/non-numeric `grade`) by returning `-1` instead of raising, so a bad API response can no longer leave a background inference job stuck in "running" forever.
- `modules.glaucoma` rejects an out-of-range `IMG_SIZE` in `convnext_tiny.json` (e.g. from a hand-edited/corrupted config) instead of risking a degenerate resize or an OOM-inducing allocation.
- `modules.gradcam` now falls back to demo mode on a corrupt/truncated Grad-CAM checkpoint instead of crashing the explanation step, mirroring the existing `modules.glaucoma` behaviour.
- `tools/download_glaucoma_model.py` now verifies the downloaded byte count against `Content-Length` before installing the file, so a truncated download can no longer be silently treated as a valid checkpoint.

## [4.0.1] - 2026-07-01
### Fixed
- **Guided Grad-CAM**: temporarily disables in-place ReLU operations during guided backpropagation to prevent "view is being modified inplace" backward-hook errors; the guided-gradient masking now exactly follows Springenberg et al. (2015).

### Changed
- **Lesion region extraction**: replaced the fixed activation threshold with adaptive Otsu thresholding (with a configurable safety floor) for more robust detection across varying image exposures.
- **Lesion splitting**: added watershed-based separation of touching lesion clusters via distance transform, correctly breaking merged blobs into distinct regions.
- **Noise reduction**: morphological open/close passes applied before connected-component analysis to eliminate small artefacts from the binary mask.
- **Richer lesion metrics**: each detected region now reports `circularity` (shape roundness, 0–1) and `relative_intensity` (mean activation relative to the whole-image mean) in addition to bounding-box coordinates.
- Grad-CAM audit JSON schema version bumped to `1.2`.

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
