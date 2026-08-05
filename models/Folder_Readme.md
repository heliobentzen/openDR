This folder holds the glaucoma screening model checkpoint (`convnext_tiny.pt`)
and its config (`convnext_tiny.json`).

The `.pt` checkpoint (~111 MB) is not committed to this repository — fetch it
with:

```
python tools/download_glaucoma_model.py
```

Without it, `modules.glaucoma` runs in demo mode with random weights instead
of failing, so the rest of the app still works.
