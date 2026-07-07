# data/demo_images/

Five synthetic placeholder "face" images, procedurally drawn with PIL
primitives (ovals, arcs, lines) by `scripts/generate_demo_images.py`. They
depict no real person, are not sourced from any dataset, and carry no
license restrictions -- safe to commit and reuse.

They exist so `scripts/run_demo.py` has something to show during a live
course demonstration without ever committing UTKFace (or any other
license-restricted dataset) images to this repository.

**Limitation, stated up front:** these are cartoon-style shapes, not
photographic human faces. The classical Haar-cascade face detector
(`src/inference/face_detection.py`) may or may not recognize a "face" in
any given one. If it doesn't, the API correctly declines to predict
(age/gender/Grad-CAM all `None`) -- that is expected behavior, not a bug,
and is itself a valid demonstration of the system's "decline rather than
guess" safety design. For a demo that reliably produces a full prediction,
supply your own consented photo through the frontend at demo time instead
of relying on these placeholders.

Regenerate at any time with:

```
python scripts/generate_demo_images.py
```
