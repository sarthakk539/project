# Wire Harness Checker (Streamlit)

Checks the colour order of a single-strand wire harness against a reference photo.

## Run locally
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (app.py and requirements.txt in the repo root).
2. Go to https://share.streamlit.io -> New app -> pick the repo, branch, main file `app.py`.
3. Deploy. First build takes a few minutes.

## Command line (no UI)
    python examples/run_demo.py detect  sample_images/reference_sample.png 6
    python examples/run_demo.py compare sample_images/reference_sample.png sample_images/test_sample.png 6

## Tuning (backend/multi_pin_bfs.py, backend/reference.py)
- `NOMINAL_WIRE_PX` (21): typical wire width in pixels. Re-measure if camera distance changes
  (`px_per_wire` from `detect_wire_colors_bfs(..., return_debug=True)`).
- `DEFAULT_MAX_DELTA_E` (15): colour tolerance. Also adjustable in the app sidebar.
- `BAND_FROM_END`: rows above the connector where wires are read.

Assumes: fixed camera, wires roughly vertical, light background, same side photographed for
reference and test. Double-strand (front/back) is not implemented yet.
