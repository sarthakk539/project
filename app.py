import os
import sys
import traceback

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import streamlit as st

st.set_page_config(page_title="Wire Harness Checker", page_icon="🔌", layout="wide")

# Streamlit Cloud hides import errors. Catch them and show the real cause on the page.
try:
    import cv2
    import numpy as np
    import pandas as pd

    from backend.reference import DEFAULT_MAX_DELTA_E
    from backend.verdict import check_harness, REVIEW_BELOW
    from backend.visualize import annotate
except Exception:
    st.error("The app could not start because of an import error. Details below.")
    st.code(traceback.format_exc())
    st.write("App folder:", APP_DIR)
    st.write("Files here:", sorted(os.listdir(APP_DIR)))
    bdir = os.path.join(APP_DIR, "backend")
    st.write("backend/ exists:", os.path.isdir(bdir),
             "| files:", sorted(os.listdir(bdir)) if os.path.isdir(bdir) else "-")
    st.stop()

st.title("🔌 Wire Harness Checker")
st.caption("Give a reference photo, a photo to check and the number of wires. "
           "You get RIGHT or WRONG with a confidence score.")


def load_image(file):
    if file is None:
        return None
    return cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)


def show(img_bgr, caption):
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=caption, width="stretch")


with st.sidebar:
    st.header("Settings")
    max_de = st.slider("Colour tolerance (Delta E)", 5.0, 40.0, float(DEFAULT_MAX_DELTA_E), 0.5,
                       help="Lower = stricter. Same wire across photos is ~2-7; "
                            "different wires are usually 24+.")
    use_samples = st.checkbox("Use sample images")
    st.caption("Photograph reference and test from the same side with the same camera.")

col_a, col_b, col_c = st.columns([2, 2, 1])
with col_a:
    st.subheader("1 · Reference image")
    ref_file = None if use_samples else st.file_uploader(
        "Known-good harness", type=["png", "jpg", "jpeg"], key="ref")
with col_b:
    st.subheader("2 · Input image")
    test_file = None if use_samples else st.file_uploader(
        "Harness to check", type=["png", "jpg", "jpeg"], key="test")
with col_c:
    st.subheader("3 · Wires")
    n_wires = st.number_input("Number of wires", min_value=1, max_value=40, value=6, step=1)

if use_samples:
    ref_img = cv2.imread(os.path.join(APP_DIR, "sample_images", "reference_sample.png"))
    test_img = cv2.imread(os.path.join(APP_DIR, "sample_images", "test_sample.png"))
else:
    ref_img, test_img = load_image(ref_file), load_image(test_file)

st.markdown("---")
if ref_img is None or test_img is None:
    st.info("Upload both images (or tick 'Use sample images' in the sidebar).")
    st.stop()

res = check_harness(ref_img, test_img, int(n_wires), max_delta_e=max_de)
conf = res["confidence"]

head1, head2 = st.columns([1, 2])
with head1:
    if res.get("reference_error"):
        st.warning("⚠️ REFERENCE PROBLEM")
    elif res["verdict"] == "RIGHT":
        st.success("## ✅ RIGHT")
    else:
        st.error("## ❌ WRONG")
with head2:
    st.metric("Confidence", f"{conf * 100:.0f}%")
    st.progress(float(conf))
    if res["needs_review"]:
        st.caption(f"⚠️ Below {REVIEW_BELOW * 100:.0f}%: please review this one manually.")
    st.caption("Confidence is a heuristic score, not a calibrated probability.")
st.write(res["reason"])

if "_test_colors" in res:
    rows = res["wires"]
    i1, i2 = st.columns(2)
    with i1:
        show(annotate(ref_img, res["_ref_colors"], res["_ref_dbg"]), "Reference (detected wires)")
    with i2:
        show(annotate(test_img, res["_test_colors"], res["_test_dbg"], rows or None),
             "Input (detected wires, X = mismatch)")
    if rows:
        df = pd.DataFrame([{
            "Wire": r["wire"], "Expected": r["expected"], "Detected": r["detected"],
            "Delta E": r["delta_e"], "Match chance": f"{r['match_probability'] * 100:.0f}%",
            "Result": "OK" if r["match"] else "FAIL"} for r in rows])
        st.dataframe(df.style.apply(
            lambda row: ["background-color:#5c1f1f" if row["Result"] == "FAIL" else "" for _ in row],
            axis=1), hide_index=True, width="stretch")
