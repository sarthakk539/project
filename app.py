import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from backend.compare import compare_single
from backend.multi_pin_bfs import detect_wire_colors_bfs
from backend.reference import DEFAULT_MAX_DELTA_E, color_name
from backend.visualize import annotate

st.set_page_config(page_title="Wire Harness Checker", page_icon="🔌", layout="wide")
st.title("🔌 Wire Harness Checker")
st.caption("Single-strand wire colour order check against a reference (fixed camera, wires vertical).")


def load_image(file):
    if file is None:
        return None
    data = np.frombuffer(file.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def show(img_bgr, caption):
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=caption, width="stretch")


def swatch_html(bgr):
    b, g, r = (int(c) for c in bgr)
    return (f"<span style='display:inline-block;width:42px;height:18px;"
            f"background:rgb({r},{g},{b});border:1px solid #888;border-radius:3px'></span>")


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    n_wires = st.number_input("Number of wires", min_value=1, max_value=40, value=6, step=1)
    max_de = st.slider("Colour tolerance (Delta E)", 5.0, 40.0, float(DEFAULT_MAX_DELTA_E), 0.5,
                       help="Lower = stricter. Same wire across photos is ~2-7; "
                            "different wires are usually 24+.")
    st.markdown("---")
    use_samples = st.checkbox("Use sample images", value=False)
    st.caption("Reference and test must be photographed from the same side with the same camera.")

if "reference" not in st.session_state:
    st.session_state.reference = None   # {"num_wires": int, "colors": [[b,g,r],...]}

tab_ref, tab_test = st.tabs(["1 · Reference", "2 · Check a harness"])

# ---------------- tab 1: reference ----------------
with tab_ref:
    c1, c2 = st.columns(2)
    with c1:
        if use_samples:
            ref_img = cv2.imread("sample_images/reference_sample.png")
        else:
            src = st.radio("Source", ["Upload", "Camera"], horizontal=True, key="ref_src")
            f = (st.file_uploader("Reference photo (known-good harness)", type=["png", "jpg", "jpeg"],
                                  key="ref_up") if src == "Upload"
                 else st.camera_input("Take reference photo", key="ref_cam"))
            ref_img = load_image(f)
        st.markdown("**or load a saved reference**")
        saved = st.file_uploader("reference.json", type=["json"], key="ref_json")
        if saved is not None:
            try:
                d = json.load(saved)
                st.session_state.reference = {"num_wires": int(d["num_wires"]),
                                              "colors": [list(map(int, c)) for c in d["color_sequence"]]}
                st.success("Reference loaded from file.")
            except Exception as e:
                st.error(f"Invalid reference file: {e}")

    with c2:
        if ref_img is not None:
            count, colors, dbg = detect_wire_colors_bfs(ref_img, int(n_wires), return_debug=True)
            if count != int(n_wires) or not colors:
                st.error(f"Detected {count} wires, expected {int(n_wires)}. "
                         "Check the wire count, framing and lighting.")
                if colors:
                    show(annotate(ref_img, colors, dbg), "What the detector saw")
            else:
                show(annotate(ref_img, colors, dbg), "Detected reference")
                st.markdown("Check the colours below are correct, then save:")
                st.markdown(" ".join(f"{i+1}.{swatch_html(c)} {color_name(c)}  "
                                     for i, c in enumerate(colors)), unsafe_allow_html=True)
                if st.button("✅ Use as reference", type="primary"):
                    st.session_state.reference = {"num_wires": count,
                                                  "colors": [[int(x) for x in c] for c in colors]}
                    st.success("Reference saved for this session.")
        else:
            st.info("Provide a reference photo (or tick 'Use sample images').")

    ref = st.session_state.reference
    if ref:
        st.markdown("---")
        st.markdown(f"**Current reference: {ref['num_wires']} wires**")
        st.markdown(" ".join(f"{i+1}.{swatch_html(c)} {color_name(c)}  "
                             for i, c in enumerate(ref["colors"])), unsafe_allow_html=True)
        st.download_button("Download reference.json", json.dumps({
            "num_wires": ref["num_wires"], "color_sequence": ref["colors"],
            "color_names": [color_name(c) for c in ref["colors"]]}, indent=2),
            file_name="reference.json")

# ---------------- tab 2: check ----------------
with tab_test:
    ref = st.session_state.reference
    if not ref:
        st.warning("Set a reference in tab 1 first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if use_samples:
                test_img = cv2.imread("sample_images/test_sample.png")
            else:
                src = st.radio("Source", ["Upload", "Camera"], horizontal=True, key="test_src")
                f = (st.file_uploader("Photo to check", type=["png", "jpg", "jpeg"], key="test_up")
                     if src == "Upload" else st.camera_input("Take photo to check", key="test_cam"))
                test_img = load_image(f)
        with c2:
            if test_img is None:
                st.info("Provide a photo to check.")
            else:
                res = compare_single([ref["num_wires"], ref["colors"]], test_img, max_delta_e=max_de)
                _, cols, dbg = detect_wire_colors_bfs(test_img, ref["num_wires"], return_debug=True)
                rows = res["wire_comparisons"]
                if res["match"]:
                    st.success("✅ PASS: wire count and colours match the reference.")
                else:
                    st.error("❌ FAIL: " + res["details"])
                if cols:
                    show(annotate(test_img, cols, dbg, rows or None), "Detected wires (X = mismatch)")
        if test_img is not None and res["wire_comparisons"]:
            df = pd.DataFrame([{
                "Wire": r["wire"], "Expected": r["expected"], "Detected": r["detected"],
                "Delta E": r["distance"], "Result": "OK" if r["match"] else "FAIL"}
                for r in res["wire_comparisons"]])
            st.dataframe(df.style.apply(
                lambda row: ["background-color:#5c1f1f" if row["Result"] == "FAIL" else "" for _ in row],
                axis=1), hide_index=True, width="stretch")
