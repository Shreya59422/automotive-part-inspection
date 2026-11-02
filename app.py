import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from backend.patchcore_service import predict_anomaly
from backend.image_processing import create_heatmap, create_overlay
from backend.database import (
    initialize_database,
    save_inspection,
    get_all_inspections,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Automotive Part Inspection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .sub-title {
        color: #9CA3AF;
        font-size: 14px;
        margin-bottom: 25px;
    }
    .status-card {
        padding: 16px;
        border-radius: 10px;
        background-color: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.2);
    }
    .small-note {
        font-size: 12px;
        color: #9CA3AF;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128,128,128,0.07);
        border: 1px solid rgba(128,128,128,0.15);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. PATHS AND PERSISTENT INSPECTION STORAGE
# ============================================================

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "inspection_artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = ARTIFACT_DIR / "latest_inspection.json"
ORIGINAL_FILE = ARTIFACT_DIR / "original_image.png"
HEATMAP_FILE = ARTIFACT_DIR / "heatmap.png"
OVERLAY_FILE = ARTIFACT_DIR / "overlay.png"

ARTIFACT_FILES = [
    METADATA_FILE,
    ORIGINAL_FILE,
    HEATMAP_FILE,
    OVERLAY_FILE,
]


def to_rgb_pil(data):
    """Convert PIL or NumPy image data into a valid RGB image."""

    if isinstance(data, Image.Image):
        return data.convert("RGB")

    arr = np.asarray(data)
    arr = np.nan_to_num(arr)

    if arr.size == 0:
        raise ValueError("An image array is empty.")

    if np.issubdtype(arr.dtype, np.floating):
        if arr.min() >= 0 and arr.max() <= 1:
            arr = arr * 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")

    if arr.ndim == 3:
        # Convert channel-first images to channel-last.
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        return Image.fromarray(arr).convert("RGB")

    raise ValueError(f"Unsupported image dimensions: {arr.shape}")


def save_pending_inspection(image, result, heatmap, overlay):
    """
    Save the latest AI inspection to disk.

    These files remain available when the user navigates between pages
    and after Human Verification has been saved.
    """

    metadata = {
        "anomaly_score": float(result["anomaly_score"]),
        "predicted_anomaly": bool(result["predicted_anomaly"]),
        "ai_signal": str(result["ai_signal"]),
        "decision": str(result["decision"]),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "review_saved": False,
    }

    to_rgb_pil(image).save(ORIGINAL_FILE, format="PNG")
    to_rgb_pil(heatmap).save(HEATMAP_FILE, format="PNG")
    to_rgb_pil(overlay).save(OVERLAY_FILE, format="PNG")

    # Write metadata atomically to avoid partially written JSON.
    temporary_file = METADATA_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    temporary_file.replace(METADATA_FILE)


def mark_review_as_saved():
    """Record that Human Verification has been saved."""

    if not METADATA_FILE.exists():
        return

    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        metadata["review_saved"] = True

        temporary_file = METADATA_FILE.with_suffix(".tmp")

        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        temporary_file.replace(METADATA_FILE)

    except Exception as error:
        st.warning(f"Could not update review status: {error}")


def load_pending_inspection():
    """Load the latest saved AI inspection and all its image evidence."""

    if not all(path.exists() for path in ARTIFACT_FILES):
        return None

    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        with Image.open(ORIGINAL_FILE) as file:
            original = file.convert("RGB").copy()

        with Image.open(HEATMAP_FILE) as file:
            heatmap = np.asarray(file.convert("RGB")).copy()

        with Image.open(OVERLAY_FILE) as file:
            overlay = np.asarray(file.convert("RGB")).copy()

        result = {
            "anomaly_score": float(metadata["anomaly_score"]),
            "predicted_anomaly": bool(metadata["predicted_anomaly"]),
            "ai_signal": str(metadata["ai_signal"]),
            "decision": str(metadata["decision"]),
        }

        return {
            "result": result,
            "image": original,
            "heatmap": heatmap,
            "overlay": overlay,
            "timestamp": metadata.get("timestamp", ""),
            "review_saved": bool(metadata.get("review_saved", False)),
        }

    except Exception as error:
        print(f"Could not load saved inspection: {error}")
        return None


def restore_pending_inspection():
    """Restore saved AI results into Streamlit session state."""

    saved = load_pending_inspection()

    if saved is None:
        return False

    st.session_state["inspection_result"] = saved["result"]
    st.session_state["inspection_image"] = saved["image"]

    st.session_state["inspection_evidence"] = {
        "heatmap": saved["heatmap"],
        "overlay": saved["overlay"],
    }

    st.session_state["inspection_timestamp"] = saved["timestamp"]
    st.session_state["review_saved"] = saved["review_saved"]

    return True


# ============================================================
# 3. DATABASE AND SESSION STATE
# ============================================================

initialize_database()

POSITIONS = ["P1", "P2", "P3", "P4"]

POSITION_OPTIONS = [
    "Not verified",
    "Present",
    "Missing",
    "Needs Review",
]

DEFAULT_STATUSES = {
    position: "Not verified"
    for position in POSITIONS
}


# Initialize non-widget session state.
if "inspection_result" not in st.session_state:
    st.session_state["inspection_result"] = None

if "inspection_image" not in st.session_state:
    st.session_state["inspection_image"] = None

if "inspection_evidence" not in st.session_state:
    st.session_state["inspection_evidence"] = {}

if "inspection_timestamp" not in st.session_state:
    st.session_state["inspection_timestamp"] = ""

if "review_saved" not in st.session_state:
    st.session_state["review_saved"] = False

if "position_statuses" not in st.session_state:
    st.session_state["position_statuses"] = DEFAULT_STATUSES.copy()


# Initialize widget keys BEFORE their selectboxes are created.
for prefix in ["human_position", "xr_position"]:
    for position in POSITIONS:
        key = f"{prefix}_{position}"

        if key not in st.session_state:
            st.session_state[key] = "Not verified"


# Restore saved inspection after app restart or page navigation.
if st.session_state["inspection_result"] is None:
    restore_pending_inspection()


# ============================================================
# 4. COMPONENT STATUS MANAGEMENT
# ============================================================

def get_current_statuses(prefix):
    """Read component statuses directly from their selectbox keys."""

    statuses = {}

    for position in POSITIONS:
        key = f"{prefix}_{position}"
        value = st.session_state.get(key, "Not verified")

        if value not in POSITION_OPTIONS:
            value = "Not verified"

        statuses[position] = value

    st.session_state["position_statuses"] = statuses

    return statuses


def reset_position_statuses():
    """
    Reset widget values.

    This function is used as a button callback or before the next
    inspection's widgets are created. Do not call it after rendering
    the corresponding selectboxes in the same Streamlit run.
    """

    st.session_state["position_statuses"] = DEFAULT_STATUSES.copy()

    for prefix in ["human_position", "xr_position"]:
        for position in POSITIONS:
            key = f"{prefix}_{position}"
            st.session_state[key] = "Not verified"


def render_position_selector(prefix):
    """
    Render P1-P4 selectboxes.

    Important: do not assign to the widget's session-state key
    after st.selectbox() has been created.
    """

    st.markdown("### Component Verification")

    columns = st.columns(4)

    for index, position in enumerate(POSITIONS):
        key = f"{prefix}_{position}"

        # Set a default only before creating the widget.
        if key not in st.session_state:
            st.session_state[key] = "Not verified"

        with columns[index]:
            st.selectbox(
                position,
                options=POSITION_OPTIONS,
                key=key,
            )

    return get_current_statuses(prefix)


def render_position_summary(statuses):
    columns = st.columns(4)

    for index, position in enumerate(POSITIONS):
        status = statuses[position]

        with columns[index]:
            if status == "Present":
                st.success(f"{position}: Present")

            elif status == "Missing":
                st.error(f"{position}: Missing")

            elif status == "Needs Review":
                st.warning(f"{position}: Needs Review")

            else:
                st.info(f"{position}: Not verified")


# ============================================================
# 5. SHARED DISPLAY HELPERS
# ============================================================

def render_ai_summary():
    result = st.session_state.get("inspection_result")

    if result is None:
        st.info("No AI inspection is currently available.")
        return

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Anomaly Score",
        f"{float(result['anomaly_score']):.4f}",
    )

    c2.metric(
        "AI Signal",
        result["ai_signal"],
    )

    c3.metric(
        "Suggested Action",
        result["decision"],
    )


def render_model_information():
    with st.expander("Model Information"):
        st.caption(
            "The PatchCore model is trained on the MVTec AD screw "
            "category. It identifies visual deviations from normal "
            "training images. The anomaly score is not a defect "
            "probability and does not independently confirm a missing "
            "automotive component. Human verification is required."
        )


def render_inspection_evidence(show_original=True):
    """Display original image, heatmap, and overlay."""

    evidence = st.session_state.get("inspection_evidence", {})
    original = st.session_state.get("inspection_image")

    if show_original:
        st.markdown("### Original Inspection Image")

        if original is not None:
            st.image(
                original,
                caption="Uploaded inspection image",
                width="stretch",
            )
        else:
            st.warning("Original inspection image is unavailable.")

    st.markdown("### AI Inspection Evidence")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Anomaly Heatmap")

        heatmap = evidence.get("heatmap")

        if heatmap is not None:
            st.image(
                heatmap,
                caption="Relative anomaly response",
                width="stretch"
            )
        else:
            st.warning("Heatmap is unavailable.")

    with right:
        st.markdown("#### Anomaly Overlay")

        overlay = evidence.get("overlay")

        if overlay is not None:
            st.image(
                overlay,
                caption="Original image with anomaly overlay",
                width="stretch",
            )
        else:
            st.warning("Overlay is unavailable.")

    st.caption(
        "Highlighted regions represent relatively higher anomaly "
        "responses. They are not automatic component labels."
    )


# ============================================================
# 6. 3D MOUNTING BRACKET VIEW
# ============================================================

def render_bracket_3d(statuses):
    """
    Display a schematic 3D mounting bracket.

    P1-P4 are manually verified checkpoints. They are not automatically
    localized by the PatchCore model.
    """

    fig = go.Figure()

    # Mounting plate surface.
    fig.add_trace(
        go.Mesh3d(
            x=[-3, 3, 3, -3],
            y=[-2, -2, 2, 2],
            z=[0, 0, 0, 0],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#64748b",
            opacity=0.85,
            name="Mounting Plate",
            showscale=False,
        )
    )

    marker_positions = {
        "P1": (-2, -1.2, 0.18),
        "P2": (2, -1.2, 0.18),
        "P3": (-2, 1.2, 0.18),
        "P4": (2, 1.2, 0.18),
    }

    status_colors = {
        "Present": "#16a34a",
        "Missing": "#dc2626",
        "Needs Review": "#f59e0b",
        "Not verified": "#06b6d4",
    }

    for position, (x, y, z) in marker_positions.items():
        status = statuses.get(position, "Not verified")
        color = status_colors.get(status, "#06b6d4")

        fig.add_trace(
            go.Scatter3d(
                x=[x],
                y=[y],
                z=[z],
                mode="markers+text",
                marker={
                    "size": 15,
                    "color": color,
                    "line": {
                        "color": "white",
                        "width": 2,
                    },
                },
                text=[position],
                textposition="top center",
                name=f"{position} - {status}",
            )
        )

    fig.update_layout(
        title="3D Mounting Bracket – Component Checkpoints",
        height=540,
        margin={
            "l": 0,
            "r": 0,
            "t": 55,
            "b": 0,
        },
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "manual",
            "aspectratio": {
                "x": 1.5,
                "y": 1,
                "z": 0.25,
            },
            "camera": {
                "eye": {
                    "x": 1.6,
                    "y": 1.6,
                    "z": 1.2,
                }
            },
        },
        legend={"orientation": "h"},
    )

    st.plotly_chart(
        fig,width="stretch",
        key="mounting_bracket_3d",
    )

    st.caption(
        "Schematic 3D layout. P1–P4 are manual verification "
        "checkpoints and are not automatically mapped defect locations."
    )


# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🔍 Inspection System")
st.sidebar.caption("Automotive Part Inspection")

PAGES = [
    "Dashboard",
    "AI Vision",
    "Human Verification",
    "XR Inspection",
    "Inspection History",
]

page = st.sidebar.radio(
    "Navigation",
    PAGES,
    key="main_navigation",
)

st.sidebar.divider()
st.sidebar.caption("AI-assisted industrial inspection")
st.sidebar.caption("Human verification supported")


# ============================================================
# 8. DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">Inspection Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">'
        'Overview of inspection activity and verification status'
        '</div>',
        unsafe_allow_html=True,
    )

    records = get_all_inspections() or []

    total = len(records)

    pending = sum(
        1 for row in records
        if row.get("decision") == "Manual Review"
    )

    reviewed = total - pending

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Inspections", total)
    c2.metric("Reviewed", reviewed)
    c3.metric("Pending Review", pending)

    st.divider()
    st.markdown("### Inspection Workflow")

    workflow = [
        ("01", "Upload Image", "Capture or upload a part image"),
        ("02", "AI Analysis", "Analyze visual anomalies"),
        ("03", "Human Review", "Verify component positions"),
        ("04", "Save Results", "Store inspection history"),
    ]

    columns = st.columns(4)

    for index, (number, title, description) in enumerate(workflow):
        with columns[index]:
            st.markdown(
                f"""
                <div class="status-card">
                    <h3>{number}</h3>
                    <b>{title}</b>
                    <p class="small-note">{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### Latest Inspection")

    if records:
        latest = records[-1]

        a, b = st.columns(2)

        with a:
            st.write(
                f"**Part ID:** {latest.get('part_id', 'N/A')}"
            )
            st.write(
                f"**Operator:** {latest.get('operator', 'N/A')}"
            )

        with b:
            st.write(
                f"**Decision:** {latest.get('decision', 'N/A')}"
            )
            st.write(
                f"**AI Signal:** {latest.get('ai_signal', 'N/A')}"
            )

        st.write(
            f"**Inspection Time:** {latest.get('timestamp', 'N/A')}"
        )

        dataframe = pd.DataFrame(records)

        columns_to_show = [
            column
            for column in [
                "timestamp",
                "part_id",
                "operator",
                "anomaly_score",
                "ai_signal",
                "decision",
            ]
            if column in dataframe.columns
        ]

        if columns_to_show:
            st.markdown("### Recent Inspection Records")

            st.dataframe(
                dataframe[columns_to_show].tail(10),
                width="stretch",
                hide_index=True,
            )

    else:
        st.info("No saved inspection records yet.")


# ============================================================
# 9. AI VISION
# ============================================================

elif page == "AI Vision":

    st.markdown(
        '<div class="main-title">AI Vision Inspection</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">'
        'Upload an image and perform AI-based visual anomaly analysis'
        '</div>',
        unsafe_allow_html=True,
    )

    method = st.radio(
        "Select Image Input",
        ["Upload Image", "Camera"],
        horizontal=True,
        key="ai_input_method",
    )

    if method == "Upload Image":
        uploaded = st.file_uploader(
            "Upload Part Image",
            type=["jpg", "jpeg", "png"],
            key="part_image_uploader",
        )
    else:
        uploaded = st.camera_input(
            "Capture Part Image",
            key="part_camera_input",
        )

    if uploaded is not None:

        try:
            image = Image.open(uploaded).convert("RGB")

            st.markdown("### Input Inspection Image")

            st.image(
                image,
                caption="Image selected for inspection",
                width="stretch",
            )

            if st.button(
                "Run AI Inspection",
                type="primary",
                width="stretch",
                key="run_ai_inspection",
            ):

                temp_path = None

                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".png",
                    ) as temp:
                        image.save(temp.name, format="PNG")
                        temp_path = temp.name

                    with st.spinner("Running PatchCore inference..."):

                        prediction = predict_anomaly(temp_path)

                        if not isinstance(prediction, dict):
                            raise ValueError(
                                "PatchCore must return a dictionary."
                            )

                        required_fields = [
                            "anomaly_map",
                            "anomaly_score",
                            "predicted_anomaly",
                        ]

                        for required in required_fields:
                            if required not in prediction:
                                raise ValueError(
                                    f"PatchCore result is missing "
                                    f"'{required}'."
                                )

                        score = float(prediction["anomaly_score"])

                        predicted = bool(
                            prediction["predicted_anomaly"]
                        )

                        signal = (
                            "Potential Anomaly"
                            if predicted
                            else "No Strong Anomaly Signal"
                        )

                        heatmap = create_heatmap(
                            prediction["anomaly_map"]
                        )

                        overlay = create_overlay(
                            image,
                            heatmap,
                        )

                        # Normalize image outputs before saving.
                        heatmap_image = to_rgb_pil(heatmap)
                        overlay_image = to_rgb_pil(overlay)

                        final_result = {
                            "anomaly_score": score,
                            "predicted_anomaly": predicted,
                            "ai_signal": signal,
                            "decision": "Manual Review",
                        }

                        # Save files BEFORE reporting success.
                        save_pending_inspection(
                            image,
                            final_result,
                            heatmap_image,
                            overlay_image,
                        )

                        # Update session state.
                        st.session_state["inspection_result"] = (
                            final_result
                        )

                        st.session_state["inspection_image"] = (
                            image.copy()
                        )

                        st.session_state["inspection_evidence"] = {
                            "heatmap": np.asarray(
                                heatmap_image
                            ).copy(),
                            "overlay": np.asarray(
                                overlay_image
                            ).copy(),
                        }

                        st.session_state["inspection_timestamp"] = (
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        )

                        st.session_state["review_saved"] = False

                        # This is safe here: Human/XR widgets are not
                        # created on the AI Vision page.
                        reset_position_statuses()

                    st.success(
                        "AI inspection completed and evidence saved."
                    )

                except Exception as error:
                    st.error(f"Inspection failed: {error}")
                    st.exception(error)

                finally:
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass

        except Exception as error:
            st.error(f"Could not open uploaded image: {error}")
            st.exception(error)

    # Display the latest saved inspection, even after page navigation.
    if st.session_state["inspection_result"] is not None:

        st.divider()
        st.markdown("### AI Inspection Results")

        render_ai_summary()

        result = st.session_state["inspection_result"]

        if result["predicted_anomaly"]:
            st.warning("Potential visual anomaly detected.")
        else:
            st.success("No strong visual anomaly signal detected.")

        render_inspection_evidence(show_original=False)
        render_model_information()

        if st.session_state.get("review_saved", False):
            st.success(
                "Human Verification has been saved. "
                "The AI evidence is still available."
            )
        else:
            st.info(
                "Continue to Human Verification or XR Inspection "
                "to review this result."
            )

    else:
        st.info(
            "Upload an image and run AI Inspection to see the results."
        )


# ============================================================
# 10. HUMAN VERIFICATION
# ============================================================

elif page == "Human Verification":

    st.markdown(
        '<div class="main-title">Human Verification</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">'
        'Review AI findings and record component-level verification'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.session_state["inspection_result"] is None:
        restore_pending_inspection()

    if st.session_state["inspection_result"] is None:

        st.warning(
            "No AI inspection is available. "
            "Run AI Vision Inspection first."
        )

    else:

        render_ai_summary()

        if st.session_state.get("review_saved", False):
            st.success(
                "This inspection has already been saved to history. "
                "Its AI evidence remains available in XR Inspection."
            )

        st.divider()

        render_inspection_evidence(show_original=True)

        st.divider()

        statuses = render_position_selector("human_position")

        render_position_summary(statuses)

        st.divider()
        st.markdown("### Inspection Details")

        c1, c2 = st.columns(2)

        with c1:
            operator = st.text_input(
                "Operator Name",
                placeholder="Enter operator name",
                key="operator_name",
                disabled=st.session_state.get("review_saved", False),
            )

        with c2:
            part_id = st.text_input(
                "Part ID",
                placeholder="Enter part identification",
                key="part_id",
                disabled=st.session_state.get("review_saved", False),
            )

        decision = st.selectbox(
            "Final Inspection Decision",
            ["Manual Review", "Accept", "Reject"],
            key="final_inspection_decision",
            disabled=st.session_state.get("review_saved", False),
        )

        comments = st.text_area(
            "Inspection Comments",
            placeholder="Enter inspection observations...",
            key="inspection_comments",
            disabled=st.session_state.get("review_saved", False),
        )

        all_present = all(
            statuses[position] == "Present"
            for position in POSITIONS
        )

        has_missing = any(
            statuses[position] == "Missing"
            for position in POSITIONS
        )

        all_verified = all(
            statuses[position] != "Not verified"
            for position in POSITIONS
        )

        if decision == "Accept" and not all_present:
            st.warning(
                "Accept requires all four checkpoints to be Present."
            )

        if decision == "Reject" and not has_missing:
            st.warning(
                "Reject requires at least one checkpoint to be Missing."
            )

        if not all_verified:
            st.info(
                "Verify all four checkpoints before saving the review."
            )

        if st.button(
            "Save Inspection",
            type="primary",
            width="stretch",
            key="save_inspection_record",
            disabled=st.session_state.get("review_saved", False),
        ):

            # Read the widget values directly.
            statuses = get_current_statuses("human_position")

            all_present = all(
                statuses[position] == "Present"
                for position in POSITIONS
            )

            has_missing = any(
                statuses[position] == "Missing"
                for position in POSITIONS
            )

            all_verified = all(
                statuses[position] != "Not verified"
                for position in POSITIONS
            )

            if not operator.strip():
                st.error("Enter the operator name.")

            elif not part_id.strip():
                st.error("Enter the Part ID.")

            elif not all_verified:
                st.error(
                    "Please verify P1, P2, P3 and P4 before saving."
                )

            elif decision == "Accept" and not all_present:
                st.error(
                    "Cannot accept: all four checkpoints must be Present."
                )

            elif decision == "Reject" and not has_missing:
                st.error(
                    "Cannot reject without a Missing checkpoint."
                )

            else:

                result = st.session_state["inspection_result"]

                record = {
                    "timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "operator": operator.strip(),
                    "part_id": part_id.strip(),
                    "anomaly_score": float(
                        result["anomaly_score"]
                    ),
                    "ai_signal": result["ai_signal"],
                    "decision": decision,
                    "P1": statuses["P1"],
                    "P2": statuses["P2"],
                    "P3": statuses["P3"],
                    "P4": statuses["P4"],
                    "comments": comments.strip(),
                }

                try:
                    # Save to SQLite first.
                    save_inspection(record)

                    # Mark review saved, but KEEP all AI artifacts.
                    mark_review_as_saved()

                    st.session_state["review_saved"] = True

                    st.success(
                        "Human Verification saved successfully. "
                        "AI inspection evidence has been retained."
                    )

                    st.markdown("### Saved Inspection Record")
                    st.json(record)

                    st.info(
                        "You can now open XR Inspection. "
                        "The original image, heatmap, overlay, "
                        "and 3D bracket view will remain available."
                    )

                except Exception as error:
                    st.error(
                        f"Could not save inspection: {error}"
                    )
                    st.exception(error)


# ============================================================
# 11. XR INSPECTION
# ============================================================

elif page == "XR Inspection":

    st.markdown(
        '<div class="main-title">XR Inspection View</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">'
        'Inspection evidence and interactive component verification layout'
        '</div>',
        unsafe_allow_html=True,
    )

    # Recover the saved AI result if session state is empty.
    if st.session_state["inspection_result"] is None:
        restore_pending_inspection()

    if st.session_state["inspection_result"] is None:

        st.warning(
            "No AI inspection is available. "
            "Run AI Vision Inspection first."
        )

        if st.button(
            "Refresh Inspection Results",
            key="refresh_xr_results",
        ):
            if restore_pending_inspection():
                st.success("Inspection results restored.")
            else:
                st.warning(
                    "No saved inspection evidence was found."
                )

            st.rerun()

    else:

        render_ai_summary()

        if st.session_state.get("review_saved", False):
            st.success(
                "Human Verification saved. "
                "Displaying the retained AI inspection."
            )
        else:
            st.info(
                "Human Verification is pending for this inspection."
            )

        st.divider()

        st.markdown("### Inspection Evidence")

        render_inspection_evidence(show_original=True)

        st.divider()

        st.markdown("### 3D Mounting Bracket View")

        xr_statuses = render_position_selector("xr_position")

        render_position_summary(xr_statuses)

        render_bracket_3d(xr_statuses)

        st.button(
            "Reset XR Component Statuses",
            width="stretch",
            key="reset_xr_positions",
            on_click=reset_position_statuses,
        )

        render_model_information()


# ============================================================
# 12. INSPECTION HISTORY
# ============================================================

elif page == "Inspection History":

    st.markdown(
        '<div class="main-title">Inspection History</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">'
        'Review saved inspection records and export results'
        '</div>',
        unsafe_allow_html=True,
    )

    records = get_all_inspections() or []

    if not records:

        st.info("No inspection records have been saved yet.")

    else:

        df = pd.DataFrame(records)

        st.markdown("### Inspection Summary")

        total = len(df)

        accepted = (
            (df["decision"] == "Accept").sum()
            if "decision" in df.columns
            else 0
        )

        rejected = (
            (df["decision"] == "Reject").sum()
            if "decision" in df.columns
            else 0
        )

        manual = (
            (df["decision"] == "Manual Review").sum()
            if "decision" in df.columns
            else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total", total)
        c2.metric("Accepted", int(accepted))
        c3.metric("Rejected", int(rejected))
        c4.metric("Manual Review", int(manual))

        st.divider()

        st.markdown("### All Inspection Records")

        preferred = [
            "timestamp",
            "part_id",
            "operator",
            "anomaly_score",
            "ai_signal",
            "decision",
            "P1",
            "P2",
            "P3",
            "P4",
            "comments",
        ]

        visible = [
            column
            for column in preferred
            if column in df.columns
        ]

        if visible:
            st.dataframe(
                df[visible],
                width="stretch",
                hide_index=True,
            )

        st.divider()

        st.markdown("### View Individual Inspection")

        selected = st.selectbox(
            "Select Inspection Record",
            options=list(range(len(df))),
            format_func=lambda index: (
                f"{df.iloc[index].get('part_id', 'Unknown')} | "
                f"{df.iloc[index].get('timestamp', 'Unknown')}"
            ),
            key="history_record_selector",
        )

        st.json(df.iloc[selected].to_dict())

        st.divider()

        st.markdown("### Decision Distribution")

        if "decision" in df.columns:

            counts = (
                df["decision"]
                .fillna("Unknown")
                .value_counts()
                .reset_index()
            )

            counts.columns = ["Decision", "Count"]

            fig = px.bar(
                counts,
                x="Decision",
                y="Count",
                text="Count",
                title="Inspection Decisions",
            )

            fig.update_layout(
                height=350,
                xaxis_title="Decision",
                yaxis_title="Number of Inspections",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

        st.divider()

        st.markdown("### Export Inspection Data")

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Inspection History (CSV)",
            data=csv_data,
            file_name="inspection_history.csv",
            mime="text/csv",
            width="stretch",
        )

        st.caption(
            "Inspection records are retrieved from the local SQLite database."
        )


# ============================================================
# 13. FOOTER
# ============================================================

st.sidebar.divider()
st.sidebar.caption("Automotive Part Inspection System")
st.sidebar.caption("AI-assisted inspection with human verification")