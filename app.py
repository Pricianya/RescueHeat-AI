import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import pandas as pd
import av
import threading

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration
)

from streamlit_geolocation import streamlit_geolocation


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RescueHeat AI",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# Thermal model can exist in either location:
# Local:  RescueHeatAI/models/best.pt
# Cloud:  RescueHeatAI/best.pt

LOCAL_THERMAL_MODEL = BASE_DIR / "models" / "best.pt"
CLOUD_THERMAL_MODEL = BASE_DIR / "best.pt"

RGB_MODEL_PATH = BASE_DIR / "yolo11n.pt"


def find_thermal_model():

    if LOCAL_THERMAL_MODEL.exists():
        return LOCAL_THERMAL_MODEL

    if CLOUD_THERMAL_MODEL.exists():
        return CLOUD_THERMAL_MODEL

    return None


THERMAL_MODEL_PATH = find_thermal_model()


# =========================================================
# CHECK MODEL FILES
# =========================================================

if THERMAL_MODEL_PATH is None:

    st.error(
        "Thermal model best.pt was not found.\n\n"
        "Expected either:\n"
        "- models/best.pt\n"
        "- best.pt"
    )

    st.stop()


if not RGB_MODEL_PATH.exists():

    st.error(
        "RGB model yolo11n.pt was not found "
        "in the same folder as app.py."
    )

    st.stop()


# =========================================================
# LOAD YOLO MODELS
# =========================================================

@st.cache_resource
def load_thermal_model():
    return YOLO(
        str(THERMAL_MODEL_PATH)
    )


@st.cache_resource
def load_rgb_model():
    return YOLO(
        str(RGB_MODEL_PATH)
    )


thermal_model = load_thermal_model()
rgb_model = load_rgb_model()


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {

    "detections": [],

    "detection_count": 0,

    "avg_confidence": 0.0,

    "priority": "WAITING",

    "recommendation":
        "No person detection available.",

    "latitude": 12.9716,

    "longitude": 77.5946,

    "gps_accuracy": 0.0,

    "gps_received": False
}


for key, value in DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# LIVE DETECTION STORE
# =========================================================

@st.cache_resource
def create_live_store():

    return {

        "count": 0,

        "confidence": 0.0,

        "max_confidence": 0.0,

        "lock": threading.Lock()
    }


live_store = create_live_store()


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #0f172a;
}

/* ------------------------------------------------------ */
/* Titles */
/* ------------------------------------------------------ */

.main-title {
    font-size: 44px;
    font-weight: 800;
    color: white;
    margin-bottom: 0px;
}

.subtitle {
    color: #38bdf8;
    font-size: 17px;
    margin-bottom: 20px;
}


/* ------------------------------------------------------ */
/* Cards */
/* ------------------------------------------------------ */

.card {
    background-color: #172554;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #334155;
    color: white;
}

.info-card {
    background-color: #111827;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #374151;
    color: white;
}

.alert-card {
    background-color: #3f1d0d;
    border-left: 5px solid #f97316;
    padding: 18px;
    border-radius: 10px;
    color: white;
}


/* ------------------------------------------------------ */
/* Rescue priority cards */
/* ------------------------------------------------------ */

.safe-card {
    background-color: #052e16;
    border: 1px solid #22c55e;
    padding: 18px;
    border-radius: 12px;
    color: white;
}

.medium-card {
    background-color: #431407;
    border: 1px solid #f97316;
    padding: 18px;
    border-radius: 12px;
    color: white;
}

.high-card {
    background-color: #450a0a;
    border: 1px solid #ef4444;
    padding: 18px;
    border-radius: 12px;
    color: white;
}


/* ------------------------------------------------------ */
/* GPS */
/* ------------------------------------------------------ */

.gps-card {
    background-color: #082f49;
    border: 1px solid #0ea5e9;
    padding: 20px;
    border-radius: 14px;
    color: white;
}


/* ------------------------------------------------------ */
/* Sidebar */
/* ------------------------------------------------------ */

section[data-testid="stSidebar"] {
    background-color: #020617;
}


h1, h2, h3 {
    color: white;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# RESCUE STATUS
# =========================================================

def rescue_status(
    confidence,
    count
):

    if count == 0:

        return (
            "NO PERSON DETECTED",
            "No rescue alert is currently triggered.",
            "safe-card"
        )


    if confidence >= 85:

        return (
            "HIGH PRIORITY RESCUE 🔴",
            "Strong person detection. "
            "Immediate rescue verification is recommended.",
            "high-card"
        )


    if confidence >= 60:

        return (
            "RESCUE CHECK NEEDED 🟠",
            "Possible person detected. "
            "Rescue personnel should verify the location.",
            "medium-card"
        )


    return (
        "LOW CONFIDENCE DETECTION 🟢",
        "Possible person detection. "
        "Verify the detection before dispatch.",
        "safe-card"
    )


# =========================================================
# CREATE PERSON DETECTION RECORDS
# =========================================================

def create_detection_records(
    result
):

    records = []


    for i, box in enumerate(
        result.boxes
    ):

        confidence = (
            float(
                box.conf[0]
            )
            * 100
        )


        record = {

            "ID":
                f"SUR-{i + 1:03d}",

            "Class":
                "person",

            "Confidence":
                round(
                    confidence,
                    2
                )
        }


        # -------------------------------------------------
        # Attach GPS if available
        # -------------------------------------------------

        if st.session_state.gps_received:

            record["Latitude"] = round(
                st.session_state.latitude,
                6
            )

            record["Longitude"] = round(
                st.session_state.longitude,
                6
            )

            record["GPS Accuracy (m)"] = round(
                st.session_state.gps_accuracy,
                1
            )


        else:

            record["Latitude"] = (
                "GPS unavailable"
            )

            record["Longitude"] = (
                "GPS unavailable"
            )

            record[
                "GPS Accuracy (m)"
            ] = "-"


        records.append(
            record
        )


    return records


# =========================================================
# SAVE DETECTION RESULTS
# =========================================================

def save_detection_results(
    records
):

    st.session_state.detections = (
        records
    )

    st.session_state.detection_count = (
        len(
            records
        )
    )


    if records:

        average = sum(
            item["Confidence"]
            for item in records
        ) / len(records)


        st.session_state.avg_confidence = (
            average
        )


        (
            status,
            recommendation,
            _
        ) = rescue_status(
            average,
            len(records)
        )


        st.session_state.priority = (
            status
        )

        st.session_state.recommendation = (
            recommendation
        )


    else:

        st.session_state.avg_confidence = (
            0.0
        )

        st.session_state.priority = (
            "NO PERSON DETECTED"
        )

        st.session_state.recommendation = (
            "No rescue alert is currently triggered."
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title(
        "🛸 RESCUEHEAT AI"
    )


    st.caption(
        "Thermal-Guided Drone Intelligence"
    )


    st.divider()


    page = st.radio(
        "Navigation",
        [
            "🏠 Mission Dashboard",
            "🌡️ Thermal Detection",
            "📹 Live Detection",
            "📍 GPS Locations",
            "🔄 Mission Workflow",
            "⚙️ Technology Stack"
        ]
    )


    st.divider()


    st.subheader(
        "Mission Status"
    )


    st.success(
        "🟢 SYSTEM ONLINE"
    )


    st.metric(
        "🔋 Battery",
        "68%"
    )


    st.metric(
        "📡 Signal",
        "92%"
    )


    st.metric(
        "🛸 Altitude",
        "42 m"
    )


    if st.session_state.gps_received:

        st.success(
            "📍 GPS ACTIVE"
        )

    else:

        st.warning(
            "📍 GPS WAITING"
        )


# =========================================================
# PAGE 1
# MISSION DASHBOARD
# =========================================================

if page == "🏠 Mission Dashboard":

    st.markdown(
        """
        <div class="main-title">
        Live Mission Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="subtitle">
        AI-assisted thermal search and rescue monitoring
        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # TOP METRICS
    # -----------------------------------------------------

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


    metric1.metric(
        "Detected Persons",
        st.session_state.detection_count
    )


    metric2.metric(
        "AI Confidence",
        f"{st.session_state.avg_confidence:.1f}%"
    )


    metric3.metric(
        "Mission Status",
        st.session_state.priority
    )


    metric4.metric(
        "GPS",
        (
            "ACTIVE"
            if st.session_state.gps_received
            else "WAITING"
        )
    )


    st.divider()


    # -----------------------------------------------------
    # MAIN AREA
    # -----------------------------------------------------

    left, right = st.columns(
        [1.5, 1]
    )


    # =====================================================
    # LEFT SIDE
    # =====================================================

    with left:

        st.subheader(
            "🌡️ Thermal Feed"
        )


        dashboard_file = st.file_uploader(
            "Upload current thermal frame",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            key="dashboard_upload"
        )


        if dashboard_file is not None:

            dashboard_image = Image.open(
                dashboard_file
            )


            st.image(
                dashboard_image,
                caption="Thermal Camera Feed",
                use_container_width=True
            )


            if st.button(
                "🔍 Run Thermal AI Detection",
                type="primary",
                key="dashboard_detect"
            ):

                with st.spinner(
                    "Analyzing thermal image..."
                ):

                    results = (
                        thermal_model.predict(
                            source=dashboard_image,
                            conf=0.25,
                            classes=[0],
                            verbose=False
                        )
                    )


                result = results[0]


                st.image(
                    result.plot(),
                    caption="Thermal YOLO Detection",
                    use_container_width=True
                )


                records = (
                    create_detection_records(
                        result
                    )
                )


                save_detection_results(
                    records
                )


                if records:

                    st.success(
                        f"👤 {len(records)} "
                        "person(s) detected."
                    )


                    if st.session_state.gps_received:

                        st.success(
                            "📍 GPS coordinates attached."
                        )

                    else:

                        st.warning(
                            "Person detected, but GPS "
                            "has not been captured yet."
                        )


                else:

                    st.warning(
                        "No person detected."
                    )


    # =====================================================
    # RIGHT SIDE
    # =====================================================

    with right:

        st.subheader(
            "🚨 Mission Information"
        )


        (
            status,
            recommendation,
            css_class
        ) = rescue_status(
            st.session_state.avg_confidence,
            st.session_state.detection_count
        )


        st.markdown(
            f"""
            <div class="{css_class}">

            <h3>
            Detected Persons
            </h3>

            <h1>
            {st.session_state.detection_count}
            </h1>

            <hr>

            <h3>
            Average Confidence
            </h3>

            <h1>
            {st.session_state.avg_confidence:.1f}%
            </h1>

            <hr>

            <h3>
            Rescue Decision
            </h3>

            <h2>
            {status}
            </h2>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.write("")


        st.info(
            recommendation
        )


        if st.session_state.detection_count > 0:

            st.markdown(
                """
                <div class="alert-card">

                🚨 <b>
                RESCUE ALERT
                </b>

                <br><br>

                Possible human heat signature detected.

                <br><br>

                Verify the detection and location
                before rescue dispatch.

                </div>
                """,
                unsafe_allow_html=True
            )


            if st.session_state.gps_received:

                st.write("")


                st.markdown(
                    f"""
                    <div class="gps-card">

                    <h3>
                    📍 Detection Location
                    </h3>

                    <b>Latitude:</b>
                    {st.session_state.latitude:.6f}

                    <br><br>

                    <b>Longitude:</b>
                    {st.session_state.longitude:.6f}

                    <br><br>

                    <b>Accuracy:</b>
                    {st.session_state.gps_accuracy:.0f} m

                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # -----------------------------------------------------
    # DETECTION TABLE
    # -----------------------------------------------------

    st.divider()


    st.subheader(
        "📊 Detection Details"
    )


    if st.session_state.detections:

        detection_df = pd.DataFrame(
            st.session_state.detections
        )


        st.dataframe(
            detection_df,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.info(
            "Run thermal detection to display results."
        )


    # -----------------------------------------------------
    # MAP
    # -----------------------------------------------------

    st.divider()


    st.subheader(
        "📍 Current Mission Location"
    )


    if st.session_state.gps_received:

        map_data = pd.DataFrame(
            {
                "lat": [
                    st.session_state.latitude
                ],

                "lon": [
                    st.session_state.longitude
                ]
            }
        )


        st.map(
            map_data,
            zoom=16
        )


    else:

        st.info(
            "Open GPS Locations and capture "
            "your current position first."
        )


# =========================================================
# PAGE 2
# THERMAL DETECTION
# =========================================================

elif page == "🌡️ Thermal Detection":

    st.markdown(
        """
        <div class="main-title">
        Thermal Human Detection
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="subtitle">
        Custom YOLO model trained on AIResQ
        </div>
        """,
        unsafe_allow_html=True
    )


    if st.session_state.gps_received:

        st.success(
            "📍 GPS ACTIVE — coordinates will "
            "be attached to detections."
        )


    else:

        st.warning(
            "📍 GPS is not active. "
            "Capture GPS first if coordinates are required."
        )


    uploaded_file = st.file_uploader(
        "Upload Thermal Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="thermal_detection"
    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )


        image_col, result_col = (
            st.columns(2)
        )


        # -------------------------------------------------
        # INPUT
        # -------------------------------------------------

        with image_col:

            st.subheader(
                "Input Thermal Image"
            )


            st.image(
                image,
                use_container_width=True
            )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        with result_col:

            st.subheader(
                "AI Detection Result"
            )


            if st.button(
                "🔍 Detect Person",
                type="primary",
                key="thermal_detect_button"
            ):

                with st.spinner(
                    "Running thermal YOLO model..."
                ):

                    results = thermal_model.predict(
                        source=image,
                        conf=0.25,
                        classes=[0],
                        verbose=False
                    )


                result = results[0]


                st.image(
                    result.plot(),
                    use_container_width=True
                )


                records = (
                    create_detection_records(
                        result
                    )
                )


                save_detection_results(
                    records
                )


                if records:

                    avg = (
                        st.session_state.avg_confidence
                    )


                    (
                        status,
                        recommendation,
                        css_class
                    ) = rescue_status(
                        avg,
                        len(records)
                    )


                    st.success(
                        f"👤 {len(records)} "
                        "person(s) detected."
                    )


                    st.markdown(
                        f"""
                        <div class="{css_class}">

                        <h3>
                        Rescue Decision
                        </h3>

                        <h2>
                        {status}
                        </h2>

                        <p>
                        Detection confidence:
                        <b>
                        {avg:.1f}%
                        </b>
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    st.info(
                        recommendation
                    )


                else:

                    st.warning(
                        "No person detected."
                    )


        # -------------------------------------------------
        # TABLE
        # -------------------------------------------------

        if st.session_state.detections:

            st.divider()


            st.subheader(
                "📊 Detection Records"
            )


            st.dataframe(
                pd.DataFrame(
                    st.session_state.detections
                ),
                use_container_width=True,
                hide_index=True
            )


    else:

        st.info(
            "Upload a thermal image to begin detection."
        )


# =========================================================
# PAGE 3
# LIVE DETECTION
# =========================================================

elif page == "📹 Live Detection":

    st.markdown(
        """
        <div class="main-title">
        Continuous Live Detection
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="subtitle">
        Real-time person detection from a live camera
        </div>
        """,
        unsafe_allow_html=True
    )


    st.warning(
        "Laptop webcam = RGB mode. "
        "Actual thermal camera = Thermal mode."
    )


    camera_mode = st.radio(
        "Select Camera Mode",
        [
            "💻 Laptop / RGB Camera",
            "🌡️ Thermal Camera"
        ],
        horizontal=True
    )


    # -----------------------------------------------------
    # SELECT MODEL
    # -----------------------------------------------------

    if camera_mode == "💻 Laptop / RGB Camera":

        live_model = rgb_model

        confidence_threshold = (
            0.45
        )


        st.success(
            "RGB detection mode active."
        )


    else:

        live_model = thermal_model

        confidence_threshold = (
            0.25
        )


        st.success(
            "Thermal detection mode active."
        )


    st.divider()


    st.subheader(
        "📹 Live Camera"
    )


    # =====================================================
    # WEBRTC PROCESSOR
    # =====================================================

    class RescueHeatProcessor:

        def recv(
            self,
            frame
        ):

            image = frame.to_ndarray(
                format="bgr24"
            )


            results = live_model.predict(
                source=image,
                conf=confidence_threshold,
                classes=[0],
                verbose=False
            )


            result = results[0]


            confidences = [

                float(
                    box.conf[0]
                ) * 100

                for box
                in result.boxes
            ]


            with live_store["lock"]:

                live_store["count"] = (
                    len(
                        result.boxes
                    )
                )


                if confidences:

                    live_store[
                        "confidence"
                    ] = (
                        sum(confidences)
                        / len(confidences)
                    )


                    live_store[
                        "max_confidence"
                    ] = max(
                        confidences
                    )


                else:

                    live_store[
                        "confidence"
                    ] = 0.0


                    live_store[
                        "max_confidence"
                    ] = 0.0


            annotated = (
                result.plot()
            )


            return (
                av.VideoFrame.from_ndarray(
                    annotated,
                    format="bgr24"
                )
            )


    # =====================================================
    # START WEBRTC
    # =====================================================

    webrtc_streamer(

        key=f"rescueheat-{camera_mode}",

        mode=WebRtcMode.SENDRECV,

        rtc_configuration=RTCConfiguration(
            {
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            }
        ),

        video_processor_factory=(
            RescueHeatProcessor
        ),

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )


    st.divider()


    st.subheader(
        "🚨 Live Rescue Decision"
    )


    # =====================================================
    # AUTO-REFRESH STATUS
    # =====================================================

    @st.fragment(
        run_every="0.5s"
    )
    def live_status_panel():

        with live_store["lock"]:

            live_count = (
                live_store["count"]
            )


            live_conf = (
                live_store["confidence"]
            )


            max_conf = (
                live_store[
                    "max_confidence"
                ]
            )


        (
            status,
            recommendation,
            css_class
        ) = rescue_status(
            live_conf,
            live_count
        )


        c1, c2, c3, c4 = (
            st.columns(4)
        )


        c1.metric(
            "Persons Detected",
            live_count
        )


        c2.metric(
            "Average Confidence",
            f"{live_conf:.1f}%"
        )


        c3.metric(
            "Highest Confidence",
            f"{max_conf:.1f}%"
        )


        c4.metric(
            "Rescue Status",
            status
        )


        if live_count > 0:

            st.markdown(
                f"""
                <div class="{css_class}">

                <h2>
                {status}
                </h2>

                <p>
                👤 <b>
                {live_count}
                </b>
                person(s) currently detected.
                </p>

                <p>
                Average confidence:
                <b>
                {live_conf:.1f}%
                </b>
                </p>

                <p>
                Highest confidence:
                <b>
                {max_conf:.1f}%
                </b>
                </p>

                <p>
                {recommendation}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            if st.session_state.gps_received:

                st.write("")


                st.markdown(
                    f"""
                    <div class="gps-card">

                    <h3>
                    📍 Current Rescue Coordinates
                    </h3>

                    <b>
                    Latitude:
                    </b>

                    {st.session_state.latitude:.6f}

                    <br><br>

                    <b>
                    Longitude:
                    </b>

                    {st.session_state.longitude:.6f}

                    <br><br>

                    <b>
                    GPS Accuracy:
                    </b>

                    {st.session_state.gps_accuracy:.0f} m

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            else:

                st.warning(
                    "Person detected but GPS "
                    "location is not available."
                )


        else:

            st.markdown(
                """
                <div class="safe-card">

                <h2>
                🟢 NO PERSON DETECTED
                </h2>

                <p>
                No person is currently detected
                in the camera frame.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        st.caption(
            "Rescue priority is based on "
            "AI detection confidence only. "
            "It is not a medical assessment."
        )


    live_status_panel()


# =========================================================
# PAGE 4
# GPS LOCATIONS
# =========================================================

elif page == "📍 GPS Locations":

    st.markdown(
        """
        <div class="main-title">
        Live Rescue GPS Location
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="subtitle">
        Capture coordinates for rescue navigation
        </div>
        """,
        unsafe_allow_html=True
    )


    st.info(
        "Click the location button below and "
        "allow location permission in your browser."
    )


    # -----------------------------------------------------
    # GET GPS
    # -----------------------------------------------------

    location = (
        streamlit_geolocation()
    )


    # -----------------------------------------------------
    # VALID LOCATION
    # -----------------------------------------------------

    if (
        location
        and location.get(
            "latitude"
        ) is not None
        and location.get(
            "longitude"
        ) is not None
    ):

        latitude = float(
            location["latitude"]
        )


        longitude = float(
            location["longitude"]
        )


        accuracy = location.get(
            "accuracy",
            0
        )


        if accuracy is None:

            accuracy = 0


        accuracy = float(
            accuracy
        )


        # -------------------------------------------------
        # SAVE GPS
        # -------------------------------------------------

        st.session_state.latitude = (
            latitude
        )


        st.session_state.longitude = (
            longitude
        )


        st.session_state.gps_accuracy = (
            accuracy
        )


        st.session_state.gps_received = (
            True
        )


        st.success(
            "📡 GPS location received!"
        )


        # -------------------------------------------------
        # GPS METRICS
        # -------------------------------------------------

        g1, g2, g3, g4 = (
            st.columns(4)
        )


        g1.metric(
            "GPS Status",
            "ACTIVE"
        )


        g2.metric(
            "Latitude",
            f"{latitude:.6f}"
        )


        g3.metric(
            "Longitude",
            f"{longitude:.6f}"
        )


        g4.metric(
            "Accuracy",
            (
                f"{accuracy:.0f} m"
                if accuracy > 0
                else "Unknown"
            )
        )


        st.divider()


        # -------------------------------------------------
        # GPS CARD
        # -------------------------------------------------

        st.markdown(
            f"""
            <div class="gps-card">

            <h2>
            📡 GPS ACTIVE
            </h2>

            <hr>

            <h3>
            Latitude
            </h3>

            <h2>
            {latitude:.6f}
            </h2>

            <h3>
            Longitude
            </h3>

            <h2>
            {longitude:.6f}
            </h2>

            <h3>
            Estimated Accuracy
            </h3>

            <h2>
            {accuracy:.0f} meters
            </h2>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.divider()


        # -------------------------------------------------
        # MAP
        # -------------------------------------------------

        st.subheader(
            "🗺️ Rescue Location Map"
        )


        gps_df = pd.DataFrame(
            {
                "lat": [
                    latitude
                ],

                "lon": [
                    longitude
                ]
            }
        )


        st.map(
            gps_df,
            zoom=16
        )


        st.success(
            "📍 GPS ready for detection records."
        )


        st.warning(
            "Prototype: this is the browser/device "
            "location. In the final drone system, "
            "coordinates should come from the drone GPS."
        )


    else:

        st.warning(
            "📡 Waiting for GPS location."
        )


        if st.session_state.gps_received:

            last_location = pd.DataFrame(
                {
                    "lat": [
                        st.session_state.latitude
                    ],

                    "lon": [
                        st.session_state.longitude
                    ]
                }
            )


            st.subheader(
                "Last Known Location"
            )


            st.map(
                last_location,
                zoom=16
            )


# =========================================================
# PAGE 5
# WORKFLOW
# =========================================================

elif page == "🔄 Mission Workflow":

    st.markdown(
        """
        <div class="main-title">
        End-to-End Mission Workflow
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="subtitle">
        From drone search to rescue response
        </div>
        """,
        unsafe_allow_html=True
    )


    workflow = [

        (
            "1",
            "🛸 Drone",
            "Search disaster area"
        ),

        (
            "2",
            "🌡️ Thermal",
            "Capture heat signatures"
        ),

        (
            "3",
            "📹 Live Frames",
            "Continuous video input"
        ),

        (
            "4",
            "🤖 YOLO",
            "Detect possible people"
        ),

        (
            "5",
            "📊 Confidence",
            "Measure detection certainty"
        ),

        (
            "6",
            "🚦 Decision",
            "Assign rescue priority"
        ),

        (
            "7",
            "📍 GPS",
            "Attach coordinates"
        ),

        (
            "8",
            "🚨 Alert",
            "Notify rescue team"
        ),

        (
            "9",
            "🚑 Rescue",
            "Human verification"
        )
    ]


    for i in range(
        0,
        len(workflow),
        3
    ):

        cols = st.columns(3)


        for col, step in zip(
            cols,
            workflow[
                i:i + 3
            ]
        ):

            with col:

                st.markdown(
                    f"""
                    <div
                    class="card"
                    style="
                    min-height:180px;
                    text-align:center;
                    "
                    >

                    <h1>
                    {step[0]}
                    </h1>

                    <h3>
                    {step[1]}
                    </h3>

                    <p>
                    {step[2]}
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        st.write("")


# =========================================================
# PAGE 6
# TECHNOLOGY STACK
# =========================================================

elif page == "⚙️ Technology Stack":

    st.markdown(
        """
        <div class="main-title">
        Technology Stack
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="subtitle">
        Technologies powering RescueHeat AI
        </div>
        """,
        unsafe_allow_html=True
    )


    technologies = [

        (
            "🐍",
            "Python",
            "Core programming"
        ),

        (
            "🤖",
            "YOLO",
            "Person detection"
        ),

        (
            "🔥",
            "PyTorch",
            "Deep learning"
        ),

        (
            "🌡️",
            "AIResQ",
            "Thermal dataset"
        ),

        (
            "📹",
            "WebRTC",
            "Live streaming"
        ),

        (
            "📊",
            "Streamlit",
            "Mission dashboard"
        ),

        (
            "📍",
            "GPS",
            "Location tracking"
        ),

        (
            "🛸",
            "Drone",
            "Aerial search platform"
        )
    ]


    for i in range(
        0,
        len(technologies),
        4
    ):

        cols = st.columns(4)


        for col, tech in zip(
            cols,
            technologies[
                i:i + 4
            ]
        ):

            with col:

                st.markdown(
                    f"""
                    <div
                    class="card"
                    style="
                    text-align:center;
                    min-height:160px;
                    "
                    >

                    <h1>
                    {tech[0]}
                    </h1>

                    <h3>
                    {tech[1]}
                    </h3>

                    <p>
                    {tech[2]}
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        st.write("")


    st.divider()


    st.subheader(
        "🏗️ System Architecture"
    )


    st.markdown(
        """
        <div class="info-card">

        <h3 style="text-align:center;">

        🛸 Drone

        &nbsp; → &nbsp;

        🌡️ Thermal Camera

        &nbsp; → &nbsp;

        📹 Live Stream

        &nbsp; → &nbsp;

        🤖 YOLO

        &nbsp; → &nbsp;

        📍 GPS

        &nbsp; → &nbsp;

        🚨 Rescue Alert

        &nbsp; → &nbsp;

        🚑 Rescue Team

        </h3>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()


st.caption(
    "RescueHeat AI • Hackathon Prototype • "
    "Thermal-Guided Drone Intelligence"
)