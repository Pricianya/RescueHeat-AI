import streamlit as st
from ultralytics import YOLO
from PIL import Image
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
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RescueHeat AI",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_thermal_model():
    return YOLO(r"C:\RescueHeatAI\models\best.pt")


@st.cache_resource
def load_rgb_model():
    return YOLO("yolo11n.pt")


thermal_model = load_thermal_model()
rgb_model = load_rgb_model()


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "detections": [],
    "detection_count": 0,
    "avg_confidence": 0.0,
    "priority": "WAITING",
    "recommendation": "No survivor detection available.",
    "latitude": 12.9716,
    "longitude": 77.5946,
    "gps_accuracy": 0.0,
    "gps_received": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# LIVE DETECTION STORE
# =========================================================

@st.cache_resource
def get_live_store():
    return {
        "count": 0,
        "confidence": 0.0,
        "max_confidence": 0.0,
        "lock": threading.Lock()
    }


live_store = get_live_store()


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
}

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

.gps-card {
    background-color: #082f49;
    border: 1px solid #0ea5e9;
    padding: 20px;
    border-radius: 14px;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #020617;
}

h1, h2, h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# RESCUE STATUS
# =========================================================

def rescue_status(confidence, count):

    if count == 0:
        return (
            "NO PERSON DETECTED",
            "No rescue action triggered.",
            "safe-card"
        )

    if confidence >= 85:
        return (
            "HIGH PRIORITY RESCUE 🔴",
            "Strong person detection. Immediate rescue verification is recommended.",
            "high-card"
        )

    elif confidence >= 60:
        return (
            "RESCUE CHECK NEEDED 🟠",
            "Possible person detected. Rescue team should verify the location.",
            "medium-card"
        )

    else:
        return (
            "LOW CONFIDENCE DETECTION 🟢",
            "Possible person detection. Verify before rescue dispatch.",
            "safe-card"
        )


# =========================================================
# CREATE DETECTION RECORDS
# =========================================================

def create_detection_records(result):

    detection_data = []

    for i, box in enumerate(result.boxes):

        confidence = float(box.conf[0]) * 100

        record = {
            "ID": f"SUR-{i + 1:03d}",
            "Class": "person",
            "Confidence": round(confidence, 2)
        }

        # Add GPS only when live GPS has been received
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

            record["Latitude"] = "GPS unavailable"
            record["Longitude"] = "GPS unavailable"
            record["GPS Accuracy (m)"] = "-"

        detection_data.append(record)

    return detection_data


# =========================================================
# SAVE DETECTION RESULTS
# =========================================================

def save_detection_results(detection_data):

    st.session_state.detections = detection_data
    st.session_state.detection_count = len(detection_data)

    if detection_data:

        average = sum(
            item["Confidence"]
            for item in detection_data
        ) / len(detection_data)

        st.session_state.avg_confidence = average

        status, recommendation, _ = rescue_status(
            average,
            len(detection_data)
        )

        st.session_state.priority = status
        st.session_state.recommendation = recommendation

    else:

        st.session_state.avg_confidence = 0.0
        st.session_state.priority = "NO PERSON DETECTED"
        st.session_state.recommendation = (
            "No rescue action triggered."
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🛸 RESCUEHEAT AI")

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

    st.subheader("Mission Status")

    st.success("🟢 SYSTEM ONLINE")

    st.metric("🔋 Battery", "68%")
    st.metric("📡 Signal", "92%")
    st.metric("🛸 Altitude", "42 m")

    if st.session_state.gps_received:
        st.success("📍 GPS ACTIVE")
    else:
        st.warning("📍 GPS WAITING")


# =========================================================
# PAGE 1 — MISSION DASHBOARD
# =========================================================

if page == "🏠 Mission Dashboard":

    st.markdown(
        '<div class="main-title">Live Mission Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-assisted thermal search and rescue monitoring'
        '</div>',
        unsafe_allow_html=True
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    count_placeholder = metric1.empty()
    confidence_placeholder = metric2.empty()
    priority_placeholder = metric3.empty()
    gps_placeholder = metric4.empty()


    def update_dashboard_metrics():

        status, _, _ = rescue_status(
            st.session_state.avg_confidence,
            st.session_state.detection_count
        )

        count_placeholder.metric(
            "Detected Persons",
            st.session_state.detection_count
        )

        confidence_placeholder.metric(
            "AI Confidence",
            f"{st.session_state.avg_confidence:.1f}%"
        )

        priority_placeholder.metric(
            "Mission Status",
            status
        )

        if st.session_state.gps_received:
            gps_placeholder.metric(
                "GPS",
                "ACTIVE"
            )
        else:
            gps_placeholder.metric(
                "GPS",
                "WAITING"
            )


    update_dashboard_metrics()

    st.divider()

    left, right = st.columns([1.5, 1])

    # -----------------------------------------------------
    # THERMAL FEED
    # -----------------------------------------------------

    with left:

        st.subheader("🌡️ Thermal Feed")

        dashboard_file = st.file_uploader(
            "Upload current thermal frame",
            type=["jpg", "jpeg", "png"],
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

                    results = thermal_model.predict(
                        source=dashboard_image,
                        conf=0.25,
                        classes=[0],
                        verbose=False
                    )

                result = results[0]

                annotated = result.plot()

                st.image(
                    annotated,
                    caption="Thermal YOLO Detection",
                    use_container_width=True
                )

                # =========================================
                # PERSON + GPS RECORDS
                # =========================================

                detection_data = create_detection_records(
                    result
                )

                save_detection_results(
                    detection_data
                )

                if detection_data:

                    st.success(
                        f"👤 {len(detection_data)} "
                        "person(s) detected."
                    )

                    if st.session_state.gps_received:

                        st.success(
                            "📍 GPS coordinates attached "
                            "to the detection."
                        )

                    else:

                        st.warning(
                            "Person detected, but live GPS "
                            "has not been captured yet."
                        )

                else:

                    st.warning(
                        "No person detected."
                    )

                update_dashboard_metrics()

    # -----------------------------------------------------
    # MISSION INFORMATION
    # -----------------------------------------------------

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

            <h3>Detected Persons</h3>

            <h1>
            {st.session_state.detection_count}
            </h1>

            <hr>

            <h3>Average Confidence</h3>

            <h1>
            {st.session_state.avg_confidence:.1f}%
            </h1>

            <hr>

            <h3>Rescue Decision</h3>

            <h2>
            {status}
            </h2>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.info(recommendation)

        if st.session_state.detection_count > 0:

            st.markdown(
                """
                <div class="alert-card">

                🚨 <b>RESCUE ALERT</b>

                <br><br>

                Possible human heat signature detected.

                <br><br>

                Verify detection and location before
                dispatching the rescue team.

                </div>
                """,
                unsafe_allow_html=True
            )

            if st.session_state.gps_received:

                st.write("")

                st.markdown(
                    f"""
                    <div class="gps-card">

                    <h3>📍 Detection Location</h3>

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

        df = pd.DataFrame(
            st.session_state.detections
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Run thermal detection to display results."
        )

    # -----------------------------------------------------
    # LOCATION MAP
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
            "Open GPS Locations and capture your "
            "location to display the live mission map."
        )


# =========================================================
# PAGE 2 — THERMAL DETECTION
# =========================================================

elif page == "🌡️ Thermal Detection":

    st.markdown(
        '<div class="main-title">'
        'Thermal Human Detection'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Custom YOLO model trained on AIResQ'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.gps_received:

        st.success(
            "📍 GPS ACTIVE — coordinates will automatically "
            "be attached to detected persons."
        )

    else:

        st.warning(
            "📍 GPS has not been captured. "
            "Open GPS Locations first if you want "
            "coordinates attached to detections."
        )

    uploaded_file = st.file_uploader(
        "Upload Thermal Image",
        type=["jpg", "jpeg", "png"],
        key="thermal_detection"
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        c1, c2 = st.columns(2)

        with c1:

            st.subheader(
                "Input Thermal Image"
            )

            st.image(
                image,
                use_container_width=True
            )

        with c2:

            st.subheader(
                "AI Detection Result"
            )

            if st.button(
                "🔍 Detect Person",
                type="primary",
                key="thermal_button"
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

                annotated = result.plot()

                st.image(
                    annotated,
                    use_container_width=True
                )

                # =========================================
                # CREATE PERSON + GPS RECORDS
                # =========================================

                detection_data = (
                    create_detection_records(
                        result
                    )
                )

                save_detection_results(
                    detection_data
                )

                if detection_data:

                    avg = (
                        st.session_state.avg_confidence
                    )

                    (
                        status,
                        recommendation,
                        css_class
                    ) = rescue_status(
                        avg,
                        len(detection_data)
                    )

                    st.success(
                        f"👤 {len(detection_data)} "
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

                    if st.session_state.gps_received:

                        st.markdown(
                            f"""
                            <div class="gps-card">

                            <h3>
                            📍 Detection GPS
                            </h3>

                            Latitude:
                            <b>
                            {st.session_state.latitude:.6f}
                            </b>

                            <br><br>

                            Longitude:
                            <b>
                            {st.session_state.longitude:.6f}
                            </b>

                            <br><br>

                            Accuracy:
                            <b>
                            {st.session_state.gps_accuracy:.0f} m
                            </b>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                else:

                    st.warning(
                        "No person detected."
                    )

        # -------------------------------------------------
        # RESULT TABLE
        # -------------------------------------------------

        if st.session_state.detections:

            st.divider()

            st.subheader(
                "📊 Person Detection Records"
            )

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
            "Upload a thermal image "
            "to begin detection."
        )


# =========================================================
# PAGE 3 — LIVE DETECTION
# =========================================================

elif page == "📹 Live Detection":

    st.markdown(
        '<div class="main-title">'
        'Continuous Live Detection'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Real-time human detection from a live camera'
        '</div>',
        unsafe_allow_html=True
    )

    st.warning(
        "Laptop camera = RGB mode. "
        "Actual thermal camera = Thermal mode."
    )

    camera_mode = st.radio(
        "Select Live Camera Mode",
        [
            "💻 Laptop / RGB Camera",
            "🌡️ Thermal Camera"
        ],
        horizontal=True
    )

    if camera_mode == "💻 Laptop / RGB Camera":

        live_model = rgb_model
        confidence_threshold = 0.45

        st.success(
            "RGB mode active."
        )

    else:

        live_model = thermal_model
        confidence_threshold = 0.25

        st.success(
            "Thermal mode active."
        )

    st.divider()

    st.subheader(
        "📹 Live Camera"
    )


    # -----------------------------------------------------
    # LIVE VIDEO PROCESSOR
    # -----------------------------------------------------

    class RescueHeatProcessor:

        def recv(self, frame):

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

            confidences = []

            for box in result.boxes:

                confidences.append(
                    float(box.conf[0]) * 100
                )

            with live_store["lock"]:

                live_store["count"] = len(
                    result.boxes
                )

                if confidences:

                    live_store["confidence"] = (
                        sum(confidences)
                        / len(confidences)
                    )

                    live_store["max_confidence"] = (
                        max(confidences)
                    )

                else:

                    live_store["confidence"] = 0.0
                    live_store["max_confidence"] = 0.0

            annotated = result.plot()

            return av.VideoFrame.from_ndarray(
                annotated,
                format="bgr24"
            )


    # -----------------------------------------------------
    # WEBRTC CAMERA
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # AUTO-REFRESH LIVE STATUS
    # -----------------------------------------------------

    @st.fragment(run_every="0.5s")
    def live_status_panel():

        with live_store["lock"]:

            live_count = (
                live_store["count"]
            )

            live_conf = (
                live_store["confidence"]
            )

            max_conf = (
                live_store["max_confidence"]
            )

        (
            status,
            recommendation,
            css_class
        ) = rescue_status(
            live_conf,
            live_count
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:

            st.metric(
                "Persons Detected",
                live_count
            )

        with col2:

            st.metric(
                "Average Confidence",
                f"{live_conf:.1f}%"
            )

        with col3:

            st.metric(
                "Highest Confidence",
                f"{max_conf:.1f}%"
            )

        with col4:

            st.metric(
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
                👤 <b>{live_count}</b>
                person(s) currently detected.
                </p>

                <p>
                Average confidence:
                <b>{live_conf:.1f}%</b>
                </p>

                <p>
                Highest confidence:
                <b>{max_conf:.1f}%</b>
                </p>

                <p>
                {recommendation}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            # =============================================
            # SHOW GPS WITH LIVE DETECTION
            # =============================================

            if st.session_state.gps_received:

                st.write("")

                st.markdown(
                    f"""
                    <div class="gps-card">

                    <h3>
                    📍 Current Rescue Coordinates
                    </h3>

                    <b>Latitude:</b>
                    {st.session_state.latitude:.6f}

                    <br><br>

                    <b>Longitude:</b>
                    {st.session_state.longitude:.6f}

                    <br><br>

                    <b>GPS Accuracy:</b>
                    {st.session_state.gps_accuracy:.0f} m

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.warning(
                    "📍 Person detected, but GPS "
                    "location has not been captured."
                )

        else:

            st.markdown(
                """
                <div class="safe-card">

                <h2>
                🟢 NO PERSON DETECTED
                </h2>

                <p>
                No human detection is currently
                present in the camera frame.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.caption(
            "Rescue status is based on AI detection "
            "confidence only. It is not a medical assessment."
        )


    live_status_panel()


# =========================================================
# PAGE 4 — GPS LOCATIONS
# =========================================================

elif page == "📍 GPS Locations":

    st.markdown(
        '<div class="main-title">'
        'Live Rescue GPS Location'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Capture current coordinates '
        'for rescue-team navigation'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Click the location button below. "
        "When the browser asks for location permission, "
        "select Allow."
    )

    # -----------------------------------------------------
    # GET BROWSER GPS
    # -----------------------------------------------------

    location = streamlit_geolocation()

    # -----------------------------------------------------
    # GPS RECEIVED
    # -----------------------------------------------------

    if (
        location
        and location.get("latitude") is not None
        and location.get("longitude") is not None
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

        accuracy = float(accuracy)

        # Save latest GPS
        st.session_state.latitude = latitude
        st.session_state.longitude = longitude
        st.session_state.gps_accuracy = accuracy
        st.session_state.gps_received = True

        st.success(
            "📡 GPS location received successfully!"
        )

        # -------------------------------------------------
        # GPS METRICS
        # -------------------------------------------------

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:

            st.metric(
                "GPS Status",
                "ACTIVE"
            )

        with col2:

            st.metric(
                "Latitude",
                f"{latitude:.6f}"
            )

        with col3:

            st.metric(
                "Longitude",
                f"{longitude:.6f}"
            )

        with col4:

            if accuracy > 0:

                st.metric(
                    "Accuracy",
                    f"{accuracy:.0f} m"
                )

            else:

                st.metric(
                    "Accuracy",
                    "Unknown"
                )

        st.divider()

        # -------------------------------------------------
        # GPS INFORMATION
        # -------------------------------------------------

        st.subheader(
            "📍 Current Rescue Coordinates"
        )

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
            "🗺️ Live Rescue Location Map"
        )

        location_data = pd.DataFrame(
            {
                "lat": [latitude],
                "lon": [longitude]
            }
        )

        st.map(
            location_data,
            zoom=16
        )

        st.success(
            "📍 GPS is ready. New person detections "
            "can now be associated with these coordinates."
        )

        st.warning(
            "Hackathon demo: this is the browser/laptop "
            "location. In the final drone system, GPS "
            "coordinates should come from the drone."
        )

    # -----------------------------------------------------
    # WAITING FOR GPS
    # -----------------------------------------------------

    else:

        st.warning(
            "📡 Waiting for GPS location. "
            "Click the location button and allow "
            "location access."
        )

        if st.session_state.gps_received:

            st.subheader(
                "📍 Last Known Location"
            )

            last_lat = (
                st.session_state.latitude
            )

            last_lon = (
                st.session_state.longitude
            )

            last_accuracy = (
                st.session_state.gps_accuracy
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                st.metric(
                    "Latitude",
                    f"{last_lat:.6f}"
                )

            with col2:

                st.metric(
                    "Longitude",
                    f"{last_lon:.6f}"
                )

            with col3:

                st.metric(
                    "Accuracy",
                    f"{last_accuracy:.0f} m"
                )

            last_location_data = pd.DataFrame(
                {
                    "lat": [last_lat],
                    "lon": [last_lon]
                }
            )

            st.map(
                last_location_data,
                zoom=16
            )

        else:

            st.info(
                "No live GPS location has "
                "been received yet."
            )


# =========================================================
# PAGE 5 — MISSION WORKFLOW
# =========================================================

elif page == "🔄 Mission Workflow":

    st.markdown(
        '<div class="main-title">'
        'End-to-End Mission Workflow'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'From drone search to rescue response'
        '</div>',
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
            "📹 Frames",
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
            "Attach location coordinates"
        ),
        (
            "8",
            "🚨 Alert",
            "Notify rescue team"
        ),
        (
            "9",
            "🚑 Rescue",
            "Human verification and response"
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
            workflow[i:i + 3]
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
# PAGE 6 — TECHNOLOGY STACK
# =========================================================

elif page == "⚙️ Technology Stack":

    st.markdown(
        '<div class="main-title">'
        'Technology Stack'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Technologies powering RescueHeat AI'
        '</div>',
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
            "Deep-learning backend"
        ),
        (
            "🌡️",
            "AIResQ",
            "Thermal training dataset"
        ),
        (
            "📹",
            "WebRTC",
            "Continuous camera streaming"
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
            technologies[i:i + 4]
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