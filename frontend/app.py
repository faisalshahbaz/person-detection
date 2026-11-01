import asyncio
import base64
import json
import os
import threading

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from api_client import BACKEND_URL, BackendClient
from overlay import draw_detections, frame_to_b64_jpeg

FRAME_SAMPLE_RATE = int(os.getenv("FRAME_SAMPLE_RATE", "3"))
STUN_SERVER = os.getenv("STUN_SERVER", "stun:stun.l.google.com:19302")

st.set_page_config(page_title="Person Detection", layout="wide")
st.title("Person Detection")

if "token" not in st.session_state:
    st.session_state.token = None
if "client" not in st.session_state:
    st.session_state.client = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None

with st.sidebar:
    st.subheader("Login")
    username = st.text_input("Username", value="demo")
    password = st.text_input("Password", type="password", value="demo123")
    if st.button("Log in"):
        try:
            client = BackendClient()
            token = client.login(username, password)
            st.session_state.token = token
            st.session_state.client = BackendClient(token)
            st.success("Logged in")
        except Exception as exc:
            st.error(f"Login failed: {exc}")

if not st.session_state.token:
    st.info("Log in with demo / demo123 to test video.")
    st.stop()

client: BackendClient = st.session_state.client
source = st.radio(
    "Video source",
    ["Webcam (live)", "Upload video file"],
    horizontal=True,
)

if source.startswith("Webcam"):
    c1, c2 = st.columns(2)
    if c1.button("Start session"):
        try:
            st.session_state.session_id = client.create_session("webcam")
            st.success("Webcam session started")
        except Exception as exc:
            st.error(str(exc))
    if c2.button("Stop session"):
        if st.session_state.session_id:
            try:
                client.stop_session(st.session_state.session_id)
            except Exception:
                pass
        st.session_state.session_id = None

    infer_lock = threading.Lock()
    frame_counter = {"n": 0}
    latest = {"boxes": [], "scores": [], "count": 0}

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if not st.session_state.session_id:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        frame_counter["n"] += 1
        if frame_counter["n"] % FRAME_SAMPLE_RATE != 0:
            with infer_lock:
                img = draw_detections(
                    img, latest["boxes"], latest["scores"], latest["count"]
                )
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        try:
            b64 = frame_to_b64_jpeg(img)
            result = client.infer(st.session_state.session_id, b64)
            with infer_lock:
                latest["boxes"] = result.get("boxes", [])
                latest["scores"] = result.get("scores", [])
                latest["count"] = result.get("person_count", 0)
            img = draw_detections(
                img, latest["boxes"], latest["scores"], latest["count"]
            )
        except Exception:
            pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="person-detection-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": [STUN_SERVER]}]},
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback,
        async_processing=True,
    )
    st.metric("Person count", latest["count"])
    st.caption("Use http://localhost:8501 so the browser allows webcam access.")

else:
    uploaded = st.file_uploader(
        "Choose a video", type=["mp4", "avi", "mov", "mkv", "webm"]
    )
    if st.button("Run detection on upload") and uploaded is not None:
        preview = st.empty()
        count_box = st.empty()
        status = st.empty()
        data = uploaded.read()
        ws_url = (
            BACKEND_URL.replace("http://", "ws://").replace("https://", "wss://")
        )

        async def run_upload_job() -> None:
            import websockets

            token = st.session_state.token
            # Upload starts the job; backend waits for this WebSocket before decoding frames.
            job = await asyncio.to_thread(client.upload, uploaded.name, data)
            job_id = job["job_id"]
            stream_url = f"{ws_url}/jobs/{job_id}/stream?token={token}"
            status.info(f"Processing job {job_id[:8]}…")
            async with websockets.connect(stream_url) as ws:
                await ws.send("ready")
                while True:
                    msg = await ws.recv()
                    payload = json.loads(msg)
                    if payload.get("type") == "frame":
                        count_box.metric("Person count", payload.get("person_count", 0))
                        raw = payload.get("image_b64")
                        if raw:
                            arr = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
                            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if bgr is not None:
                                preview.image(
                                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                    caption=f"Frame {payload.get('frame_index', 0)}",
                                    use_container_width=True,
                                )
                    elif payload.get("type") == "done":
                        status.success(
                            f"Finished — {payload.get('frames_processed', 0)} frames"
                        )
                        break
                    elif payload.get("type") == "error":
                        status.error(payload.get("detail", "Error"))
                        break

        try:
            asyncio.run(run_upload_job())
        except Exception as exc:
            st.error(f"Upload failed: {exc}")
