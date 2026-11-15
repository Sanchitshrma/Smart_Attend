import streamlit as st

from src.pipelines.video_pipeline import ALLOWED_VIDEO_EXTENSIONS, MAX_VIDEO_BYTES
from src.settings import VIDEO_MAX_SECONDS


@st.dialog("Add a classroom video")
def add_video_dialog():
    st.write(
        f"Record a slow 5–10 second pan across the classroom on your phone (under {VIDEO_MAX_SECONDS} s), "
        "then upload it here. Every face is checked in several frames, so students at the back or "
        "turning their heads are still recognised. The video stays in memory and is never saved."
    )
    f = st.file_uploader("Choose a video", type=ALLOWED_VIDEO_EXTENSIONS, key="dialog_video")
    if f and st.button("Use this video", type="primary", width="stretch"):
        data = f.getvalue()
        if len(data) > MAX_VIDEO_BYTES:
            st.error(f"The video is larger than {MAX_VIDEO_BYTES // (1024 * 1024)} MB. Record a shorter clip.")
            return
        st.session_state.attendance_video = {"name": f.name, "bytes": data}
        st.rerun()
