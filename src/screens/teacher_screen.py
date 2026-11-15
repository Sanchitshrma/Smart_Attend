from __future__ import annotations

import logging
from html import escape

import streamlit as st

from src.components.dialog_add_photo import add_photos_dialog
from src.components.dialog_attendance_results import attendance_result_dialog
from src.components.dialog_create_subject import create_subject_dialog
from src.components.dialog_share_subject import share_subject_dialog
from src.components.dialog_add_video import add_video_dialog
from src.components.camera_capture import attendance_camera
from src.components.footer import footer_dashboard
from src.components.header import header_dashboard
from src.components.subject_card import subject_card
from src.database import db
from src.screens.admin_ui import admin_tab
from src.screens.marks_ui import teacher_marks_tab
from src.database.db import DatabaseError
from src.pipelines.face_pipeline import analyze_photo
from src.services.attendance import (
    DEFAULT_MIN_ATTENDANCE,
    attendance_trend,
    build_review_session,
    students_at_risk,
    learnable_samples,
    merge_photo_results,
    new_session_timestamp,
    parse_ts,
    records_to_frame,
    session_summary,
    subject_rates,
    teacher_overview,
)
from src.ui.base_layout import empty_state, page_header, show_flash, style_base_layout, user_badge

log = logging.getLogger("smartattend.teacher")
TABS = ["Overview", "Take attendance", "Subjects", "Marks", "Records"]
ADMIN_TAB = "Admin"


def teacher_screen():
    style_base_layout()
    if "teacher_data" in st.session_state:
        teacher_dashboard()
    elif st.session_state.get("teacher_login_type") == "register":
        teacher_screen_register()
    else:
        teacher_screen_login()


# ------------------------------------------------------------------- dashboard
def teacher_dashboard():
    teacher = st.session_state.teacher_data
    right = header_dashboard()
    with right:
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.markdown(user_badge(teacher["name"]), unsafe_allow_html=True)
        if c2.button("Log out", key="teacher_logout", icon=":material/logout:"):
            st.session_state.pop("teacher_data", None)
            st.session_state.is_logged_in = False
            st.session_state.user_role = None
            st.rerun()
    show_flash()

    # The role shown here only decides what to DISPLAY; db.* re-reads the role on every call.
    tabs = TABS + ([ADMIN_TAB] if teacher.get("role") == "admin" else [])
    tab = st.segmented_control("Section", tabs, default=TABS[0], key="teacher_tab", label_visibility="collapsed")
    st.divider()
    try:
        {
            ADMIN_TAB: lambda: admin_tab(teacher["teacher_id"]),
            "Overview": tab_overview,
            "Take attendance": tab_take_attendance,
            "Subjects": tab_subjects,
            "Marks": lambda: teacher_marks_tab(teacher["teacher_id"]),
            "Records": tab_records,
        }.get(tab or TABS[0], tab_overview)()
    except DatabaseError as exc:
        st.error(str(exc))
    footer_dashboard()


def _load_detail(teacher_id):
    return records_to_frame(db.get_attendance_for_teacher(teacher_id))


def tab_overview():
    teacher_id = st.session_state.teacher_data["teacher_id"]
    page_header("Overview", "A summary of your classes, calculated from saved attendance.")
    subjects = db.get_teacher_subjects(teacher_id)
    detail = _load_detail(teacher_id)
    o = teacher_overview(subjects, detail)

    with st.container(key="sa_metrics_overview"):
        c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subjects", o["subjects"])
    c2.metric("Enrolments", o["students"], help="Students enrolled, counted once in each of your subjects.")
    c3.metric("Sessions taken", o["sessions"])
    c4.metric("Overall attendance", "–" if o["rate"] is None else f"{o['rate']}%")
    st.write("")

    if detail.empty:
        empty_state("No attendance has been recorded yet. Go to <b>Take attendance</b> to record your first session.")
        return
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Recent sessions")
        st.dataframe(session_summary(detail).head(8), hide_index=True, width="stretch")
    with right:
        st.subheader("Attendance rate by subject")
        st.bar_chart(subject_rates(detail), x="Subject", y="Rate %", horizontal=True, color="#4F46E5")

    trend = attendance_trend(detail)
    if len(trend) > 1:
        st.subheader("Attendance trend")
        st.line_chart(trend, x="Date", y="Rate %", color="#0D9488")
    st.subheader("Students at risk")
    minimum = st.number_input("Minimum attendance (%)", 0.0, 100.0, DEFAULT_MIN_ATTENDANCE, 5.0, key="min_attendance",
                              help="Students below this rate in a subject are listed.")
    risk = students_at_risk(detail, minimum)
    if risk.empty:
        st.success(f"Nobody is below {minimum:g}% in any subject.")
    else:
        st.dataframe(risk, hide_index=True, width="stretch")


# ------------------------------------------------------------ take attendance
LIVE_SECONDS = 8
LIVE_FPS = 3
CAPTURE_MODES = {"live": ":material/videocam: Live camera", "video": ":material/upload: Upload video",
                 "photos": ":material/photo_camera: Photos"}


def _clear_capture() -> None:
    st.session_state.attendance_images, st.session_state.attendance_video = [], None
    st.session_state.attendance_live = None
    st.session_state.live_round = st.session_state.get("live_round", 0) + 1   # fresh camera widget


def _live_capture_input() -> None:
    """Live camera: the browser sends a few JPEG frames per second; no video file is created."""
    rnd = st.session_state.setdefault("live_round", 0)
    cap = attendance_camera(key=f"live_cam_{rnd}", seconds=LIVE_SECONDS, fps=LIVE_FPS)
    if not (isinstance(cap, dict) and cap.get("submitted") and cap.get("mode") == "attendance"):
        return
    if cap.get("request_id") == st.session_state.get("live_request"):
        return
    st.session_state.live_request = cap.get("request_id")
    frames = [f for f in (cap.get("frames") or []) if isinstance(f, str)]
    if not frames:
        st.error("No frames were received from the camera. Allow camera access and try again.")
        return
    st.session_state.attendance_live = {"frames": frames, "times": cap.get("times") or []}


def _frame_bytes(b64: str) -> bytes | None:
    import base64
    try:
        return base64.b64decode(b64.split(",", 1)[1] if b64.startswith("data:") else b64)
    except (ValueError, TypeError):
        return None


def tab_take_attendance():
    teacher_id = st.session_state.teacher_data["teacher_id"]
    page_header("Take attendance", "Pick a subject, capture the class with the live camera, then review the result.")
    st.session_state.setdefault("attendance_images", [])
    st.session_state.setdefault("attendance_video", None)
    st.session_state.setdefault("attendance_live", None)

    subjects = db.get_teacher_subjects(teacher_id)
    if not subjects:
        empty_state("You have not created a subject yet. Open <b>Subjects</b> to create one.")
        return
    by_id = {s["subject_id"]: s for s in subjects}
    subject_id = st.selectbox(
        "Subject", list(by_id), format_func=lambda i: f"{by_id[i]['name']} · {by_id[i]['subject_code']} (Sec {by_id[i]['section']})"
    )
    mode = st.segmented_control("Capture with", list(CAPTURE_MODES), format_func=CAPTURE_MODES.get,
                                default="live", key="capture_mode") or "live"
    if mode == "live":
        st.caption(f"Stand at the front and pan slowly across the room for {LIVE_SECONDS} seconds. "
                   "Only still frames are analysed in memory; no video is recorded or saved.")
        _live_capture_input()
    elif mode == "video":
        st.caption("Already recorded a short clip on your phone? Upload it here (it is never stored).")
        if st.button("Choose a video", icon=":material/upload:", width="stretch"):
            add_video_dialog()
    else:
        st.caption("Classroom photos work too. Add several angles for a large class.")
        if st.button("Add photos", icon=":material/photo_camera:", width="stretch"):
            add_photos_dialog()

    images, video, live = st.session_state.attendance_images, st.session_state.attendance_video, st.session_state.attendance_live
    ready = []
    if live:
        ready.append(f"live capture ({len(live['frames'])} frames)")
    if video:
        ready.append(f"video “{escape(video['name'])}”")
    if images:
        ready.append(f"{len(images)} photo(s)")
    if ready:
        with st.container(border=True, key="capture_ready"):
            st.markdown("**Ready to analyse:** " + ", ".join(ready))
            if live:
                picks = live["frames"][:: max(1, len(live["frames"]) // 4)][:4]
                cols = st.columns(4)
                for i, b64 in enumerate(picks):
                    data = _frame_bytes(b64)
                    if data:
                        cols[i].image(data, width="stretch")
            if video:
                st.video(video["bytes"])
            if images:
                cols = st.columns(4)
                for i, img in enumerate(images):
                    cols[i % 4].image(img, width="stretch", caption=f"Photo {i + 1}")

    has_input = bool(images or video or live)
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Clear", width="stretch", icon=":material/delete:", disabled=not has_input):
            _clear_capture()
            st.rerun()
    with b2:
        run = st.button("Run face analysis", width="stretch", type="primary", icon=":material/analytics:", disabled=not has_input)

    if not run:
        return
    enrolled = db.get_enrolled_students(teacher_id, subject_id)
    if not enrolled:
        st.warning("No students are enrolled in this subject yet. Share the subject code from the Subjects tab.")
        return

    allowed = {int(s["student_id"]) for s in enrolled}
    results, labels, headcount, video_note = [], [], None, ""
    try:
        if live:
            from src.pipelines.video_pipeline import VideoError, analyze_frames, decode_live_frames
            bar = st.progress(0.0, text="Checking the live frames…")
            try:
                frames, seconds = decode_live_frames(live["frames"], live.get("times"))
                lres = analyze_frames(frames, seconds or LIVE_SECONDS, allowed_ids=allowed,
                                      progress=lambda f: bar.progress(f, text="Checking the live frames…"))
            except VideoError as exc:
                bar.empty()
                st.error(str(exc))
                return
            bar.empty()
            results.append(lres.photo)
            labels.append("Live camera")
            headcount = lres.headcount
            video_note = f"Live camera: {lres.frames_analysed} frames, about {lres.headcount} people seen · "
        if video:
            from src.pipelines.video_pipeline import VideoError, analyze_video
            bar = st.progress(0.0, text="Scanning the video frame by frame…")
            try:
                vres = analyze_video(video["bytes"], allowed_ids=allowed, progress=lambda f: bar.progress(f, text="Scanning the video frame by frame…"))
            except VideoError as exc:
                bar.empty()
                st.error(str(exc))
                return
            bar.empty()
            results.append(vres.photo)
            labels.append("Video")
            headcount = max(headcount or 0, vres.headcount)
            video_note += f"Video: {vres.frames_analysed} frames over {vres.seconds:.0f} s, about {headcount} people seen · "
        if images:
            with st.spinner("Scanning classroom photos…"):
                for k, img in enumerate(images, start=1):
                    results.append(analyze_photo(img, allowed_ids=allowed))
                    labels.append(f"Photo {k}")
    except DatabaseError as exc:
        st.error(str(exc))
        return
    except Exception:  # noqa: BLE001
        log.exception("Face analysis failed")
        st.error("Face analysis failed. Check that the video or photos are valid and try again.")
        return

    present, scores, notes, t = merge_photo_results(results, allowed, labels)
    if t["faces"] == 0:
        st.warning("No faces were detected. Try a closer, steadier or better-lit video or photo.")
        return
    summary = video_note + (
        f"{t['faces']} face(s)/people · {len(present)} identified · {t['ambiguous'] + t['duplicate']} need review · "
        f"{t['unknown']} not recognised" + (f" · {t['small']} very small" if t["small"] else "")
    )
    if t["unknown"]:
        st.info("Faces that were not recognised are not assigned to anyone. Use the voice check or tick students manually.")
    df, logs = build_review_session(enrolled, present, subject_id, new_session_timestamp(), scores=scores, review_notes=notes)
    st.session_state.review = {
        "df": df, "logs": logs, "summary": summary, "headcount": headcount,
        "learn": learnable_samples(results),
        "names": {int(s["student_id"]): s["name"] for s in enrolled},
        "has_voice": {int(s["student_id"]): bool(s.get("voice_embedding")) for s in enrolled},
    }
    attendance_result_dialog()


# --------------------------------------------------------------------- subjects
def tab_subjects():
    teacher_id = st.session_state.teacher_data["teacher_id"]
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    with left:
        page_header("Subjects", "Create subjects and share their join code with students.")
    with right:
        if st.button("Create subject", type="primary", width="stretch", icon=":material/add:"):
            create_subject_dialog(teacher_id)

    subjects = db.get_teacher_subjects(teacher_id)
    if not subjects:
        empty_state("No subjects yet. Press <b>Create subject</b> to add your first one.")
        return
    cols = st.columns(2)
    for i, sub in enumerate(sorted(subjects, key=lambda s: s["name"].lower())):  # every subject, not just the last
        def share_ui(sub=sub):
            if st.button("Share code", key=f"share_{sub['subject_id']}", icon=":material/share:"):
                share_subject_dialog(sub["name"], sub["subject_code"])

        with cols[i % 2]:
            subject_card(
                sub["name"], sub["subject_code"], sub["section"],
                stats=[("🫂", "students", sub["total_students"]), ("🕰️", "classes", sub["total_classes"])],
                footer_callback=share_ui,
            )
            with st.expander(f"Edit {sub['name']}"):
                with st.form(f"edit_subject_{sub['subject_id']}"):
                    edited_name = st.text_input("Subject name", value=sub["name"], max_chars=100)
                    edited_code = st.text_input("Subject code", value=sub["subject_code"], max_chars=20)
                    edited_section = st.text_input("Section", value=sub["section"], max_chars=20)
                    save_subject = st.form_submit_button("Save changes", type="primary")
                if save_subject:
                    try:
                        db.update_subject(teacher_id, sub["subject_id"], edited_code, edited_name, edited_section)
                        st.session_state.flash = "Subject updated."
                        st.rerun()
                    except DatabaseError as exc:
                        st.error(str(exc))

                st.caption("Deleting a subject permanently removes its enrollments, attendance records, assessments, and marks.")
                confirm_delete = st.checkbox(
                    "I understand this cannot be undone",
                    key=f"confirm_delete_subject_{sub['subject_id']}",
                )
                if st.button(
                    "Delete subject", key=f"delete_subject_{sub['subject_id']}",
                    type="secondary", disabled=not confirm_delete,
                ):
                    try:
                        db.delete_subject(teacher_id, sub["subject_id"])
                        st.success(f"{sub['name']} deleted.")
                        st.rerun()
                    except DatabaseError as exc:
                        st.error(str(exc))


# --------------------------------------------------------------------- records
def tab_records():
    teacher_id = st.session_state.teacher_data["teacher_id"]
    page_header("Attendance records", "Every saved session. Filter by subject or export the detailed log.")
    detail = _load_detail(teacher_id)
    if detail.empty:
        empty_state("No attendance records yet.")
        return

    choice = st.selectbox("Subject", ["All subjects"] + sorted(detail["Subject"].unique()))
    view = detail if choice == "All subjects" else detail[detail["Subject"] == choice]
    st.dataframe(session_summary(view), hide_index=True, width="stretch")

    history = view.assign(
        When=view["when"].map(_format_timestamp),
        Attendance=view["Present"].map({True: "Present", False: "Absent"}),
    )
    st.subheader("Student attendance history")
    session_columns = ["session", "Subject", "Subject Code"]
    for (session, subject, code), rows in history.groupby(session_columns, sort=False, dropna=False):
        when = _format_timestamp(session) or "Unknown date"
        absent_count = int((rows["Attendance"] == "Absent").sum())
        label = f"{when} · {subject} ({code}) · {len(rows)} students · {absent_count} absent"
        with st.expander(label, expanded=False):
            _render_attendance_history(rows.sort_values("When", ascending=False))

    _corrections_panel(teacher_id, view)

    export = view.assign(Status=view["Present"].map({True: "Present", False: "Absent"}))
    export = export.assign(When=export["when"].map(_format_timestamp))
    csv = export[["When", "Subject", "Subject Code", "Student", "Student status", "Status"]].sort_values("When", ascending=False).to_csv(index=False)
    st.download_button("Download detailed CSV", csv, file_name="attendance.csv", mime="text/csv", icon=":material/download:")


def _format_timestamp(value) -> str:
    """Show attendance timestamps in the same readable format as the summary table."""
    if value is None:
        return ""
    parsed = parse_ts(str(value))
    if parsed is None:
        return str(value)
    return parsed.strftime("%d %b %Y, %H:%M")


def _render_attendance_history(history):
    """Render names with a visible enrollment-status badge in the detailed history."""
    body = []
    status_classes = {"Active": "active", "Unenrolled": "unenrolled", "Deleted": "deleted"}
    for row in history.to_dict("records"):
        status = str(row.get("Student status") or "Active")
        badge_class = status_classes.get(status, "deleted")
        attendance = str(row.get("Attendance", ""))
        attendance_class = "present" if attendance == "Present" else "absent"
        cells = (
            row.get("When", ""), row.get("Subject", ""), row.get("Subject Code", ""),
            row.get("Student", ""), row.get("Attendance", ""),
        )
        body.append(
            "<tr>"
            + "".join(f"<td>{escape(str(value if value is not None else ''))}</td>" for value in cells[:4])
            + f'<td><span class="sa-status-badge {badge_class}">{escape(status)}</span></td>'
            + f'<td><span class="sa-status-badge {attendance_class}">{escape(attendance)}</span></td>'
            + "</tr>"
        )
    st.markdown(
        '<div class="sa-history-scroll"><table class="sa-history-table">'
        "<thead><tr><th>When</th><th>Subject</th><th>Code</th><th>Student</th>"
        "<th>Enrollment</th><th>Attendance</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _corrections_panel(teacher_id, view):
    """Correct a saved record. Every change is audited (who, when, old/new, reason)."""
    with st.expander("Correct a saved attendance record"):
        sessions = view[["session", "Subject", "Subject Code"]].drop_duplicates().sort_values("session", ascending=False)
        if sessions.empty:
            st.caption("No sessions to correct.")
            return
        labels = {r.session: f"{r.Subject} · {r.session}" for r in sessions.itertuples()}
        ts = st.selectbox("Session", list(labels), format_func=labels.get, key="corr_session")
        rows = view[view["session"] == ts].sort_values("Student")
        rows = rows[(rows["Student status"] == "Active") & rows["student_id"].notna()]
        if rows.empty:
            st.info("Only deleted or unenrolled students are in this session; their history is read-only.")
            return
        # Keyed by student id: two students can share a name.
        by_id = {int(r["student_id"]): r for _, r in rows.iterrows()}
        dup = rows["Student"].duplicated(keep=False).tolist()
        labels_s = {int(r["student_id"]): (f"{r['Student']} (#{int(r['student_id'])})" if d else str(r["Student"]))
                    for (_, r), d in zip(rows.iterrows(), dup)}
        pick = st.selectbox("Student", list(by_id), format_func=labels_s.get, key="corr_student")
        who = labels_s[pick]
        current = bool(by_id[pick]["Present"])
        new = st.radio("Mark as", ["Present", "Absent"], index=0 if current else 1, horizontal=True, key="corr_new")
        reason = st.text_input("Reason (required)", max_chars=200, key="corr_reason")
        if st.button("Save correction", key="corr_save"):
            subject_id = by_id[pick]["subject_id"]
            student_id = pick
            try:
                changed = db.correct_attendance(teacher_id, int(subject_id), ts, int(student_id), new == "Present", reason)
            except DatabaseError as exc:
                st.error(str(exc))
                return
            if changed:
                st.session_state.flash = f"Updated {who}: now {new.lower()}. The change was recorded in the audit trail."
                st.rerun()
            st.info("That is already the saved value; nothing changed.")
        trail = db.get_audit_trail(teacher_id, int(rows.iloc[0]["subject_id"])) if len(rows) else []
        if trail:
            st.caption("Audit trail for this subject")
            st.dataframe(
                [{"When": t["changed_at"] if "changed_at" in t else "", "Student": (t.get("students") or {}).get("name") or t.get("student_name") or t.get("student_id"),
                  "From": "Present" if t["old_present"] else "Absent", "To": "Present" if t["new_present"] else "Absent", "Reason": t["reason"]} for t in trail],
                hide_index=True, width="stretch")


# ------------------------------------------------------------------ auth screens
def _auth_header(back_key: str):
    right = header_dashboard()
    with right:
        if st.button("Back to home", key=back_key):
            st.session_state.login_type = None
            st.session_state.teacher_login_type = "login"
            st.query_params.clear()
            st.rerun()


def teacher_screen_login():
    _auth_header("teacher_back_login")
    show_flash()
    page_header("Teacher login", "Sign in to manage subjects and take attendance.")
    with st.form("teacher_login"):
        username = st.text_input("Username", placeholder="ananya7", max_chars=32, autocomplete="username",
                                 help="3-32 characters; include at least one letter and one number. Dot, dash and underscore are also allowed.")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Log in", type="primary", width="stretch")
    if submitted:
        if not username or not password:
            st.error("Enter both username and password.")
        else:
            try:
                teacher = db.teacher_login(username, password)
            except DatabaseError as exc:
                st.error(str(exc))
                teacher = None
            else:
                if teacher:
                    st.session_state.update(teacher_data=teacher, is_logged_in=True, user_role="teacher")
                    st.rerun()
                st.error("Invalid username or password.")
    if st.button("I have an invitation code"):
        st.session_state.teacher_login_type = "register"
        st.rerun()
    _admin_setup_panel()
    footer_dashboard()


def teacher_screen_register():
    _auth_header("teacher_back_register")
    page_header("Teacher sign-up (invitation only)", "Ask an administrator for an invitation code. Each code works once and stays valid until used or revoked.")
    with st.form("teacher_register"):
        token = st.text_input("Invitation code", max_chars=100, autocomplete="off")
        username = st.text_input("Username", placeholder="ananya7", max_chars=32, autocomplete="username",
                                 help="3-32 characters; include at least one letter and one number. Dot, dash and underscore are also allowed.")
        name = st.text_input("Full name", placeholder="Ananya Roy", max_chars=60,
                             help="Use letters, spaces, apostrophes, hyphens or dots; numbers are not allowed.")
        password = st.text_input("Password", type="password", autocomplete="new-password")
        confirm = st.text_input("Confirm password", type="password", autocomplete="new-password")
        submitted = st.form_submit_button("Create account", type="primary", width="stretch")
    if submitted:
        try:
            db.register_teacher_with_invite(username, name, password, confirm, token)
        except DatabaseError as exc:  # InvalidInput, PermissionDenied, DuplicateError, RateLimited, ...
            st.error(str(exc))
        else:
            st.session_state.teacher_login_type = "login"
            st.session_state.flash = "Account created. Please log in."
            st.rerun()
    if st.button("I already have an account"):
        st.session_state.teacher_login_type = "login"
        st.rerun()
    footer_dashboard()


def _admin_setup_panel():
    """Shown only while no administrator exists AND ADMIN_SETUP_TOKEN is configured."""
    try:
        available = db.admin_setup_available()
    except DatabaseError:
        return
    if not available:
        return
    with st.expander("First-time setup: create the administrator"):
        st.caption("Needs the ADMIN_SETUP_TOKEN from your server secrets. This disappears once an administrator exists.")
        with st.form("admin_setup"):
            setup_token = st.text_input("Setup token", type="password", autocomplete="off")
            username = st.text_input("Admin username", max_chars=32, autocomplete="username",
                                     help="3-32 characters; include at least one letter and one number. Dot, dash and underscore are also allowed.")
            name = st.text_input("Full name", max_chars=60,
                                 help="Use letters, spaces, apostrophes, hyphens or dots; numbers are not allowed.")
            password = st.text_input("Password", type="password", autocomplete="new-password")
            confirm = st.text_input("Confirm password", type="password", autocomplete="new-password")
            if st.form_submit_button("Create administrator", type="primary"):
                try:
                    db.bootstrap_admin(username, name, password, confirm, setup_token)
                except DatabaseError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.flash = "Administrator created. Remove ADMIN_SETUP_TOKEN from your secrets, then log in."
                    st.rerun()
