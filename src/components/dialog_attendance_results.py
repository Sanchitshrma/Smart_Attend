"""Review dialog: editable attendance table, headcount check, voice check for students the camera
missed, and learning over time after the teacher saves. Nothing is saved until "Confirm & save"."""
import html
import logging

import pandas as pd
import streamlit as st

from src.components.voice_capture import voice_capture
from src.database import db
from src.database.db import DatabaseError, save_attendance_session
from src.services.attendance import apply_review_edits, confirmed_learning_samples
from src.validation import ValidationError, check_audio_bytes, decode_audio_base64

log = logging.getLogger("smartattend.review")


def _clear_session() -> None:
    st.session_state.review = None
    st.session_state.attendance_images = []
    st.session_state.attendance_video = None
    st.session_state.attendance_live = None
    st.session_state.live_round = st.session_state.get("live_round", 0) + 1


def _checkbox_key(ts, sid) -> str:
    return f"review_present_{ts}_{int(sid)}"


GROUP_LIMIT = 15  # names per recording; bigger groups make a long recording that is harder to split


def _set_note(review: dict, sid: int, note: str) -> None:
    df = review["df"]
    rows = df.index[df["ID"] == sid]
    if len(rows):
        df.at[rows[0], "Review"] = note


def _apply_group_check(review: dict, called: list[int], result, other_names: dict | None = None) -> dict:
    """Tick students whose voice was verified (method 'voice'); add a note for everyone else called."""
    df, logs, names = review["df"], review["logs"], review["names"]
    for sid, score in result.verified.items():
        row = df.index[df["ID"] == sid][0]
        df.at[row, "Present"], df.at[row, "Evidence"], df.at[row, "Review"] = True, f"Voice check {score:.2f}", ""
        for entry in logs:
            if entry["student_id"] == sid:
                entry.update(is_present=True, method="voice", match_score=float(score), source="auto")
        # A rendered checkbox cannot be changed in the same run: tick it before the table is drawn next run.
        review.setdefault("pending_ticks", set()).add(sid)
    for sid in called:
        if sid in result.verified:
            continue
        _set_note(review, sid, result.unsure.get(sid) or "Not heard clearly in the voice check: decide manually")
    for sid in result.proxies:
        _set_note(review, sid, "Their voice answered during the voice check for someone else (possible proxy)")
    return {
        "verified": [names.get(i, str(i)) for i in result.verified],
        "unsure": [names.get(i, str(i)) for i in called if i in result.unsure and i not in result.verified],
        "missing": [names.get(i, str(i)) for i in called if i not in result.verified and i not in result.unsure],
        "proxies": [names.get(i) or f"{(other_names or {}).get(i) or 'a student'} (not in this class)" for i in result.proxies],
        "speakers": result.speakers,
    }


def _show_group_summary(summary: dict) -> None:
    if summary["verified"]:
        st.success("Voice verified and marked present: " + ", ".join(summary["verified"]))
    if summary["proxies"]:
        st.error("Possible proxy: a voice in the recording belongs to " + ", ".join(summary["proxies"]) +
                 ", who was not one of the names called.")
    if summary["unsure"]:
        st.warning("Close but not certain (decide yourself): " + ", ".join(summary["unsure"]))
    if summary["missing"]:
        st.info("No matching voice found for: " + ", ".join(summary["missing"]))


def _voice_check_section(review: dict) -> None:
    logs = review["logs"]
    ts = logs[0]["timestamp"]
    names = review["names"]
    missing = [l["student_id"] for l in logs if not st.session_state.get(_checkbox_key(ts, l["student_id"]))]
    last = review.get("voice_summary")
    if not missing and not last:
        return
    with st.expander(f"Voice check · {len(missing)} student(s) not recognised", icon=":material/mic:", expanded=True):
        if last:
            _show_group_summary(last)
        if not missing:
            return
        with_voice = [sid for sid in missing if review["has_voice"].get(sid)]
        without = [names.get(sid, str(sid)) for sid in missing if not review["has_voice"].get(sid)]
        if without:
            st.caption("No voice sample, decide manually: " + ", ".join(without))
        if not with_voice:
            return
        called = st.multiselect(
            "Names to call in this recording", with_voice, default=with_voice[:GROUP_LIMIT],
            format_func=lambda i: names.get(i, str(i)), max_selections=GROUP_LIMIT, key=f"vc_group_pick_{ts}")
        if not called:
            return
        st.markdown("**How:** start recording, read the names below one by one, and let each student answer "
                    "*“Present, my name is …”* (2–3 seconds) before you read the next name. Then press **Check voices**.")
        st.markdown("**Call in this order:** " + " → ".join(html.escape(names.get(i, str(i))) for i in called))
        rnd = review.setdefault("voice_round", 0)
        capture = voice_capture(key=f"vc_group_{ts}_{rnd}", action_label="Check voices")
        if not (isinstance(capture, dict) and capture.get("submitted")):
            return
        if capture.get("request_id") == review.get("voice_request"):
            return
        review["voice_request"] = capture.get("request_id")
        try:
            from src.pipelines.voice_pipeline import VoiceError, check_group_recording
            audio = check_audio_bytes(decode_audio_base64(capture.get("audio_base64", "")))
            with st.spinner("Separating the voices and checking each one…"):
                gallery = db.get_voice_gallery()
                profiles = {int(r["student_id"]): r["voice_embedding"] for r in gallery}
                result = check_group_recording(audio, [int(c) for c in called], profiles)
            other_names = {int(r["student_id"]): r.get("name") for r in gallery}
        except ImportError:
            st.error("Voice recognition is not installed on this server (see README: resemblyzer).")
            return
        except (ValidationError, VoiceError, DatabaseError) as exc:
            st.error(str(exc))
            return
        review["voice_summary"] = _apply_group_check(review, [int(c) for c in called], result, other_names)
        review["voice_round"] = rnd + 1
        st.rerun(scope="fragment")


def show_attendance_result(review: dict) -> None:
    """Editable review table. Nothing is saved until the teacher confirms; edits are recorded as manual."""
    df, logs = review["df"], review["logs"]
    st.write("Review attendance before confirming. Tick **Present** for anyone who attended.")
    timestamp = logs[0]["timestamp"] if logs else "x"
    for sid in review.pop("pending_ticks", set()):
        st.session_state[_checkbox_key(timestamp, sid)] = True
    edited_rows = []
    with st.container(key="attendance_review_grid"):
        for _, row in df.iterrows():
            sid = int(row["ID"])
            key = _checkbox_key(timestamp, sid)
            st.session_state.setdefault(key, bool(row["Present"]))
            with st.container(key=f"review_row_{sid}"):
                info, tick = st.columns([5, 1.6], vertical_alignment="center", gap="small")
                is_present = bool(st.session_state.get(key))
                evidence = row["Evidence"] if row["Evidence"] not in ("-", "", None) else "No match"
                detail = evidence + (f" · {row['Review']}" if row["Review"] else "")
                badge = "present" if is_present else "absent"
                info.markdown(
                    f"<div class='sa-row-name'>{html.escape(str(row['Name']))}"
                    f"<span class='sa-badge sa-badge-{badge}'>{badge.title()}</span></div>"
                    f"<div class='sa-row-detail'>{html.escape(detail)}</div>", unsafe_allow_html=True)
                is_present = tick.checkbox("Present", key=key)
            edited_rows.append({"ID": sid, "Present": is_present})

    edited = df[["ID"]].merge(pd.DataFrame(edited_rows), on="ID", how="left")
    final = apply_review_edits(logs, edited)
    n_present = sum(1 for l in final if l["is_present"])
    n_manual = sum(1 for l in final if l.get("source") == "manual")
    st.caption(f"{n_present} present · {len(final) - n_present} absent · {n_manual} manually changed")

    headcount = review.get("headcount")
    if headcount is not None:
        # Students confirmed by the voice check were not visible in the video, so they are not counted.
        n_by_camera = sum(1 for l in final if l["is_present"] and l.get("method") != "voice")
        if n_by_camera > headcount:
            st.warning(f"{n_by_camera} students are marked present from the video, but it shows only about {headcount} people. "
                       "Check for proxy attendance.")
        elif headcount > n_by_camera:
            st.info(f"The video shows about {headcount} people and {n_by_camera} are marked present from it. "
                    "Use the voice check below for students who were not recognised.")

    if review.get("voice_check", True):
        _voice_check_section(review)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Discard", width="stretch", key="att_discard"):
            _clear_session()
            st.rerun()
    with col2:
        if st.button("Confirm & save", width="stretch", type="primary", key="att_confirm"):
            teacher_id = st.session_state.teacher_data["teacher_id"]
            try:
                written = save_attendance_session(final, teacher_id=teacher_id)
            except DatabaseError as exc:
                st.error(str(exc))
                return
            learned = 0
            if written and review.get("learn"):
                try:  # learning over time never blocks saving attendance
                    learned = db.add_face_templates(teacher_id, logs[0]["subject_id"],
                                                    confirmed_learning_samples(review["learn"], final))
                except DatabaseError:
                    log.warning("Could not store learned face templates", exc_info=True)
            _clear_session()
            if not written:
                st.session_state.flash = "This session was already saved, so nothing new was added."
            else:
                st.session_state.flash = "Attendance saved." + (
                    f" Learned {learned} new face sample(s) to recognise students better next time." if learned else "")
            st.rerun()


@st.dialog("Attendance report", width="large")
def attendance_result_dialog():
    review = st.session_state.get("review")
    if not review:
        st.info("Nothing to review.")
        return
    if review.get("summary"):
        st.caption(review["summary"])
    show_attendance_result(review)
