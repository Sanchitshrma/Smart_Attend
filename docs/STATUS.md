# SmartAttend: status of every requested item

Legend: **V** fixed and verified (automated tests and/or browser run in a sandbox with an in-memory DB) ·
**R** implemented, needs real-world validation · **P** partially addressed · **N** not addressed (reason given)

| Area | Status | Evidence / note |
|---|---|---|
| Face threshold configurable, user's 0.45 and jitter 0 preserved | V | `src/settings.py`; defaults unchanged |
| Unknown faces never forced onto nearest student | V | `test_face_matching.py` (unknown, ambiguous, other-class) |
| Ambiguous / duplicate / other-class matches routed to teacher | V (logic) · R (real faces) | per-face statuses in `face_pipeline.py` |
| Threshold *validated* on labelled data | **N** | No real labelled multi-session data exists. `class_data/` is synthetic with near-identical faces (median nearest-other-identity distance 0.22), so it cannot tune a threshold. Framework ready: `scripts/evaluate_faces.py` |
| Margin (0.05 face / 0.05 voice) | R | Provisional defaults, untuned |
| Evaluation framework (identity-split FAR/FRR/precision/recall/coverage) | V | `evaluation/`, 6 tests; run on synthetic set (see README) |
| Group photo: per-face bbox/status, small-face flag, duplicate claims | V (logic, 2 real photos) · R | Small faces now require teacher review instead of automatic attendance; see `face_pipeline.py` |
| Large-image performance | P | Uploads are downscaled to 2000px before storing/processing, reducing session memory; ~0.13 s/image was measured only on 424-px tiles, so classroom-size profiling remains |
| Embedding cache invalidation | P | 30 s TTL + gallery version and `clear_gallery_cache()` after registration, face re-enrolment, and profile deletion; re-enrolment is available to the signed-in student |
| Voice audio quality validation (empty/short/silent/clipped/noisy) | V | `test_voice_logic.py` (numpy only) |
| Voice ambiguity / unknown handling | V (logic) | embeddings are synthetic vectors in tests |
| Voice model accuracy, thresholds, VAD tuning | P | WebRTC VAD + speaker-change splitting tuned on simulated recordings (`scripts/evaluate_voice_group.py`); no real labelled recordings |
| Multimodal face+voice fusion | **N** | Needs paired labelled data to validate; modules kept separate so it can be added |
| Liveness / anti-spoofing (photo, replay) | V (logic, real dlib landmarks on simulated blinks, browser run with a fake camera) · R (real phones, lighting, glasses) | Blink challenge at sign-in/registration/face update: `src/pipelines/liveness.py`, `tests/test_liveness.py`. Does not stop a live deepfake |
| Duplicate attendance prevented (app + DB unique) | V (app) · R (real DB) | `test_attendance_integrity.py`; migration 002 unique index |
| Server-side ownership / enrolment checks | V | same tests |
| Review table, manual vs auto vs corrected provenance | V | tests + browser |
| Audited corrections | P (app path updated; migration not applied/verified here) | migration 008 performs the attendance update and audit insert in one locked database transaction |
| Marks: assessments, CSV validation, atomic import, publish, student view | V (app, fake DB, browser) · R (real DB triggers) | `test_marks.py`; SQL triggers in migration 003 are **not executed** anywhere here |
| Marks match by institutional email | **N** | `students` has no email column; needs your decision (see below) |
| RLS | P | Enabled with no policies (server key only). Per-user policies need Supabase Auth: decision required |
| Logo + favicon + wordmark (light/dark), used in app | V | `assets/brand/` |
| Teacher dashboard: totals, sessions, trend, by-subject, at-risk | V | tests + browser. "Today's %", announcements, upcoming assessments: **N** |
| Student dashboard: attendance, marks, warnings | V | tests. Exam info/announcements: **N** |
| Timetable, announcements/notifications, bulk import UI, QR attendance | **N** | Deferred to keep this change reviewable |
| Biometric consent / deletion / retention workflow | P (implemented; migration 009 not applied or validated against Supabase) | Registration records consent version/time; student deletion removes biometric profile and enrollment, while attendance, marks, and audit rows remain with the identity label “Former student” |
| Concurrency under real load | **N** | Only the DB unique index guards races; not load-tested |


## Hardening pass (branch `security-hardening`)
| Item | Status | Evidence |
|---|---|---|
| "SVM retraining" | **Not applicable**: no classifier exists in this code | original `train_classifier()` cleared all caches and retrained per registration; removed in stage 2 |
| Gallery cache consistent across instances, reruns cheap | V (logic) | `test_recognition_lifecycle.py`; version via `get_gallery_version` |
| Corrupt embedding cannot crash or mislabel recognition | V | same file |
| Duplicate registration blocked server-side incl. race | V (fake DB) · R (real concurrency) | same file |
| Server-side validation (names, subjects, images, audio, CSV, embeddings) | V | `test_validation.py` |
| No half-complete registration | V | same |
| Public teacher/admin self-grant removed; invitations; first-admin setup | V (fake DB, browser) · R (real DB) | `test_account_security.py`, browser run |
| Rate limiting (login, invitation guessing, setup, registration) | V (fake DB) · R | same; per-user not per-IP |
| Key-type guard, RLS/grant migrations, verification SQL | V (static checks) · **R**: SQL and `check_supabase_security.py` never run against a real project | `test_supabase_policy.py` |
| Recognition setting validation | P (implemented; not exercised in this review) | `src/settings.py`, `src/pipelines/face_pipeline.py`, `src/pipelines/voice_pipeline.py` |
| Student-ID checks on student-facing database methods | P (bound to active Streamlit session; not exercised in this review) | `src/database/db.py:_require_student_session` |
| Authorisation matrix (wrong teacher, disabled, forged ids, students) | V | same |
| Authorisation gap found and fixed: `save_attendance_session` skipped the check without `teacher_id`; class-list readers had no owner check | V | same |
| Session expiry / liveness / per-IP limits | P (configurable inactivity expiry implemented; liveness and per-IP controls remain N) | see docs/SECURITY.md |


## Video, voice check and learning (feature/video-learning-voicecheck)
| Item | Status | Evidence |
|---|---|---|
| Video attendance: frames grouped into people, average face matched, 2-frame rule, split-person merge, headcount | V (logic, real dlib on a synthetic pan) · R (real classrooms) | `test_video.py`; `scripts/evaluate_video.py`; browser run |
| Voice check for students the camera missed (1:1 verification, proxy named) | V (logic + real Resemblyzer on synthetic voices) · R (real voices) | `test_voice_check.py`; `scripts/evaluate_voice.py`; browser run with a simulated microphone |
| Whole-class voice identification | Removed from the UI | Took as long as a roll call; replaced by the voice check |
| Learning over time (extra templates from confirmed sessions, drift/novelty/cap rules, reset on re-registration, deleted with profile) | V (fake DB) · R (migration 010 on a real project) | `test_learning.py`; `scripts/evaluate_learning.py`; browser run |


## Live camera, one-recording voice check, blink sign-in, responsive UI
| Item | Status | Evidence |
|---|---|---|
| Live camera attendance (frames in memory, no video file; switch camera; upload clip/photos still available) | V (browser run with a fake camera) · R (real phones) | `src/components/camera_capture*.py`, `video_pipeline.decode_live_frames/analyze_frames`; Playwright run at desktop and phone sizes |
| One recording for all missed students: VAD, speaker-change split, clustering, teacher voice skipped, one voice per student, proxy named | V (logic + real Resemblyzer on synthetic voices; browser run with a fake microphone) · R (real voices) | `tests/test_voice_group.py`; `scripts/evaluate_voice_group.py`: 71/180 answers verified (67/180 with perfectly cut answers), 0/120 absent students verified, 28/60 proxies named |
| Blink liveness: random challenge, nonce + expiry, exact count, prompt timing, same person in every frame (pairwise), template bound to the frames | V (unit + real landmarks + browser) · R | `tests/test_liveness.py`; Playwright: live blink signs in, static photo rejected, new face registers |
| Face sign-in rate limit (per session, looser per network; hashed keys) | V (fake DB) | `tests/test_liveness.py` |
| Responsive layout: review rows, top bar, scrollable navigation, 2-column metrics, dialogs | V (screenshots at 390, 820, 1366 px) | `src/ui/base_layout.py` |
| Display time zone (servers run in UTC) | V | `APP_TIMEZONE`, `services/attendance.parse_ts` |
| Corrections keyed by student id (duplicate names) | V | `tests/test_app_flows.py` |
| Known limitation: a browser refresh signs the user out (Streamlit session) | N | would need cookie-based sessions |
