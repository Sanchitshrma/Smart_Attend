"""Video attendance: frame reading, grouping faces into people, and the 'seen in 2+ frames' rule."""
from pathlib import Path

import numpy as np
import pytest

from src.pipelines.face_pipeline import AMBIGUOUS, MATCHED, build_gallery
from src.pipelines.video_pipeline import VideoError, group_sightings, match_people, read_frames
from tests.helpers import fake_face

rng = np.random.default_rng(3)
ROOT = Path(__file__).resolve().parents[1]


def near(v, d=0.12):
    r = rng.normal(size=128)
    return np.asarray(v) + r / np.linalg.norm(r) * d


def gallery(n=4):
    rows = [{"student_id": i, "face_embedding": fake_face(i)} for i in range(1, n + 1)]
    return build_gallery(rows)


def test_faces_are_grouped_into_people_across_frames():
    a, b, c = (np.asarray(fake_face(i)) for i in (1, 2, 3))
    frames = [
        [((0, 0, 80, 80), near(a)), ((100, 0, 180, 80), near(b))],
        [((0, 0, 80, 80), near(a)), ((100, 0, 180, 80), near(b)), ((200, 0, 280, 80), near(c))],
        [((0, 0, 80, 80), near(a))],                                  # b and c out of view
        [((100, 0, 180, 80), near(b)), ((200, 0, 280, 80), near(c))],
    ]
    people = group_sightings(frames)
    assert len(people) == 3
    assert sorted(len(p.frames) for p in people) == [2, 3, 3]


def test_two_faces_in_the_same_frame_are_never_one_person():
    a = np.asarray(fake_face(1))
    twin = near(a, 0.05)                                               # looks almost identical
    people = group_sightings([[((0, 0, 80, 80), a), ((100, 0, 180, 80), twin)]])
    assert len(people) == 2


def test_people_seen_once_go_to_review_and_averages_are_matched():
    ids, mat = gallery()
    a, b = np.asarray(fake_face(1)), np.asarray(fake_face(2))
    frames = [[((0, 0, 80, 80), near(a)), ((100, 0, 180, 80), near(b))], [((0, 0, 80, 80), near(a))]]
    res = match_people(group_sightings(frames), ids, mat)
    by_student = {f.candidates[0]: f for f in res.faces}
    assert by_student[1].status == MATCHED and by_student[1].sightings == 2
    assert by_student[2].status == AMBIGUOUS and "seen_once" in by_student[2].flags


def test_one_student_split_into_two_groups_is_merged_not_flagged():
    ids, mat = gallery()
    a = np.asarray(fake_face(1))
    side = a + 0.42 * (np.asarray(fake_face(9)) - a) / np.linalg.norm(np.asarray(fake_face(9)) - a)  # head turned
    frames = [[((0, 0, 80, 80), near(a, 0.05))], [((0, 0, 80, 80), near(a, 0.05))],
              [((0, 0, 80, 80), near(side, 0.02))], [((0, 0, 80, 80), near(side, 0.02))]]
    people = group_sightings(frames)
    assert len(people) == 2                                            # grouping alone splits them
    res = match_people(people, ids, mat)
    assert len(res.faces) == 1 and res.faces[0].status == MATCHED and res.faces[0].sightings == 4


def _write_video(path, frames, fps=10):
    import cv2
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    return Path(path).read_bytes()


def test_read_frames_samples_validates_and_rejects_bad_files(tmp_path):
    pytest.importorskip("cv2")
    frames = [np.full((120, 160, 3), k * 8, dtype=np.uint8) for k in range(30)]   # 3 s at 10 fps
    data = _write_video(tmp_path / "clip.mp4", frames)
    got, seconds = read_frames(data, fps=2)
    assert 5 <= len(got) <= 7 and abs(seconds - 3.0) < 0.2 and got[0].shape == (120, 160, 3)
    with pytest.raises(VideoError, match="keep it under"):
        read_frames(data, fps=2, max_seconds=1)
    with pytest.raises(VideoError):
        read_frames(b"definitely not a video")
    with pytest.raises(VideoError, match="empty"):
        read_frames(b"")


def test_real_video_end_to_end(tmp_path, fake_db):
    """Real dlib models on a synthetic pan across three enrolled students."""
    pytest.importorskip("cv2")
    pytest.importorskip("dlib")
    from PIL import Image
    from src.database import db
    from src.pipelines.face_pipeline import get_face_embeddings
    from src.pipelines.video_pipeline import analyze_video

    photos = ROOT / "class_data" / "photos"
    files = ["student_05.jpg", "student_24.jpg", "student_19.jpg"]
    tiles = [np.asarray(Image.open(photos / f).convert("RGB").resize((220, 220))) for f in files]
    canvas = np.full((300, 900, 3), 235, dtype=np.uint8)
    for k, t in enumerate(tiles):
        canvas[40:260, 40 + k * 290:260 + k * 290] = t
    frames = [np.ascontiguousarray(canvas[:, x:x + 640]) for x in np.linspace(0, 260, 16).astype(int)]  # camera pan
    data = _write_video(tmp_path / "pan.mp4", frames, fps=8)
    ids = [db.create_student(n, get_face_embeddings(str(photos / f))[0].tolist())[0]["student_id"]
           for n, f in zip(["Aarav Sharma", "Ananya Verma", "Rohan Gupta"], files)]
    res = analyze_video(data, allowed_ids=set(ids))
    matched = {f.student_id for f in res.photo.faces if f.status == MATCHED}
    assert matched == set(ids) and res.headcount == 3
