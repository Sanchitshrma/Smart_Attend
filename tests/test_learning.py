"""Learning over time: extra face templates from teacher-confirmed sessions, with safety rules."""
import numpy as np
import pytest

from src.database import db
from src.pipelines.face_pipeline import MATCHED, FaceMatch, PhotoResult, build_gallery, match_encodings
from src.services import biometrics
from src.services.attendance import confirmed_learning_samples, learnable_samples
from tests.helpers import fake_face

rng = np.random.default_rng(8)


def toward(a, b, d):
    """Point at distance d from a, in the direction of b."""
    a, b = np.asarray(a), np.asarray(b)
    return (a + (b - a) / np.linalg.norm(b - a) * d).tolist()


def test_template_rules_drift_redundancy_and_cap():
    enrol = np.asarray(fake_face(1))
    other = fake_face(2)
    kw = dict(min_novelty=0.12, max_from_enrolment=0.50, cap=2)
    assert biometrics.plan_template_update(toward(enrol, other, 0.3), enrol, [], **kw) == (True, [], "added")
    assert biometrics.plan_template_update(toward(enrol, other, 0.7), enrol, [], **kw)[0] is False      # drift
    assert biometrics.plan_template_update(toward(enrol, other, 0.05), enrol, [], **kw)[0] is False     # nothing new
    learned = [(11, toward(enrol, fake_face(3), 0.3)), (12, toward(enrol, fake_face(4), 0.3))]
    add, drop, _ = biometrics.plan_template_update(toward(enrol, fake_face(5), 0.3), enrol, learned, **kw)
    assert add and drop == [11]                                                                          # oldest dropped


def test_matching_uses_each_students_closest_template_and_counts_students_not_rows():
    rows = [{"student_id": 1, "face_embedding": fake_face(1)}, {"student_id": 2, "face_embedding": fake_face(2)},
            {"student_id": 1, "face_embedding": toward(fake_face(1), fake_face(7), 0.5)}]   # learned: new glasses
    ids, mat = build_gallery(rows)
    probe = toward(toward(fake_face(1), fake_face(7), 0.5), fake_face(8), 0.1)       # today's photo, glasses on
    plain = match_encodings([probe], [1, 2], np.asarray([fake_face(1), fake_face(2)]))
    learned = match_encodings([probe], ids, mat)
    assert plain.faces[0].status != MATCHED                                            # missed without learning
    assert learned.faces[0].status == MATCHED and learned.faces[0].student_id == 1
    assert learned.faces[0].second_distance > 1.0                                      # runner-up is student 2, not student 1 again


def test_learnable_and_confirmed_samples_only_take_confident_kept_matches():
    emb = np.asarray(fake_face(1))
    res = PhotoResult(3, [FaceMatch(0, status="matched", student_id=1, distance=0.3, embedding=emb),
                          FaceMatch(1, status="matched", student_id=2, distance=0.2, embedding=emb, flags=["small_face"]),
                          FaceMatch(2, status="ambiguous", candidates=[3], embedding=emb)])
    samples = learnable_samples([res])
    assert [s["student_id"] for s in samples] == [1]
    logs = [{"student_id": 1, "is_present": True, "source": "auto", "method": "face"}]
    assert confirmed_learning_samples(samples, logs) == samples
    for changed in ({"source": "manual"}, {"is_present": False}, {"method": "voice"}):
        assert confirmed_learning_samples(samples, [{**logs[0], **changed}]) == []


@pytest.fixture()
def world(fake_db):
    db.create_teacher("tee1", "password-1", "Tee One"); db.create_teacher("tee2", "password-2", "Tee Two")
    t1, t2 = db.teacher_login("tee1", "password-1")["teacher_id"], db.teacher_login("tee2", "password-2")["teacher_id"]
    sub = db.create_subject("ECE401", "DSP", "A", t1)[0]["subject_id"]
    s1 = db.create_student("Aarav Sharma", fake_face(1))[0]["student_id"]
    s2 = db.create_student("Diya Kapoor", fake_face(2))[0]["student_id"]
    db._run(db._t("subject_students").insert({"student_id": s1, "subject_id": sub}), "seed")
    return fake_db, t1, t2, sub, s1, s2


def test_confirmed_samples_are_stored_with_server_side_rules(world, monkeypatch):
    fake, t1, t2, sub, s1, s2 = world
    good = toward(fake_face(1), fake_face(7), 0.3)
    version = db.get_gallery_version()
    assert db.add_face_templates(t1, sub, [{"student_id": s1, "embedding": good}]) == 1
    assert db.get_gallery_version() != version                                     # every instance reloads the gallery
    assert len([r for r in db.get_face_gallery() if r["student_id"] == s1]) == 2
    bad = [{"student_id": s1, "embedding": good},                                   # duplicate of the one just learned
           {"student_id": s1, "embedding": toward(fake_face(1), fake_face(7), 0.8)},  # drifted too far
           {"student_id": s2, "embedding": toward(fake_face(2), fake_face(7), 0.3)},  # not enrolled in this subject
           {"student_id": s1, "embedding": [float("nan")] * 128}]
    assert db.add_face_templates(t1, sub, bad) == 0
    with pytest.raises(db.PermissionDenied):
        db.add_face_templates(t2, sub, [{"student_id": s1, "embedding": good}])     # not this teacher's subject
    monkeypatch.setattr(db.cfg, "FACE_MAX_TEMPLATES", 2)
    for k in (8, 9, 10):
        db.add_face_templates(t1, sub, [{"student_id": s1, "embedding": toward(fake_face(1), fake_face(k), 0.3)}])
    assert len(fake.tables["face_templates"]) == 2                                 # oldest dropped
    monkeypatch.setattr(db.cfg, "FACE_LEARNING", 0)
    assert db.add_face_templates(t1, sub, [{"student_id": s1, "embedding": toward(fake_face(1), fake_face(11), 0.3)}]) == 0


def test_reregistering_or_deleting_a_face_clears_learned_templates(world, monkeypatch):
    fake, t1, _, sub, s1, _ = world
    db.add_face_templates(t1, sub, [{"student_id": s1, "embedding": toward(fake_face(1), fake_face(7), 0.3)}])
    monkeypatch.setattr(db, "_require_student_session", lambda sid: None)
    db.update_student_face_embedding(s1, toward(fake_face(1), fake_face(3), 0.2))
    assert fake.tables["face_templates"] == []
    db.add_face_templates(t1, sub, [{"student_id": s1, "embedding": toward(fake_face(1), fake_face(7), 0.3)}])
    db.delete_student_profile(s1)
    assert fake.tables["face_templates"] == []


def test_without_migration_010_learning_is_off_but_attendance_works(world):
    fake, t1, _, sub, s1, _ = world
    fake.missing_tables.add("face_templates")
    assert db.add_face_templates(t1, sub, [{"student_id": s1, "embedding": toward(fake_face(1), fake_face(7), 0.3)}]) == 0
    assert {r["student_id"] for r in db.get_face_gallery()} == {1, 2}
    assert db.get_gallery_version()
