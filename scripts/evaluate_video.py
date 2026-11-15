"""Photo mode vs video mode on simulated classroom pans (real dlib models).

    python scripts/evaluate_video.py

Twelve simulated classrooms of 7 distinct synthetic students (back row smaller), with random head
turns, occlusion and blur per frame. Compares one wide photo, three photos and a 16-frame video pan.
Synthetic faces: use the numbers to compare modes, not as real-world accuracy.
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from src.pipelines.face_pipeline import get_face_embeddings, detect_faces, build_gallery, match_encodings, MATCHED
from src.pipelines.video_pipeline import group_sightings, match_people
R=str(ROOT / "class_data" / "photos") + "/"
FILES=["student_02.jpg","student_24.jpg","student_44.jpg","student_27.jpg","student_41.jpg","student_51.jpg","student_16.jpg"]
emb={k:get_face_embeddings(R+f)[0] for k,f in enumerate(FILES)}
ids, mat = build_gallery([{"student_id":k,"face_embedding":emb[k].tolist()} for k in emb])
rng=np.random.default_rng(4)
def scene():
    """Wide classroom canvas: 7 students, back row smaller, random order; per-frame occlusion/turn/blur."""
    order=list(rng.permutation(7)); W,H=1700,560; canvas=Image.new("RGB",(W,H),(205,200,190))
    slots=[]
    for k,sid in enumerate(order):
        back = k>=4
        size = int(rng.uniform(70,95)) if back else int(rng.uniform(130,170))
        x = 60+ (k%4)*400 + (200 if back else 0) + int(rng.uniform(-30,30)); y = 60 if back else 270
        slots.append((sid,x,y,size))
    return order, canvas, slots
def render(canvas, slots, frame_rng, cond):
    img=canvas.copy()
    for sid,x,y,size in slots:
        t=Image.open(R+FILES[sid]).convert("RGB")
        if frame_rng.random()<cond["turn"]: t=t.rotate(float(frame_rng.choice([-35,35])), fillcolor=(205,200,190))
        t=t.resize((size,size))
        img.paste(t,(x,y))
        if frame_rng.random()<cond["occlude"]:  # someone leans in front: cover lower 60% of the face
            img.paste((60,60,70),(x, y+int(size*0.35), x+size, y+size))
    img=ImageEnhance.Brightness(img).enhance(float(frame_rng.uniform(0.6,1.1)))
    if frame_rng.random()<cond["blur"]: img=img.filter(ImageFilter.GaussianBlur(float(frame_rng.uniform(1.5,2.5))))
    return img
def frames_for(canvas, slots, n=16, cond=None):
    fr=np.random.default_rng(int(rng.integers(1e9))); W=canvas.size[0]; out=[]
    for x0 in np.linspace(0, W-960, n).astype(int):
        out.append(render(canvas,slots,fr,cond).crop((x0,0,x0+960,560)))
    return out
cond=dict(turn=0.25, occlude=0.25, blur=0.25)
tot={"single_photo":{"correct":0,"wrong":0,"review":0},"three_photos":{"correct":0,"wrong":0,"review":0},"video":{"correct":0,"wrong":0,"review":0}}
N=12
for s in range(N):
    order, canvas, slots = scene(); present=set(order)
    fr=np.random.default_rng(s+100)
    # single photo: the whole room in one shot (wide photo, downscaled like a phone photo of the room)
    whole=render(canvas,slots,fr,cond)
    found=detect_faces(whole); res=match_encodings([e for _,e in found], ids, mat, bboxes=[b for b,_ in found])
    m={f.student_id for f in res.faces if f.status==MATCHED}
    tot["single_photo"]["correct"]+=len(m&present); tot["single_photo"]["wrong"]+=len(m-present); tot["single_photo"]["review"]+=sum(f.status in("ambiguous","duplicate") for f in res.faces)
    # three photos (left / middle / right)
    m=set(); rv=0
    for x0 in (0,370,740):
        img=render(canvas,slots,fr,cond).crop((x0,0,x0+960,560)); found=detect_faces(img)
        r=match_encodings([e for _,e in found], ids, mat, bboxes=[b for b,_ in found]); m|={f.student_id for f in r.faces if f.status==MATCHED}; rv+=sum(f.status in("ambiguous","duplicate") for f in r.faces)
    tot["three_photos"]["correct"]+=len(m&present); tot["three_photos"]["wrong"]+=len(m-present); tot["three_photos"]["review"]+=rv
    # video: 16 frames of a pan
    per=[detect_faces(f) for f in frames_for(canvas,slots,16,cond)]
    vr=match_people(group_sightings(per), ids, mat)
    m={f.student_id for f in vr.faces if f.status==MATCHED}
    tot["video"]["correct"]+=len(m&present); tot["video"]["wrong"]+=len(m-present); tot["video"]["review"]+=sum(f.status in("ambiguous","duplicate") for f in vr.faces)
    print(s, {k:v["correct"] for k,v in tot.items()}, flush=True)
tot["students"]=7*N
print(json.dumps(tot)); 
