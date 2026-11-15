"""Learning over time on simulated weeks of changing appearance (real dlib models).

    python scripts/evaluate_learning.py

Seven distinct synthetic students attend three lectures a week for six weeks while lighting gets
darker, the camera softer and a fringe appears. Each confirmed match may become a template under
the same rules as the app (biometrics.plan_template_update). Synthetic faces: compare, do not
read as real-world accuracy.
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from src.pipelines.face_pipeline import get_face_embeddings, build_gallery, match_encodings, MATCHED
from src.services.biometrics import plan_template_update
from src.settings import FACE_LEARN_MIN_NOVELTY, FACE_LEARN_MAX_FROM_ENROLMENT, FACE_MAX_TEMPLATES
R=str(ROOT / "class_data" / "photos") + "/"
FILES=["student_02.jpg","student_24.jpg","student_44.jpg","student_27.jpg","student_41.jpg","student_51.jpg","student_16.jpg"]
rng=np.random.default_rng(12)
enrol={k:get_face_embeddings(R+f)[0] for k,f in enumerate(FILES)}
def week_photo(k, w, take):
    """Week w: lighting gets darker and warmer, the camera softer, a 'new hairstyle' band appears."""
    im=Image.open(R+FILES[k]).convert("RGB")
    sev=w/6
    im=ImageEnhance.Brightness(im).enhance(1.0-0.5*sev+float(take.uniform(-0.05,0.05)))
    im=ImageEnhance.Color(im).enhance(1.0+0.6*sev)
    if w>=3:  # fringe covering the forehead
        d=im.copy(); px=d.load(); W,H=d.size
        for y in range(int(H*0.18), int(H*(0.18+0.05*sev))):
            for x in range(int(W*0.2), int(W*0.8)): px[x,y]=(40,30,25)
        im=d
    im=im.filter(ImageFilter.GaussianBlur(0.6+1.6*sev))
    im=im.rotate(float(take.uniform(-6,6)), fillcolor=(0,0,0))
    return im
learned={k:[] for k in enrol}; tid=0
rows=[]
for w in range(1,7):
    plain_ok=learn_ok=wrong_plain=wrong_learn=0; n=0; added=0
    gal_plain=build_gallery([{"student_id":k,"face_embedding":enrol[k].tolist()} for k in enrol])
    gal_learn=build_gallery([{"student_id":k,"face_embedding":enrol[k].tolist()} for k in enrol]+
                            [{"student_id":k,"face_embedding":e} for k,ts in learned.items() for _,e in ts])
    for k in enrol:
        for t in range(3):  # three lectures per week
            e=get_face_embeddings(week_photo(k,w,rng))
            if not e: continue
            n+=1; e=e[0]
            a=match_encodings([e],*gal_plain).faces[0]; b=match_encodings([e],*gal_learn).faces[0]
            plain_ok+= a.status==MATCHED and a.student_id==k; wrong_plain+= a.status==MATCHED and a.student_id!=k
            learn_ok+= b.status==MATCHED and b.student_id==k; wrong_learn+= b.status==MATCHED and b.student_id!=k
            if b.status==MATCHED and b.student_id==k:  # teacher confirms -> learn
                add,drop,_=plan_template_update(e, enrol[k], learned[k], min_novelty=FACE_LEARN_MIN_NOVELTY,
                    max_from_enrolment=FACE_LEARN_MAX_FROM_ENROLMENT, cap=FACE_MAX_TEMPLATES)
                if add:
                    tid+=1; learned[k]=[x for x in learned[k] if x[0] not in drop]+[(tid,e.tolist())]; added+=1
    rows.append(dict(week=w, n=n, without_learning=plain_ok, with_learning=learn_ok, wrong_without=wrong_plain, wrong_with=wrong_learn, templates_added=added))
    print(rows[-1], flush=True)
print(json.dumps(rows, indent=1))
