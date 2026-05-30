import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header {visibility: hidden;}
.stApp {background: #0e1117;}
.hero {
    background: linear-gradient(135deg, #1a2744 0%, #0e1117 60%, #1a1030 100%);
    border: 1px solid #2a3347; border-radius: 20px;
    padding: 48px 40px 36px; margin-bottom: 32px;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(79,142,247,.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.6rem;
    background: linear-gradient(90deg, #4f8ef7, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 8px; line-height: 1.1;
}
.hero-sub { color: #8892a4; font-size: 1.05rem; font-weight: 300; margin: 0; }
.section-label {
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.7rem;
    letter-spacing: 0.14em; text-transform: uppercase; color: #4f8ef7; margin: 28px 0 12px;
}
.card {
    background: #1c2333; border: 1px solid #2a3347;
    border-radius: 14px; padding: 22px 24px; margin-bottom: 16px;
}
.result-wrap {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: linear-gradient(145deg, #1c2745, #12182a);
    border: 2px solid #4f8ef7; border-radius: 20px; padding: 40px 24px; text-align: center;
    box-shadow: 0 0 40px rgba(79,142,247,.15);
}
.result-class { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 4.5rem; line-height: 1; margin-bottom: 8px; }
.result-label { font-size: 1.1rem; font-weight: 500; color: #8892a4; margin-bottom: 20px; }
.result-pill {
    display: inline-block; padding: 6px 18px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
}
.prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.prob-lbl { font-family:'Syne',sans-serif; font-weight:700; font-size:0.85rem; width:24px; color:#e2e8f0; }
.prob-bar-bg { flex:1; height:10px; background:#2a3347; border-radius:999px; overflow:hidden; }
.prob-fill { height:100%; border-radius:999px; }
.prob-pct { font-size:0.8rem; color:#8892a4; width:38px; text-align:right; }
.stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:4px; }
.stat-tile { background:#161b27; border:1px solid #2a3347; border-radius:12px; padding:16px; text-align:center; }
.stat-val { font-family:'Syne',sans-serif; font-weight:700; font-size:1.7rem; color:#4f8ef7; }
.stat-key { font-size:0.72rem; color:#8892a4; margin-top:2px; letter-spacing:.04em; }
.tip-box {
    background: #161b27; border-left: 3px solid #a78bfa;
    border-radius: 0 10px 10px 0; padding: 14px 16px;
    font-size: 0.87rem; color: #e2e8f0; margin-top: 12px; line-height: 1.6;
}
.tip-box b { color:#a78bfa; }
.stButton > button {
    width: 100%; background: linear-gradient(90deg, #4f8ef7, #a78bfa) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    padding: 14px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important; letter-spacing: 0.04em !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load("knn_model.joblib")

model = load_model()

GRADE_OPTIONS   = ["G-02","G-04","G-05","G-06","G-07","G-08","G-09","G-10","G-11","G-12"]
TOPIC_OPTIONS   = ["Arabic","Biology","Chemistry","English","French","Geology","History","IT","Math","Quran","Science","Spanish"]
NATION_OPTIONS  = ["Egypt","Iran","Iraq","Jordan","KW","Lebanon","Libya","Morocco","Palestine","Saudi","Syria","Tunis","USA","Venezuela"]
STAGE_OPTIONS   = ["HighSchool","MiddleSchool","lowerlevel"]
SECTION_OPTIONS = ["A","B","C"]
SEMESTER_OPTIONS= ["F","S"]
RELATION_OPTIONS= ["Father","Mum"]

COLUMN_ORDER = [
    "raisedhands","VisITedResources","AnnouncementsView","Discussion",
    "GradeID_G-04","GradeID_G-05","GradeID_G-06","GradeID_G-07","GradeID_G-08",
    "GradeID_G-09","GradeID_G-10","GradeID_G-11","GradeID_G-12",
    "NationalITy_Iran","NationalITy_Iraq","NationalITy_Jordan","NationalITy_KW",
    "NationalITy_Lebanon","NationalITy_Libya","NationalITy_Morocco",
    "NationalITy_Palestine","NationalITy_Saudi","NationalITy_Syria",
    "NationalITy_Tunis","NationalITy_USA","NationalITy_Venezuela",
    "PlaceofBirth_Iran","PlaceofBirth_Iraq","PlaceofBirth_Jordan",
    "PlaceofBirth_KuwaIT","PlaceofBirth_Lebanon","PlaceofBirth_Libya",
    "PlaceofBirth_Morocco","PlaceofBirth_Palestine","PlaceofBirth_SaudiArabia",
    "PlaceofBirth_Syria","PlaceofBirth_Tunis","PlaceofBirth_USA","PlaceofBirth_venzuela",
    "StageID_MiddleSchool","StageID_lowerlevel",
    "SectionID_B","SectionID_C",
    "Topic_Biology","Topic_Chemistry","Topic_English","Topic_French",
    "Topic_Geology","Topic_History","Topic_IT","Topic_Math",
    "Topic_Quran","Topic_Science","Topic_Spanish",
    "Semester_S","Relation_Mum",
    "ParentAnsweringSurvey_Yes","ParentschoolSatisfaction_Good",
    "StudentAbsenceDays_Under-7","gender_M",
]

STATS = {
    "raisedhands":(46.0,30.8),"VisITedResources":(54.0,33.0),
    "AnnouncementsView":(37.0,26.0),"Discussion":(43.0,27.0),
    "GradeID_G-04":(0.125,0.331),"GradeID_G-05":(0.008,0.088),
    "GradeID_G-06":(0.083,0.277),"GradeID_G-07":(0.263,0.441),
    "GradeID_G-08":(0.302,0.460),"GradeID_G-09":(0.013,0.114),
    "GradeID_G-10":(0.010,0.102),"GradeID_G-11":(0.034,0.182),"GradeID_G-12":(0.029,0.167),
    "NationalITy_Iran":(0.010,0.102),"NationalITy_Iraq":(0.021,0.143),
    "NationalITy_Jordan":(0.438,0.497),"NationalITy_KW":(0.292,0.455),
    "NationalITy_Lebanon":(0.010,0.102),"NationalITy_Libya":(0.008,0.088),
    "NationalITy_Morocco":(0.010,0.102),"NationalITy_Palestine":(0.063,0.242),
    "NationalITy_Saudi":(0.031,0.174),"NationalITy_Syria":(0.031,0.174),
    "NationalITy_Tunis":(0.021,0.143),"NationalITy_USA":(0.013,0.114),"NationalITy_Venezuela":(0.010,0.102),
    "PlaceofBirth_Iran":(0.010,0.102),"PlaceofBirth_Iraq":(0.021,0.143),
    "PlaceofBirth_Jordan":(0.438,0.497),"PlaceofBirth_KuwaIT":(0.292,0.455),
    "PlaceofBirth_Lebanon":(0.010,0.102),"PlaceofBirth_Libya":(0.008,0.088),
    "PlaceofBirth_Morocco":(0.010,0.102),"PlaceofBirth_Palestine":(0.063,0.242),
    "PlaceofBirth_SaudiArabia":(0.031,0.174),"PlaceofBirth_Syria":(0.031,0.174),
    "PlaceofBirth_Tunis":(0.021,0.143),"PlaceofBirth_USA":(0.013,0.114),"PlaceofBirth_venzuela":(0.010,0.102),
    "StageID_MiddleSchool":(0.469,0.500),"StageID_lowerlevel":(0.219,0.414),
    "SectionID_B":(0.313,0.464),"SectionID_C":(0.063,0.242),
    "Topic_Biology":(0.052,0.222),"Topic_Chemistry":(0.052,0.222),
    "Topic_English":(0.063,0.242),"Topic_French":(0.031,0.174),
    "Topic_Geology":(0.052,0.222),"Topic_History":(0.052,0.222),
    "Topic_IT":(0.229,0.421),"Topic_Math":(0.125,0.331),
    "Topic_Quran":(0.042,0.200),"Topic_Science":(0.063,0.242),"Topic_Spanish":(0.052,0.222),
    "Semester_S":(0.479,0.500),"Relation_Mum":(0.323,0.468),
    "ParentAnsweringSurvey_Yes":(0.563,0.497),
    "ParentschoolSatisfaction_Good":(0.594,0.492),
    "StudentAbsenceDays_Under-7":(0.604,0.490),
    "gender_M":(0.667,0.472),
}

def build_feature_vector(inputs):
    row = {}
    row["raisedhands"]       = inputs["raisedhands"]
    row["VisITedResources"]  = inputs["VisITedResources"]
    row["AnnouncementsView"] = inputs["AnnouncementsView"]
    row["Discussion"]        = inputs["Discussion"]
    for g in ["G-04","G-05","G-06","G-07","G-08","G-09","G-10","G-11","G-12"]:
        row[f"GradeID_{g}"] = 1.0 if inputs["GradeID"] == g else 0.0
    for n in ["Iran","Iraq","Jordan","KW","Lebanon","Libya","Morocco","Palestine","Saudi","Syria","Tunis","USA","Venezuela"]:
        row[f"NationalITy_{n}"] = 1.0 if inputs["NationalITy"] == n else 0.0
    for p in ["Iran","Iraq","Jordan","KuwaIT","Lebanon","Libya","Morocco","Palestine","SaudiArabia","Syria","Tunis","USA","venzuela"]:
        row[f"PlaceofBirth_{p}"] = 1.0 if inputs["PlaceofBirth"] == p else 0.0
    for s in ["MiddleSchool","lowerlevel"]:
        row[f"StageID_{s}"] = 1.0 if inputs["StageID"] == s else 0.0
    for s in ["B","C"]:
        row[f"SectionID_{s}"] = 1.0 if inputs["SectionID"] == s else 0.0
    for t in ["Biology","Chemistry","English","French","Geology","History","IT","Math","Quran","Science","Spanish"]:
        row[f"Topic_{t}"] = 1.0 if inputs["Topic"] == t else 0.0
    row["Semester_S"]                   = 1.0 if inputs["Semester"] == "S" else 0.0
    row["Relation_Mum"]                 = 1.0 if inputs["Relation"] == "Mum" else 0.0
    row["ParentAnsweringSurvey_Yes"]     = 1.0 if inputs["ParentAnsweringSurvey"] == "Yes" else 0.0
    row["ParentschoolSatisfaction_Good"] = 1.0 if inputs["ParentschoolSatisfaction"] == "Good" else 0.0
    row["StudentAbsenceDays_Under-7"]   = 1.0 if inputs["StudentAbsenceDays"] == "Under-7" else 0.0
    row["gender_M"]                     = 1.0 if inputs["gender"] == "M" else 0.0
    return row

def scale_features(row_dict):
    arr = []
    for col in COLUMN_ORDER:
        val = row_dict.get(col, 0.0)
        m, s = STATS.get(col, (0.0, 1.0))
        arr.append((val - m) / (s if s > 0 else 1.0))
    return np.array(arr).reshape(1, -1)


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <p class="hero-title">🎓 Student Performance Predictor</p>
  <p class="hero-sub">KNN model · xAPI-Edu dataset · 3-class classification (H / M / L)</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.6, 1], gap="large")

with left:
    st.markdown('<p class="section-label">📊 Engagement Metrics</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        raisedhands   = st.slider("✋ Raised Hands",          0, 100, 45)
        announcements = st.slider("📢 Announcements Viewed",  0, 100, 35)
    with c2:
        visited    = st.slider("📚 Visited Resources",   0, 100, 50)
        discussion = st.slider("💬 Discussion Posts",    0, 100, 40)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-label">🏫 Academic Information</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        stage   = st.selectbox("Stage",    STAGE_OPTIONS,    index=1)
        grade   = st.selectbox("Grade",    GRADE_OPTIONS,    index=6)
        section = st.selectbox("Section",  SECTION_OPTIONS,  index=0)
    with a2:
        topic    = st.selectbox("Topic",    TOPIC_OPTIONS,    index=7)
        semester = st.selectbox("Semester", SEMESTER_OPTIONS, index=0)
        gender   = st.selectbox("Gender",   ["M","F"],        index=0)
    with a3:
        nationality = st.selectbox("Nationality",   NATION_OPTIONS, index=3)
        place_birth = st.selectbox("Place of Birth",
                                   ["Egypt","Iran","Iraq","Jordan","KuwaIT","Lebanon",
                                    "Libya","Morocco","Palestine","SaudiArabia","Syria",
                                    "Tunis","USA","venzuela"], index=4)
        relation = st.selectbox("Parent Relation", RELATION_OPTIONS, index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-label">👨‍👩‍👧 Parental Involvement</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        parent_survey = st.selectbox("Answered Survey",      ["Yes","No"], index=0)
    with p2:
        parent_sat    = st.selectbox("School Satisfaction",  ["Good","Bad"], index=0)
    with p3:
        absence       = st.selectbox("Absence Days",         ["Under-7","Above-7"], index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  Predict Performance Class")

with right:
    avg_engage = (raisedhands + visited + announcements + discussion) / 4
    st.markdown('<p class="section-label">📈 Input Summary</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-tile"><div class="stat-val">{avg_engage:.0f}</div><div class="stat-key">Avg Engagement</div></div>
      <div class="stat-tile"><div class="stat-val">{raisedhands + visited}</div><div class="stat-key">Active Score</div></div>
      <div class="stat-tile"><div class="stat-val">{announcements + discussion}</div><div class="stat-key">Social Score</div></div>
      <div class="stat-tile"><div class="stat-val">{'✅' if absence == 'Under-7' else '⚠️'}</div><div class="stat-key">Attendance</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label" style="margin-top:28px">🔮 Prediction</p>', unsafe_allow_html=True)

    if predict_btn:
        inputs = {
            "raisedhands": raisedhands, "VisITedResources": visited,
            "AnnouncementsView": announcements, "Discussion": discussion,
            "GradeID": grade, "NationalITy": nationality, "PlaceofBirth": place_birth,
            "StageID": stage, "SectionID": section, "Topic": topic,
            "Semester": semester, "Relation": relation,
            "ParentAnsweringSurvey": parent_survey, "ParentschoolSatisfaction": parent_sat,
            "StudentAbsenceDays": absence, "gender": gender,
        }
        row  = build_feature_vector(inputs)
        X    = scale_features(row)
        pred  = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        CLASS_META = {
            "H": {"label": "High Performer",   "color": "#34d399", "pill_bg": "#064e3b"},
            "M": {"label": "Mid-Level Learner", "color": "#f4c542", "pill_bg": "#451a03"},
            "L": {"label": "Needs Support",     "color": "#f87171", "pill_bg": "#450a0a"},
        }
        meta = CLASS_META[pred]

        st.markdown(f"""
        <div class="result-wrap">
          <div class="result-class" style="color:{meta['color']}">{pred}</div>
          <div class="result-label">{meta['label']}</div>
          <span class="result-pill" style="background:{meta['pill_bg']};color:{meta['color']}">Predicted Class</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-label" style="margin-top:20px">Confidence</p>', unsafe_allow_html=True)
        BAR_COLORS = {"H":"#34d399","M":"#f4c542","L":"#f87171"}
        for cls, prob in zip(model.classes_, proba):
            pct = int(prob * 100)
            st.markdown(f"""
            <div class="prob-row">
              <span class="prob-lbl">{cls}</span>
              <div class="prob-bar-bg"><div class="prob-fill" style="width:{pct}%;background:{BAR_COLORS[cls]}"></div></div>
              <span class="prob-pct">{pct}%</span>
            </div>
            """, unsafe_allow_html=True)

        tips = {
            "H": "🌟 <b>Excellent!</b> Strong engagement across all dimensions. Keep challenging with advanced material.",
            "M": "📘 <b>On track.</b> Encouraging more resource visits and discussion could push this student to High.",
            "L": "🚨 <b>Attention needed.</b> Early intervention on attendance and resource access is recommended.",
        }
        st.markdown(f'<div class="tip-box">{tips[pred]}</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#161b27;border:1px dashed #2a3347;border-radius:14px;
                    padding:40px 24px;text-align:center;color:#8892a4;font-size:0.9rem;line-height:1.7;">
          <div style="font-size:2.5rem;margin-bottom:12px">🎯</div>
          Configure the student profile on the left,<br>
          then click <b style="color:#4f8ef7">Predict Performance Class</b><br>to see the result.
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<hr style="border-color:#2a3347;margin-top:36px">
<p style="text-align:center;color:#8892a4;font-size:0.75rem;margin-top:12px">
  KNN (k=19) · xAPI-Edu-Data · Classes: <b style="color:#34d399">H</b> High &nbsp;|&nbsp;
  <b style="color:#f4c542">M</b> Middle &nbsp;|&nbsp; <b style="color:#f87171">L</b> Low
</p>
""", unsafe_allow_html=True)