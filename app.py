import os
import streamlit as st


import pandas as pd
import joblib

# ==================================================================
#  Session State Initialization
# ==================================================================

if "page" not in st.session_state:
    st.session_state["page"] = "🏠 Home"

if "subject_percentages" not in st.session_state:
    st.session_state["subject_percentages"] = {}

if "interest_scores" not in st.session_state:
    st.session_state["interest_scores"] = {}

if "skill_scores" not in st.session_state:
    st.session_state["skill_scores"] = {}

if "career_preferences" not in st.session_state:
    st.session_state["career_preferences"] = {}

if "final_matric_percentage" not in st.session_state:
    st.session_state["final_matric_percentage"] = 0.0

if "final_inter_percentage" not in st.session_state:
    st.session_state["final_inter_percentage"] = 0.0

if "final_intermediate_group" not in st.session_state:
    st.session_state["final_intermediate_group"] = "Pre-Medical"

# ==================================================================
#  Model & Data
# ==================================================================

@st.cache_resource
def load_model_files():
    model = joblib.load("bachelor_degree_recommendation_model_compressed.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    model_columns = joblib.load("model_columns.joblib")
    return model, label_encoder, model_columns

model, label_encoder, model_columns = load_model_files()

SUBJECTS = {
    "Pre-Medical": ["English", "Physics", "Chemistry", "Biology", "Urdu"],
    "Pre-Engineering": ["English", "Physics", "Chemistry", "Mathematics", "Urdu"],
    "ICS": ["English", "Physics", "Computer_Science", "Mathematics", "Urdu"],
    "Commerce": ["English", "Accounting", "Economics", "Business_Mathematics", "Urdu"],
    "Arts": ["English", "Urdu", "Civics", "History"],
}

PAGES = [
    "🏠 Home",
    "📚 Academic Information",
    "❤️ Interests",
    "💡 Skills",
    "💼 Career Preferences",
    "🎯 Recommendation",
    "ℹ️ About",
]

STEP_FLOW = [
    ("📚", "Academic", "📚 Academic Information"),
    ("❤️", "Interests", "❤️ Interests"),
    ("💡", "Skills", "💡 Skills"),
    ("💼", "Career", "💼 Career Preferences"),
    ("🎯", "Result", "🎯 Recommendation"),
]

# ==================================================================
#  Page Configuration
# ==================================================================

st.set_page_config(
    page_title="Degree Compass — AI Bachelor Degree Recommendation",
    page_icon="🎓",
    layout="wide",
)

# ==================================================================
#  Styling
# ==================================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Manrope:wght@400;500;700;800&display=swap');

:root{
    --ink:#101A33;
    --ink-soft:#4A5570;
    --paper:#F3F6FC;
    --card:#FFFFFF;
    --gold:#C9971E;
    --gold-soft:#E9C468;
    --emerald:#1E6F5C;
    --line:#E1E6F2;
}

html, body, [class*="css"]{
    font-family:'Manrope', sans-serif;
    color:var(--ink);
}

section[data-testid="stAppViewContainer"]{
    background:var(--paper);
}

h1, h2, h3{
    font-family:'Fraunces', serif !important;
    color:var(--ink) !important;
    letter-spacing:-0.01em;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg, #101A33 0%, #17244A 100%);
}
section[data-testid="stSidebar"] *{
    color:#EDF1FA !important;
}
section[data-testid="stSidebar"] .stButton>button{
    background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.14);
    border-radius:10px;
    text-align:left;
    font-weight:600;
    transition:all .2s ease;
}
section[data-testid="stSidebar"] .stButton>button:hover{
    background:rgba(201,151,30,0.18);
    border-color:var(--gold-soft);
    transform:translateX(2px);
}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{
    background:linear-gradient(135deg, var(--gold), var(--gold-soft));
    border:none;
    color:#101A33 !important;
    box-shadow:0 4px 14px rgba(201,151,30,0.35);
}
section[data-testid="stSidebar"] div[data-testid="stExpander"]{
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.1);
    border-radius:12px;
}

/* ---------- Buttons (main area) ---------- */
.stButton>button{
    border-radius:10px;
    font-weight:700;
    padding:0.55rem 1.2rem;
    border:1px solid var(--line);
    transition:all .2s ease;
}
.stButton>button:hover{
    transform:translateY(-2px);
    box-shadow:0 8px 18px rgba(16,26,51,0.12);
}
div[data-testid="stAppViewContainer"] .stButton>button[kind="primary"]{
    background:linear-gradient(135deg, var(--gold), var(--gold-soft));
    border:none;
    color:#101A33;
}

/* ---------- Inputs ---------- */
.stNumberInput input, .stTextInput input{
    border-radius:10px !important;
    border:1px solid var(--line) !important;
}
div[data-baseweb="select"]>div{
    border-radius:10px !important;
    border-color:var(--line) !important;
}

/* ---------- Radio pills ---------- */
div[role="radiogroup"]{
    display:flex;
    flex-wrap:wrap;
    gap:.5rem;
    margin-top:.25rem;
}
div[role="radiogroup"] label{
    background:var(--card);
    border:1px solid var(--line);
    border-radius:999px;
    padding:.3rem 1rem;
    transition:all .2s ease;
}
div[role="radiogroup"] label:hover{
    border-color:var(--gold-soft);
}

/* ---------- Dividers ---------- */
hr{
    border:none;
    height:2px;
    background:linear-gradient(90deg, var(--gold), transparent);
    margin:1.4rem 0;
}

/* ---------- Cards ---------- */
.card{
    background:var(--card);
    border:1px solid var(--line);
    border-radius:16px;
    padding:1.4rem 1.2rem;
    transition:all .25s ease;
}
.card:hover{
    transform:translateY(-4px);
    box-shadow:0 14px 28px rgba(16,26,51,0.10);
    border-color:var(--gold-soft);
}

/* ---------- Hero ---------- */
.hero{
    background:linear-gradient(135deg, #101A33 0%, #1B2A55 60%, #22366B 100%);
    border-radius:22px;
    padding:3rem 2.6rem;
    color:#F3F6FC;
    position:relative;
    overflow:hidden;
    margin-bottom:1.4rem;
}
.hero::after{
    content:"";
    position:absolute;
    right:-60px; top:-60px;
    width:220px; height:220px;
    border-radius:50%;
    background:radial-gradient(circle, rgba(201,151,30,0.35), transparent 70%);
}
.hero h1{
    color:#FFFFFF !important;
    font-size:2.6rem;
    margin:0 0 .6rem 0;
}
.hero .accent{ color:var(--gold-soft); }
.hero p{
    color:#C9D2E8;
    font-size:1.05rem;
    max-width:640px;
    margin-bottom:1.4rem;
}
.eyebrow{
    display:inline-block;
    font-family:'Manrope',sans-serif;
    font-weight:800;
    letter-spacing:.12em;
    text-transform:uppercase;
    font-size:.72rem;
    color:var(--gold-soft);
    background:rgba(201,151,30,0.14);
    padding:.3rem .8rem;
    border-radius:999px;
    margin-bottom:.9rem;
}

/* ---------- Stat chips ---------- */
.stat-card{
    background:var(--card);
    border:1px solid var(--line);
    border-radius:14px;
    padding:1rem 1.2rem;
    text-align:center;
}
.stat-card .num{
    font-family:'Fraunces', serif;
    font-size:1.8rem;
    font-weight:700;
    color:var(--ink);
}
.stat-card .lbl{
    font-size:.78rem;
    color:var(--ink-soft);
    text-transform:uppercase;
    letter-spacing:.06em;
    font-weight:700;
}

/* ---------- How it works ---------- */
.step-card{
    background:var(--card);
    border:1px solid var(--line);
    border-radius:16px;
    padding:1.2rem 1rem;
    text-align:left;
    height:100%;
    transition:all .25s ease;
}
.step-card:hover{
    transform:translateY(-4px);
    box-shadow:0 14px 28px rgba(16,26,51,0.10);
    border-color:var(--gold-soft);
}
.step-card .idx{
    font-family:'Fraunces', serif;
    font-size:.85rem;
    font-weight:700;
    color:var(--gold);
}
.step-card .icon{ font-size:1.6rem; margin:.3rem 0 .5rem 0; }
.step-card .title{ font-weight:800; margin-bottom:.25rem; }
.step-card .desc{ font-size:.85rem; color:var(--ink-soft); }

/* ---------- Stepper ---------- */
.stepper{
    display:flex;
    align-items:flex-start;
    justify-content:center;
    margin:0 0 1.6rem 0;
    padding:.6rem 0;
    flex-wrap:wrap;
}
.step{
    display:flex;
    flex-direction:column;
    align-items:center;
    gap:.35rem;
    min-width:70px;
}
.step-circle{
    width:44px; height:44px;
    border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:1.1rem;
    background:#fff;
    border:2px solid var(--line);
    color:var(--ink-soft);
    transition:all .25s ease;
}
.step.active .step-circle{
    background:linear-gradient(135deg, var(--gold), var(--gold-soft));
    border-color:var(--gold);
    color:#fff;
    box-shadow:0 6px 16px rgba(201,151,30,0.35);
    transform:scale(1.08);
}
.step.done .step-circle{
    background:var(--emerald);
    border-color:var(--emerald);
    color:#fff;
}
.step-label{
    font-size:.7rem;
    font-weight:800;
    letter-spacing:.03em;
    text-transform:uppercase;
    color:var(--ink-soft);
}
.step.active .step-label{ color:var(--ink); }
.step-line{
    flex:0 0 36px;
    height:2px;
    background:var(--line);
    margin:22px 4px 0 4px;
}
.step-line.done{ background:var(--emerald); }

/* ---------- Recommendation cards ---------- */
.rec-card{
    background:var(--card);
    border:1px solid var(--line);
    border-radius:18px;
    padding:1.4rem 1.5rem;
    margin-bottom:1rem;
}
.rec-card .rank{
    font-family:'Fraunces', serif;
    font-size:1.1rem;
    font-weight:700;
    color:var(--ink-soft);
    margin-bottom:.2rem;
}
.rec-card .dept{
    font-family:'Fraunces', serif;
    font-size:1.5rem;
    font-weight:700;
    color:var(--ink);
    margin-bottom:.6rem;
}
.bar-track{
    width:100%; height:12px;
    background:#EEF1F9;
    border-radius:999px;
    overflow:hidden;
}
.bar-fill{
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg, var(--gold), var(--gold-soft));
}
.conf-label{
    font-size:.85rem;
    color:var(--ink-soft);
    margin-top:.4rem;
    font-weight:700;
}

.img-fallback{
    font-size:5rem;
    text-align:center;
    background:var(--card);
    border:1px solid var(--line);
    border-radius:18px;
    padding:2.4rem 0;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ==================================================================
#  Helpers
# ==================================================================

def goto(page_name):
    st.session_state["page"] = page_name
    st.rerun()


def safe_image(path, **kwargs):
    if os.path.exists(path):
        st.image(path, **kwargs)
    else:
        st.markdown('<div class="img-fallback">🎓</div>', unsafe_allow_html=True)


def render_stepper(current_page):
    idx = next((i for i, (_, _, p) in enumerate(STEP_FLOW) if p == current_page), -1)
    if idx == -1:
        return
    parts = []
    for i, (icon, label, _) in enumerate(STEP_FLOW):
        if i < idx:
            cls = "done"
        elif i == idx:
            cls = "active"
        else:
            cls = "todo"
        parts.append(
            f'<div class="step {cls}"><div class="step-circle">{icon}</div>'
            f'<div class="step-label">{label}</div></div>'
        )
        if i < len(STEP_FLOW) - 1:
            line_cls = "done" if i < idx else ""
            parts.append(f'<div class="step-line {line_cls}"></div>')
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def nav_footer(current_page):
    if current_page not in PAGES:
        return
    idx = PAGES.index(current_page)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if idx > 0:
            if st.button("⬅ Back", use_container_width=True, key=f"back_{idx}"):
                goto(PAGES[idx - 1])
    with c3:
        if idx < len(PAGES) - 1:
            next_label = "Start ➡" if idx == 0 else "Next ➡"
            if st.button(next_label, use_container_width=True, type="primary", key=f"next_{idx}"):
                goto(PAGES[idx + 1])


# ==================================================================
#  Sidebar Navigation (expanding style)
# ==================================================================

page = st.session_state.get("page", "🏠 Home")

with st.sidebar:
    st.markdown(
        """
        <div style="padding:.4rem 0 1rem 0;">
            <div style="font-family:'Fraunces',serif;font-size:1.5rem;font-weight:700;">
                🎓 Degree Compass
            </div>
            <div style="font-size:.8rem;color:#B9C3DD;">
                AI-guided bachelor's degree recommendations
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("☰  Navigation", expanded=True):
        for p in PAGES:
            is_current = (p == page)
            if st.button(
                p,
                key=f"nav_{p}",
                use_container_width=True,
                type=("primary" if is_current else "secondary"),
            ):
                goto(p)

    step_idx = next((i for i, (_, _, sp) in enumerate(STEP_FLOW) if sp == page), None)
    if step_idx is not None:
        st.markdown("---")
        st.caption(f"Step {step_idx + 1} of {len(STEP_FLOW)}")

# ==================================================================
#  Home Page
# ==================================================================

if page == "🏠 Home":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">AI-Powered Guidance</div>
            <h1>Find the Bachelor's Degree That Fits <span class="accent">You</span>.</h1>
            <p>Answer a few questions about your academics, interests, skills, and career
            goals — the model matches your profile against real degree outcomes and
            recommends the programs where you're most likely to thrive.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([2, 1])
    with left:
        if st.button("🚀 Start Your Assessment", type="primary"):
            goto("📚 Academic Information")
    with right:
        pass

    st.markdown("<br>", unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            '<div class="stat-card"><div class="num">5</div>'
            '<div class="lbl">Step Assessment</div></div>',
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            '<div class="stat-card"><div class="num">AI</div>'
            '<div class="lbl">Powered Matching</div></div>',
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            '<div class="stat-card"><div class="num">Top 3</div>'
            '<div class="lbl">Personalized Picks</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("How It Works")

    steps_info = [
        ("01", "📚", "Academic Information", "Enter your Matric & Intermediate marks and subject group."),
        ("02", "❤️", "Interests", "Tell us what genuinely excites you to learn about."),
        ("03", "💡", "Skills", "Rate the abilities you already bring to the table."),
        ("04", "💼", "Career Preferences", "Pick the career directions you'd want to grow into."),
        ("05", "🎯", "Recommendation", "Get your top 3 AI-matched bachelor's degrees."),
    ]

    cols = st.columns(5)
    for col, (num, icon, title, desc) in zip(cols, steps_info):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="idx">{num}</div>
                    <div class="icon">{icon}</div>
                    <div class="title">{title}</div>
                    <div class="desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    nav_footer("🏠 Home")

# ==================================================================
#  Academic Information
# ==================================================================

elif page == "📚 Academic Information":

    st.title("📚 Academic Information")
    render_stepper(page)
    st.write("Please provide your academic details.")

    st.divider()

    # ---------------- Matric ---------------- #

    st.subheader("Matric Details")

    matric_method = st.radio(
        "Enter Matric Result As",
        ["Percentage", "Marks"],
        horizontal=True,
        key="matric_method"
    )

    if matric_method == "Percentage":

        matric_percentage = st.number_input(
            "Matric Percentage",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.get("final_matric_percentage", 70.0),
            step=0.1,
            key="matric_percentage"
        )

    else:

        col1, col2 = st.columns(2)

        with col1:
            matric_obtained = st.number_input(
                "Obtained Marks",
                min_value=0,
                value=st.session_state.get("matric_obtained_saved", 900),
                key="matric_obtained"
            )

        with col2:
            matric_total = st.number_input(
                "Total Marks",
                min_value=1,
                value=st.session_state.get("matric_total_saved", 1100),
                key="matric_total"
            )

        st.session_state["matric_obtained_saved"] = matric_obtained
        st.session_state["matric_total_saved"] = matric_total

        matric_percentage = (matric_obtained / matric_total) * 100

        st.success(f"Calculated Percentage: {matric_percentage:.2f}%")

    st.session_state["final_matric_percentage"] = matric_percentage

    st.divider()

    # ---------------- Intermediate ---------------- #

    st.subheader("Intermediate Details")

    inter_method = st.radio(
        "Enter Intermediate Result As",
        ["Percentage", "Marks"],
        horizontal=True,
        key="inter_method"
    )

    if inter_method == "Percentage":

        inter_percentage = st.number_input(
            "Intermediate Percentage",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.get("final_inter_percentage", 70.0),
            step=0.1,
            key="inter_percentage"
        )

    else:

        col1, col2 = st.columns(2)

        with col1:
            inter_obtained = st.number_input(
                "Obtained Marks",
                min_value=0,
                value=st.session_state.get("inter_obtained_saved", 900),
                key="inter_obtained"
            )

        with col2:
            inter_total = st.number_input(
                "Total Marks",
                min_value=1,
                value=st.session_state.get("inter_total_saved", 1100),
                key="inter_total"
            )

        st.session_state["inter_obtained_saved"] = inter_obtained
        st.session_state["inter_total_saved"] = inter_total

        inter_percentage = (inter_obtained / inter_total) * 100

        st.success(f"Calculated Percentage: {inter_percentage:.2f}%")

    st.session_state["final_inter_percentage"] = inter_percentage

    st.divider()

    # ---------------- Intermediate Group ---------------- #

    groups = ["Pre-Medical", "Pre-Engineering", "ICS", "Commerce", "Arts"]

    group_default = st.session_state.get("final_intermediate_group_saved", "Pre-Medical")

    intermediate_group = st.selectbox(
        "Intermediate Group",
        groups,
        index=groups.index(group_default),
        key="final_intermediate_group"
    )

    st.session_state["final_intermediate_group_saved"] = intermediate_group

    st.divider()

    # ---------------- Subject Marks ---------------- #

    st.subheader("📘 Subject Marks")

    subjects = SUBJECTS[intermediate_group]

    subject_method = st.radio(
        "Enter Subject Results As",
        ["Percentage", "Marks"],
        horizontal=True,
        key="subject_method"
    )

    subject_percentages = {}

    col1, col2 = st.columns(2)

    for i, subject in enumerate(subjects):

        column = col1 if i % 2 == 0 else col2

        with column:

            st.markdown(f"### {subject}")

            if subject_method == "Percentage":

                savekey = f"{subject}_percentage_saved"
                default_val = st.session_state.get(savekey, 70.0)

                percentage = st.number_input(
                    f"{subject} Percentage",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_val,
                    step=0.1,
                    key=f"{subject}_percentage"
                )

                st.session_state[savekey] = percentage

            else:

                obt_savekey = f"{subject}_obtained_saved"
                tot_savekey = f"{subject}_total_saved"

                obtained = st.number_input(
                    f"{subject} Obtained Marks",
                    min_value=0,
                    value=st.session_state.get(obt_savekey, 90),
                    step=1,
                    key=f"{subject}_obtained"
                )

                total = st.number_input(
                    f"{subject} Total Marks",
                    min_value=1,
                    value=st.session_state.get(tot_savekey, 100),
                    step=1,
                    key=f"{subject}_total"
                )

                st.session_state[obt_savekey] = obtained
                st.session_state[tot_savekey] = total

                percentage = (obtained / total) * 100

                st.success(f"Calculated Percentage: {percentage:.2f}%")

            subject_percentages[subject] = percentage

    all_subjects = [
        "English", "Physics", "Computer_Science", "Mathematics",
        "Chemistry", "Biology", "Accounting", "Economics",
        "Business_Mathematics", "Urdu", "Civics", "History"
    ]

    for subject in all_subjects:
        if subject not in subject_percentages:
            subject_percentages[subject] = -1

    st.session_state["subject_percentages"] = subject_percentages

    st.divider()

    nav_footer(page)

# ---------------- Interests ---------------- #
elif page == "❤️ Interests":

    st.title("❤️ Interests")
    render_stepper(page)

    st.info(
        "Please indicate how much you agree with each statement. "
        "There are no right or wrong answers—answer honestly based on your interests."
    )

    st.divider()

    options = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]

    score_map = {
        "Strongly Disagree": 1,
        "Disagree": 2,
        "Neutral": 3,
        "Agree": 4,
        "Strongly Agree": 5,
    }

    INTERESTS = [
        ("💻 I enjoy using computers, learning new technologies, or solving problems through programming.", "Interest_Computing"),
        ("🧮 I enjoy solving mathematical problems...", "Interest_Mathematics"),
        ("⚡ I enjoy learning how things work in nature, such as electricity, motion, energy, or machines.", "Interest_Physics"),
        ("🧪 I enjoy learning about chemicals, reactions, experiments, and how different substances interact.", "Interest_Chemistry"),
        ("🧬 I enjoy learning about the human body, animals, plants, diseases, and living organisms.", "Interest_Biology"),
        ("💼 I enjoy thinking about business ideas, entrepreneurship, leadership, or managing money.", "Interest_Business"),
        ("📚 I enjoy reading, writing, discussing society, history, culture, or understanding people.", "Interest_Humanities"),
        ("🏥 I would enjoy helping patients, improving people's health, or working in hospitals or healthcare settings.", "Interest_Healthcare"),
        ("🤝 I naturally understand other people's feelings and enjoy helping others when they need support.", "Empathy"),
    ]

    interest_scores = {}

    for label, key in INTERESTS:

        default_option = st.session_state.get(f"{key}_response", options[0])

        response = st.radio(
            label,
            options,
            index=options.index(default_option),
            horizontal=True,
            key=key
        )

        st.session_state[f"{key}_response"] = response
        interest_scores[key] = score_map[response]

        st.divider()

    st.session_state["interest_scores"] = interest_scores

    nav_footer(page)

# ---------------- Skills ---------------- #
elif page == "💡 Skills":

    st.title("💡 Skills")
    render_stepper(page)

    st.info(
        "Please rate your current skill level in the following areas. "
        "Choose the option that best describes you."
    )

    st.divider()

    options = ["Very Low", "Low", "Moderate", "High", "Very High"]

    score_map = {
        "Very Low": 1,
        "Low": 2,
        "Moderate": 3,
        "High": 4,
        "Very High": 5,
    }

    skill_scores = {}

    st.markdown("### 🧠 Cognitive Skills")

    cognitive_skills = [
        ("🧩 Problem Solving", "Problem_Solving"),
        ("🧠 Logical Thinking", "Logical_Thinking"),
        ("📊 Analytical Thinking", "Analytical_Thinking"),
        ("🔢 Numerical Ability", "Numerical_Ability"),
        ("⚖️ Critical Thinking", "Critical_Thinking"),
        ("✅ Decision Making", "Decision_Making"),
    ]

    for label, key in cognitive_skills:

        default_option = st.session_state.get(f"{key}_response", options[0])

        response = st.radio(
            label,
            options,
            index=options.index(default_option),
            horizontal=True,
            key=key
        )

        st.session_state[f"{key}_response"] = response
        skill_scores[key] = score_map[response]

    st.divider()

    st.markdown("### 🌟 Professional & Personal Skills")

    professional_skills = [
        ("🔍 Research Skills", "Research_Skills"),
        ("💬 Communication Skills", "Communication_Skills"),
        ("👥 Leadership", "Leadership"),
        ("🤝 Teamwork", "Teamwork"),
        ("🎨 Creativity", "Creativity"),
        ("🔎 Attention to Detail", "Attention_to_Detail"),
        ("👀 Observation Skills", "Observation_Skills"),
        ("✋ Manual Dexterity", "Manual_Dexterity"),
    ]

    for label, key in professional_skills:

        default_option = st.session_state.get(f"{key}_response", options[0])

        response = st.radio(
            label,
            options,
            index=options.index(default_option),
            horizontal=True,
            key=key
        )

        st.session_state[f"{key}_response"] = response
        skill_scores[key] = score_map[response]

    st.session_state["skill_scores"] = skill_scores

    st.divider()

    nav_footer(page)

# ---------------- Career Preferences ---------------- #
elif page == "💼 Career Preferences":

    st.title("💼 Career Preferences")
    render_stepper(page)

    st.info(
        "Select how interested you are in the following career paths. "
        "Your preferences will help us recommend bachelor's degree programs that align with your future career goals."
    )

    st.divider()

    options = [
        "Not Interested",
        "Slightly Interested",
        "Moderately Interested",
        "Interested",
        "Highly Interested",
    ]

    score_map = {
        "Not Interested": 1,
        "Slightly Interested": 2,
        "Moderately Interested": 3,
        "Interested": 4,
        "Highly Interested": 5,
    }

    career_preferences = {}

    st.markdown("### 🧑‍🔬 Academic & Technical Careers")

    academic_technical = [
        ("🔬 Research & Innovation", "Career_Preference_Research"),
        ("💻 Technology & IT", "Career_Preference_Technology"),
        ("🏥 Healthcare & Medical Services", "Career_Preference_Healthcare"),
        ("🩺 Clinical Practice", "Career_Preference_Clinical_Practice"),
    ]

    for label, key in academic_technical:
        default_option = st.session_state.get(f"{key}_response", options[0])
        response = st.radio(label, options, index=options.index(default_option), horizontal=True, key=key)
        st.session_state[f"{key}_response"] = response
        career_preferences[key] = score_map[response]

    st.divider()

    st.markdown("### 💼 Business & Leadership")

    business = [
        ("📈 Management & Leadership", "Career_Preference_Management"),
        ("🚀 Entrepreneurship & Startups", "Career_Preference_Entrepreneurship"),
        ("🏭 Industry & Manufacturing", "Career_Preference_Industry"),
    ]

    for label, key in business:
        default_option = st.session_state.get(f"{key}_response", options[0])
        response = st.radio(label, options, index=options.index(default_option), horizontal=True, key=key)
        st.session_state[f"{key}_response"] = response
        career_preferences[key] = score_map[response]

    st.divider()

    st.markdown("### 🌍 Public & Community Careers")

    public_service = [
        ("🏛 Government & Public Administration", "Career_Preference_Government"),
        ("🤝 Public & Community Service", "Career_Preference_Public_Service"),
        ("👨‍🏫 Teaching & Education", "Career_Preference_Teaching"),
    ]

    for label, key in public_service:
        default_option = st.session_state.get(f"{key}_response", options[0])
        response = st.radio(label, options, index=options.index(default_option), horizontal=True, key=key)
        st.session_state[f"{key}_response"] = response
        career_preferences[key] = score_map[response]

    st.divider()

    st.markdown("### 🎨 Creative Careers")

    default_option = st.session_state.get("Career_Preference_Creative_Arts_response", options[0])

    response = st.radio(
        "🎨 Creative Arts & Design",
        options,
        index=options.index(default_option),
        horizontal=True,
        key="Career_Preference_Creative_Arts"
    )

    st.session_state["Career_Preference_Creative_Arts_response"] = response
    career_preferences["Career_Preference_Creative_Arts"] = score_map[response]

    st.session_state["career_preferences"] = career_preferences

    st.divider()

    nav_footer(page)

# ---------------- Recommendation ---------------- #
elif page == "🎯 Recommendation":

    st.title("🎯 Degree Recommendation")
    render_stepper(page)

    if st.button("🔮 Recommend Department", type="primary"):

        input_data = {}

        input_data["Matric_Percentage"] = st.session_state["final_matric_percentage"]
        input_data["Intermediate_Percentage"] = st.session_state["final_inter_percentage"]
        input_data["Intermediate_Group"] = st.session_state["final_intermediate_group"]

        input_data.update(st.session_state["subject_percentages"])
        input_data.update(st.session_state["interest_scores"])
        input_data.update(st.session_state["skill_scores"])
        input_data.update(st.session_state["career_preferences"])

        input_df = pd.DataFrame([input_data])

        input_df = pd.get_dummies(
            input_df,
            columns=["Intermediate_Group"],
            drop_first=False
        )

        input_df = input_df.reindex(
            columns=model_columns,
            fill_value=0
        )

        probabilities = model.predict_proba(input_df)[0]

        top_indices = probabilities.argsort()[::-1][:3]

        top_departments = label_encoder.inverse_transform(top_indices)

        st.success("🎓 Top 3 Recommended Bachelor's Degrees")

        medals = ["🥇", "🥈", "🥉"]

        for i in range(3):
            st.markdown(
                f"""
                <div class="rec-card">
                    <div class="rank">{medals[i]} Recommendation #{i+1}</div>
                    <div class="dept">{top_departments[i]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    nav_footer(page)

# ---------------- About ---------------- #
elif page == "ℹ️ About":

    st.title("ℹ️ About")

    st.markdown(
        """
        <p style='text-align:center;
                  font-size:20px;
                  color:gray;
                  font-style:italic;'>
        Empowering Students with AI-Driven Career Guidance
        </p>
        """,
        unsafe_allow_html=True
    )

    st.write("""
Welcome to the **AI-Powered Bachelor Degree Recommendation System**.

This intelligent system helps students discover bachelor's degree programs that best match their **academic background, interests, skills, and career preferences**. By combining Machine Learning with a user-friendly interface, the system provides personalized recommendations to support students in making informed academic decisions.
""")

    st.divider()

    st.subheader("🎯 Purpose")

    st.write("""
Choosing the right bachelor's degree is an important decision that can shape a student's future career. This system is designed to simplify that process by analyzing multiple aspects of a student's profile instead of relying only on academic marks.
""")

    st.divider()

    st.subheader("⚙️ How It Works")

    st.markdown("""
The recommendation process consists of four simple steps:

1. 📚 Enter your academic information.
2. ❤️ Share your interests.
3. 💡 Rate your skills.
4. 💼 Select your career preferences.

The collected information is analyzed by a trained **Machine Learning model**, which predicts the bachelor's degree program that best matches your profile.
""")

    st.divider()

    st.subheader("✨ Key Features")

    st.markdown("""
- 🎓 Personalized bachelor's degree recommendations
- 🤖 AI-powered prediction using Machine Learning
- 📊 Considers academic, personal, and career-related factors
- 💻 Simple and interactive user interface
- ⚡ Fast and reliable recommendation process
""")

    st.divider()

    st.subheader("👨‍🎓 Intended Users")

    st.markdown("""
This system is designed for:

- Intermediate students
- Students exploring higher education options
- Career counselors
- Educational institutions
""")

    st.divider()

    st.subheader("🛠 Technology Stack")

    st.markdown("""
- **Programming Language:** Python
- **Framework:** Streamlit
- **Machine Learning:** Scikit-learn
- **Libraries:** Pandas, NumPy, Joblib
""")

    st.divider()

    st.subheader("⚠ Disclaimer")

    st.info("""
The recommendations generated by this system are intended to assist students in selecting suitable bachelor's degree programs. Final academic decisions should also consider personal interests, university admission requirements, and professional career guidance.
""")

    st.divider()

    st.subheader("👨‍💻 Developed By")

    st.markdown("""
**Muhammad Bilal**

Bachelor of Science in Physics

**Final Year Project**

AI-Powered Bachelor Degree Recommendation System
""")

    st.divider()

    nav_footer(page)
    
