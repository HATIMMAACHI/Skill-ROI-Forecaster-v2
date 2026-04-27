import streamlit as st
import joblib
import importlib
import re
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# ── PAGE CONFIG
st.set_page_config(
    page_title='Skill ROI Forecaster',
    page_icon='🎯',
    layout='centered',
    initial_sidebar_state='collapsed'
)

# ── CSS LUXE OR/NOIR
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

.main { background-color: #ffffff; }
.block-container { 
    padding-top: 0rem !important; 
    padding-bottom: 3rem !important;
    max-width: 800px !important;
}

/* HERO */
.hero {
    text-align: center;
    padding: 60px 20px 40px;
    border-bottom: 1px solid #e8e0d0;
    margin-bottom: 40px;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 64px;
    font-weight: 700;
    color: #0a0a0a;
    line-height: 1.0;
    letter-spacing: -2px;
    margin-bottom: 8px;
}
.hero-title span { color: #c9a84c; }
.hero-subtitle {
    font-size: 14px;
    color: #888;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 300;
}

/* PROFILE CARD */
.profile-card {
    background: #0a0a0a;
    border-radius: 16px;
    padding: 36px 40px;
    margin: 0 auto 40px;
    max-width: 600px;
}
.profile-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px;
    color: #c9a84c;
    margin-bottom: 4px;
    font-weight: 600;
}
.profile-subtitle {
    font-size: 12px;
    color: #666;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 24px;
}

/* RESULTS */
.result-section {
    margin-bottom: 40px;
}
.section-label {
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #c9a84c;
    margin-bottom: 6px;
    font-weight: 500;
}
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 32px;
    font-weight: 600;
    color: #0a0a0a;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid #c9a84c;
}

/* SALARY CARD */
.salary-display {
    background: #0a0a0a;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    margin-bottom: 20px;
}
.salary-label {
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 8px;
}
.salary-amount {
    font-family: 'Cormorant Garamond', serif;
    font-size: 56px;
    font-weight: 700;
    color: #c9a84c;
    line-height: 1;
}
.salary-level {
    font-size: 12px;
    color: #555;
    margin-top: 8px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* SKILL TAGS */
.skill-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid #f0ece4;
}
.skill-name {
    font-size: 14px;
    font-weight: 500;
    color: #0a0a0a;
    text-transform: capitalize;
}
.skill-score {
    font-family: 'Cormorant Garamond', serif;
    font-size: 20px;
    font-weight: 600;
    color: #c9a84c;
}
.skill-rank {
    font-size: 11px;
    color: #bbb;
    margin-right: 12px;
}

/* DIVIDER */
.divider {
    border: none;
    border-top: 1px solid #e8e0d0;
    margin: 40px 0;
}

/* BUTTON */
.stButton > button {
    background: #c9a84c !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px 40px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #b8943d !important;
    transform: translateY(-1px) !important;
}

/* MULTISELECT */
.stMultiSelect > div > div {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    color: white !important;
}

/* MULTISELECT TAGS — dorés au lieu du rouge par défaut */
span[data-baseweb="tag"] {
    background-color: #c9a84c !important;
    color: #0a0a0a !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    padding: 2px 10px !important;
}
span[data-baseweb="tag"] svg {
    fill: #0a0a0a !important;
    opacity: 0.7;
}
span[data-baseweb="tag"]:hover svg {
    opacity: 1;
}

/* SELECTBOX */
.stSelectbox > div > div {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    color: white !important;
}

/* SLIDER — tous les éléments en doré */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #c9a84c !important;
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 4px rgba(201,168,76,0.2) !important;
}
/* Valeur affichée (chiffre) */
.stSlider p, .stSlider span {
    color: #c9a84c !important;
}
/* Barre remplie — Streamlit l'injecte en inline blue, on la réoriente avec hue-rotate */
/* blue (#1c83e1) → gold via hue-rotate(38deg) + ajustements */
.stSlider > div > div > div > div:nth-child(2) {
    filter: hue-rotate(38deg) saturate(2.5) brightness(0.95) !important;
}
/* Fallback: tenter override direct sur tous les divs de la track */
[data-testid="stSlider"] div[style*="background"] {
    background-color: #c9a84c !important;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── CHARGER LES MODÈLES
@st.cache_resource
def load_models():
    model_rf = joblib.load('best_model.pkl')
    rules = joblib.load('apriori_rules.pkl')
    roi_df = joblib.load('roi_df.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model_rf, rules, roi_df, feature_columns


@st.cache_resource
def load_kmeans_assets():
    kmeans = joblib.load('kmeans_model.pkl')
    
    # EXACT column order from the training notebook to avoid feature misalignment
    correct_order = [
        'salary', 'python', 'java', 'javascript', 'aws', 'sql', 'agile', 'git', 'c#', 
        'software engineering', 'c++', 'kubernetes', 'docker', 'react', 'typescript', 
        'software development', 'linux', 'angular', 'go', 'html', 'css', 'azure', 
        'unit testing', 'jira', 'microservices', 'scrum', 'devops', 'node.js', 
        'cloud computing', '.net', 'jenkins', 'ci/cd', 'communication', 'computer science', 
        'c', 'agile development', 'machine learning', 'postgresql', 'gcp', 'nosql', 
        'mysql', 'confluence', 'distributed systems', 'sql server', 'continuous integration', 
        'software design', 'rest', "'machine learning'", 'terraform', 'kafka', 
        'data structures', 'seniority_encoded', 'job_category_data engineer', 
        'job_category_data scientist', 'job_category_ml engineer', 'job_category_software engineer'
    ]

    df_train = pd.read_csv('dataset_encode.csv').drop_duplicates()
    df_num = df_train.select_dtypes(include=['number', 'bool']).copy()

    for col in df_num.columns:
        if df_num[col].dtype == bool:
            df_num[col] = df_num[col].astype(int)

    # Reorder columns to match training
    # Handle missing columns if any (shouldn't happen with correct dataset)
    existing_cols = [c for c in correct_order if c in df_num.columns]
    df_num = df_num[existing_cols]

    scaler = StandardScaler()
    scaler.fit(df_num)

    return kmeans, scaler, existing_cols


model_rf, rules, roi_df, feature_columns = load_models()
kmeans_model, kmeans_scaler, kmeans_feature_columns = load_kmeans_assets()
feature_columns = list(getattr(model_rf, 'feature_names_in_', feature_columns))
skills_cols = roi_df['skill'].tolist()
job_category_columns = [col for col in feature_columns if col.startswith('job_category_')]
job_category_labels = {
    col: col.replace('job_category_', '').replace('_', ' ').title() for col in job_category_columns
}

niveau_map = {'Junior': 2, 'Mid Level': 4, 'Lead': 3, 'Senior': 5}


def detect_job_category_column(columns, selected_job_col=None):
    job_cols = [col for col in columns if col.startswith('job_category_')]
    if selected_job_col and selected_job_col in job_cols:
        return selected_job_col
    # No default return, let the model decide based on other features (skills)
    return None


def build_model_input(columns, selected_skills, seniority_value, salary_value=None, selected_job_col=None):
    input_df = pd.DataFrame([[0] * len(columns)], columns=columns)
    for skill in selected_skills:
        if skill in columns:
            input_df[skill] = 1

    if 'seniority_encoded' in columns:
        input_df['seniority_encoded'] = seniority_value

    job_col = detect_job_category_column(columns, selected_job_col)
    if job_col:
        input_df[job_col] = 1

    if salary_value is not None and 'salary' in columns:
        input_df['salary'] = salary_value

    return input_df


def normalize_skill_name(skill):
    if not isinstance(skill, str):
        return ''
    cleaned = skill.strip().lower().replace('_', ' ').replace('-', ' ')
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def deduplicate_skill_list(skills):
    unique = []
    seen = set()
    for skill in skills:
        key = normalize_skill_name(skill)
        if key and key not in seen:
            seen.add(key)
            unique.append(skill)
    return unique


def skill_signature(skill):
    stop_tokens = {'developer', 'development', 'engineer', 'engineering', 'advanced', 'basics', 'fundamentals'}
    tokens = [t for t in normalize_skill_name(skill).split(' ') if t and t not in stop_tokens]
    if not tokens:
        return ''
    return ' '.join(tokens[:2])


def diversify_recommendations(scored_df, top_n):
    if scored_df.empty:
        return scored_df

    selected_idx = []
    seen_names = set()
    seen_signatures = set()

    for idx, row in scored_df.iterrows():
        norm_name = normalize_skill_name(row['skill'])
        sign = skill_signature(row['skill'])
        if norm_name in seen_names:
            continue
        if sign and sign in seen_signatures:
            continue

        selected_idx.append(idx)
        seen_names.add(norm_name)
        if sign:
            seen_signatures.add(sign)

        if len(selected_idx) >= top_n:
            break

    if len(selected_idx) < top_n:
        for idx, row in scored_df.iterrows():
            norm_name = normalize_skill_name(row['skill'])
            if idx not in selected_idx and norm_name not in seen_names:
                selected_idx.append(idx)
                seen_names.add(norm_name)
            if len(selected_idx) >= top_n:
                break

    return scored_df.loc[selected_idx]


def build_personalized_roadmap(
    selected_skills,
    top_n,
    current_salary,
    selected_job_col,
    seniority_value
):
    selected_skills = deduplicate_skill_list(selected_skills)
    selected_skill_set = {normalize_skill_name(s) for s in selected_skills}

    missing_df = roi_df[~roi_df['skill'].apply(lambda s: normalize_skill_name(s) in selected_skill_set)].copy()
    missing_df['skill_norm'] = missing_df['skill'].apply(normalize_skill_name)
    missing_df = missing_df.sort_values('roi_score', ascending=False).drop_duplicates(subset=['skill_norm'], keep='first')
    if missing_df.empty:
        return missing_df

    rows = []
    for _, row in missing_df.iterrows():
        skill = row['skill']
        test_skills = selected_skills + [skill]

        salary_input = build_model_input(
            feature_columns,
            test_skills,
            seniority_value,
            selected_job_col=selected_job_col
        )
        simulated_salary = float(model_rf.predict(salary_input)[0])
        salary_gain = simulated_salary - current_salary

        job_gain = 0.0

        matching_rules = rules[
            rules['antecedents'].apply(
                lambda ant: all(normalize_skill_name(s) in selected_skill_set for s in ant)
            )
            & rules['consequents'].apply(
                lambda cons: normalize_skill_name(skill) in {normalize_skill_name(s) for s in cons}
            )
        ]
        if len(matching_rules) > 0:
            rule_strength = float((matching_rules['lift'] * matching_rules['confidence']).max())
        else:
            rule_strength = 0.0

        rows.append({
            'skill': skill,
            'avg_salary': float(row['avg_salary']),
            'roi_score': float(row['roi_score']),
            'simulated_salary': simulated_salary,
            'salary_gain': salary_gain,
            'job_gain': job_gain,
            'rule_strength': rule_strength
        })

    scored = pd.DataFrame(rows)

    # Min-max normalization for a stable blended ranking.
    for col in ['salary_gain', 'job_gain', 'rule_strength', 'roi_score']:
        cmin = scored[col].min()
        cmax = scored[col].max()
        if cmax > cmin:
            scored[f'{col}_norm'] = (scored[col] - cmin) / (cmax - cmin)
        else:
            scored[f'{col}_norm'] = 0.0

    scored['personalized_score'] = (
        0.60 * scored['salary_gain_norm']
        + 0.30 * scored['rule_strength_norm']
        + 0.10 * scored['roi_score_norm']
    )
    scored['personalized_score_100'] = scored['personalized_score'] * 100

    scored = scored.sort_values(['personalized_score', 'salary_gain'], ascending=False)
    return diversify_recommendations(scored, top_n).head(top_n)

# ══════════════════════════════════════
# HERO
# ══════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-title">Skill ROI<br><span>Forecaster</span></div>
    <div class="hero-subtitle">Discover your optimal learning path</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════
# PROFILE CARD
# ══════════════════════════════════════
st.markdown("""
<div style="text-align:center; margin-bottom: 8px;">
    <span style="font-size:11px; letter-spacing:3px; text-transform:uppercase; color:#888;">Step 1</span>
    <br>
    <span style="font-family:'Cormorant Garamond',serif; font-size:28px; font-weight:600; color:#0a0a0a;">
        Build Your Profile
    </span>
</div>
""", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1, 4, 1])

with col_center:
    mes_skills = st.multiselect(
        '🛠️ Your current skills',
        options=skills_cols,
        default=['python', 'sql'],
        help='Select all the skills you currently master'
    )

    niveau = st.selectbox(
        '📈 Experience level',
        options=['Junior', 'Mid Level', 'Lead', 'Senior'],
        index=0
    )

    top_n = st.slider('Number of recommendations', 3, 10, 10)

    analyser = st.button('✦ Analyse My Profile')

# ══════════════════════════════════════
# RESULTS — apparaissent APRÈS le clic
# ══════════════════════════════════════
if analyser and mes_skills:
    mes_skills = deduplicate_skill_list(mes_skills)

    # ── CATÉGORIE MÉTIER (Interne)
    # Initialize as None to avoid bias unless we have logic to detect it from skills
    selected_job_col = None

    # ── SALAIRE ESTIMÉ
    st.markdown("""
    <div style="text-align:center; margin-bottom: 8px;">
        <span style="font-size:11px; letter-spacing:3px; text-transform:uppercase; color:#888;">Step 2</span>
        <br>
        <span style="font-family:'Cormorant Garamond',serif; font-size:28px; font-weight:600; color:#0a0a0a;">
            Your Estimated Value
        </span>
    </div>
    """, unsafe_allow_html=True)

    input_data = build_model_input(feature_columns, mes_skills, niveau_map[niveau], selected_job_col=selected_job_col)
    salaire_predit = model_rf.predict(input_data)[0]

    st.markdown(f"""
    <div class="salary-display">
        <div class="salary-label">Estimated Annual Salary</div>
        <div class="salary-amount">${salaire_predit:,.0f}</div>
        <div class="salary-level">Based on {len(mes_skills)} skills · {niveau} level</div>
    </div>
    """, unsafe_allow_html=True)

    # ── ÉVOLUTION PAR NIVEAU
    niveaux = ['Junior', 'Mid Level', 'Lead', 'Senior']
    salaires_par_niveau = []
    for niv in niveaux:
        inp = build_model_input(feature_columns, mes_skills, niveau_map[niv], selected_job_col=selected_job_col)
        sal = model_rf.predict(inp)[0]
        salaires_par_niveau.append(sal)

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    fig2.patch.set_facecolor('#fafaf8')
    ax2.set_facecolor('#fafaf8')
    colors_niv = ['#e8e0d0' if niv != niveau else '#c9a84c' for niv in niveaux]
    bars2 = ax2.bar(niveaux, salaires_par_niveau, color=colors_niv, edgecolor='none', width=0.5)
    for bar, val in zip(bars2, salaires_par_niveau):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f'${val:,.0f}', ha='center', color='#0a0a0a', fontsize=9, fontweight='500')
    ax2.set_ylabel('Salary (USD)', color='#888', fontsize=10)
    ax2.tick_params(colors='#888', labelsize=10)
    ax2.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax2.yaxis.grid(True, color='#f0ece4', linewidth=0.5)
    ax2.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── RECOMMANDATIONS ROI
    st.markdown("""
    <div style="text-align:center; margin-bottom: 24px;">
        <span style="font-size:11px; letter-spacing:3px; text-transform:uppercase; color:#888;">Step 3</span>
        <br>
        <span style="font-family:'Cormorant Garamond',serif; font-size:28px; font-weight:600; color:#0a0a0a;">
            Your Learning Roadmap
        </span>
    </div>
    """, unsafe_allow_html=True)

    recommandations = build_personalized_roadmap(
        selected_skills=mes_skills,
        top_n=top_n,
        current_salary=float(salaire_predit),
        selected_job_col=selected_job_col,
        seniority_value=niveau_map[niveau]
    )

    for i, (_, row) in enumerate(recommandations.iterrows()):
        medal = ['🥇', '🥈', '🥉'][i] if i < 3 else f'#{i+1}'
        st.markdown(f"""
        <div class="skill-row">
            <div style="display:flex; align-items:center; gap:12px;">
                <span style="font-size:20px;">{medal}</span>
                <div>
                    <div class="skill-name">{row['skill'].title()}</div>
                    <div style="font-size:11px; color:#aaa;">Salary gain simulé: +${row['salary_gain']:,.0f}</div>
                </div>
            </div>
            <div class="skill-score">{row['personalized_score_100']:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── COMBINAISONS
    st.markdown("""
    <div style="text-align:center; margin-bottom: 24px;">
        <span style="font-size:11px; letter-spacing:3px; text-transform:uppercase; color:#888;">Step 4</span>
        <br>
        <span style="font-family:'Cormorant Garamond',serif; font-size:28px; font-weight:600; color:#0a0a0a;">
            Power Combinations
        </span>
    </div>
    """, unsafe_allow_html=True)

    suggestions = rules[rules['antecedents'].apply(
        lambda x: any(skill in x for skill in mes_skills)
    )]
    suggestions = suggestions[~suggestions['consequents'].apply(
        lambda x: 'skills_clean' in x
    )]
    suggestions = suggestions[suggestions['consequents'].apply(
        lambda x: any(skill not in mes_skills for skill in x)
    )]

    if len(suggestions) > 0:
        suggestions = suggestions.copy()
        suggestions['ant_key'] = suggestions['antecedents'].apply(lambda ant: tuple(sorted(normalize_skill_name(s) for s in ant)))
        suggestions['cons_key'] = suggestions['consequents'].apply(lambda cons: tuple(sorted(normalize_skill_name(s) for s in cons)))
        suggestions['rule_score'] = suggestions['lift'] * suggestions['confidence']
        suggestions = suggestions.sort_values('rule_score', ascending=False).drop_duplicates(subset=['cons_key'], keep='first')
        top_suggestions = suggestions.head(5)
        for _, row in top_suggestions.iterrows():
            ant = ', '.join(row['antecedents'])
            cons = ', '.join(row['consequents'])
            conf = row['confidence'] * 100
            lift = row['lift']
            st.markdown(f"""
            <div class="skill-row">
                <div>
                    <div class="skill-name">{ant.title()} → {cons.title()}</div>
                    <div style="font-size:11px; color:#aaa;">Score règle: {row['rule_score']:.2f} · Lift: {lift:.2f}</div>
                </div>
                <div class="skill-score">{conf:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info('No combinations found for your skills.')

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── STEP 5 : TON PROFIL TYPE (K-MEANS)
    st.markdown("""
    <div style="text-align:center; margin-bottom: 24px;">
        <span style="font-size:11px; letter-spacing:3px; text-transform:uppercase; color:#888;">Step 5</span>
        <br>
        <span style="font-family:'Cormorant Garamond',serif; font-size:28px; font-weight:600; color:#0a0a0a;">
            Your Profile Type
        </span>
    </div>
    """, unsafe_allow_html=True)

    cluster_names = {
        0: ('🔬 Data Scientist', '$126,191', 'Orienté machine learning · Séniorité élevée'),
        1: ('💻 Software Engineer', '$273,558', 'Généraliste · Python, Java, C++'),
        2: ('☁️ Cloud / Backend', '$272,504', 'AWS, Kubernetes, Docker, SQL'),
        3: ('🌐 Frontend Developer', '$277,456', 'JavaScript, React, HTML, CSS · Salaire le plus élevé'),
    }

    input_kmeans = build_model_input(
        kmeans_feature_columns,
        mes_skills,
        niveau_map[niveau],
        salaire_predit,
        selected_job_col=selected_job_col
    )

    try:
        input_kmeans_scaled = kmeans_scaler.transform(input_kmeans)
        cluster = kmeans_model.predict(input_kmeans_scaled)[0]
    except Exception:
        # Fallback to raw input if scaling fails unexpectedly.
        cluster = kmeans_model.predict(input_kmeans)[0]
    nom, salaire_cluster, desc = cluster_names.get(cluster, (f'Cluster {cluster}', 'N/A', ''))

    st.markdown(f"""
    <div class="salary-display">
        <div class="salary-label">Ton Profil Type</div>
        <div style="font-family:'Cormorant Garamond',serif; font-size:48px; font-weight:700; color:#c9a84c; line-height:1.1;">{nom}</div>
        <div style="font-size:13px; color:#aaa; margin-top:8px; letter-spacing:1px;">{desc}</div>
        <div style="font-size:12px; color:#666; margin-top:12px; letter-spacing:2px; text-transform:uppercase;">Salaire moyen de ce profil : {salaire_cluster}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding: 20px 0;">
        <div style="font-family:'Cormorant Garamond',serif; font-size:18px; color:#c9a84c;">Skill ROI Forecaster</div>
        <div style="font-size:11px; color:#bbb; letter-spacing:2px; margin-top:4px;">MAACHI HATIM · 2025</div>
    </div>
    """, unsafe_allow_html=True)

elif analyser and not mes_skills:
    st.warning('👆 Please select at least one skill first!')