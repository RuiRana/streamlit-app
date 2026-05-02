import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Completion Predictor",
    page_icon="🎓",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}
.main { background-color: #f7f5f0; }
.block-container { padding: 2rem 3rem; }

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-left: 4px solid #2d6a4f;
    margin-bottom: 1rem;
}
.risk-high {
    background: #fff5f5;
    border-left: 4px solid #e63946;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
}
.risk-low {
    background: #f0faf4;
    border-left: 4px solid #2d6a4f;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
}
.big-prob {
    font-family: 'DM Serif Display', serif;
    font-size: 3.5rem;
    line-height: 1;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #1a1a2e;
    margin-bottom: 0.5rem;
    border-bottom: 2px solid #2d6a4f;
    padding-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)


# ── Train model (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    features = ['english_grade', 'maths_grade', 'attendance_pct', 'enrolled_level3']
    X = df[features]
    y = df['completed_course']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc, features


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎓 Student Completion Predictor")
st.markdown("Upload your student data to train the model, then predict individual students or view the full cohort risk report.")
st.markdown("---")

# ── Sidebar — Upload ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Upload Student Data")
    uploaded = st.file_uploader("Upload students.csv", type=["csv"])
    st.markdown("---")
    st.markdown("**Required columns:**")
    st.markdown("- `english_grade` (3 or 4)")
    st.markdown("- `maths_grade` (3 or 4)")
    st.markdown("- `attendance_pct` (0–100)")
    st.markdown("- `enrolled_level3` (0 or 1)")
    st.markdown("- `completed_course` (0 or 1)")
    st.markdown("---")
    st.markdown("*Model: Random Forest Classifier*")

if uploaded is None:
    st.info("👈 Upload your **students.csv** file in the sidebar to get started.")
    st.stop()

df = pd.read_csv(uploaded)
model, accuracy, features = train_model(df)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Cohort Overview", "🔍 Predict a Student", "⚠️ At-Risk Report"])


# ═══════════════════════════════════════════════════════════
# TAB 1 — Cohort Overview
# ═══════════════════════════════════════════════════════════
with tab1:
    df['completion_probability'] = model.predict_proba(df[features])[:, 1]
    at_risk = df[df['completion_probability'] < 0.5]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#666;font-size:0.85rem;'>TOTAL STUDENTS</div>
            <div style='font-size:2rem;font-weight:600;'>{len(df)}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        rate = df['completed_course'].mean() * 100
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#666;font-size:0.85rem;'>HISTORICAL COMPLETION</div>
            <div style='font-size:2rem;font-weight:600;'>{rate:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#666;font-size:0.85rem;'>AT RISK STUDENTS</div>
            <div style='font-size:2rem;font-weight:600;color:#e63946;'>{len(at_risk)}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#666;font-size:0.85rem;'>MODEL ACCURACY</div>
            <div style='font-size:2rem;font-weight:600;'>{accuracy*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-title'>Attendance vs Completion</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#f7f5f0')
        ax.set_facecolor('#f7f5f0')
        colors = ['#e63946', '#2d6a4f']
        labels = ['Did Not Complete', 'Completed']
        for val, color, label in zip([0, 1], colors, labels):
            subset = df[df['completed_course'] == val]['attendance_pct']
            ax.hist(subset, bins=15, alpha=0.7, color=color, label=label)
        ax.set_xlabel('Attendance %')
        ax.set_ylabel('Number of Students')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    with col_b:
        st.markdown("<div class='section-title'>What Matters Most (Feature Importance)</div>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        fig2.patch.set_facecolor('#f7f5f0')
        ax2.set_facecolor('#f7f5f0')
        importance = pd.Series(model.feature_importances_, index=features).sort_values()
        colors_imp = ['#2d6a4f' if v == importance.max() else '#95b8a2' for v in importance.values]
        importance.plot(kind='barh', ax=ax2, color=colors_imp)
        ax2.set_xlabel('Importance Score')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        st.pyplot(fig2)


# ═══════════════════════════════════════════════════════════
# TAB 2 — Predict a Student
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Enter a Student's Details")
    col1, col2 = st.columns(2)

    with col1:
        eng = st.selectbox("English Grade", [3, 4])
        maths = st.selectbox("Maths Grade", [3, 4])

    with col2:
        attendance = st.slider("Previous Year Attendance %", 0, 100, 70)
        level3 = 1 if eng == 4 and maths == 4 else 0
        st.markdown(f"**Enrolled on Level 3:** {'Yes ✅' if level3 else 'No'}")
        st.caption("(Auto-calculated: requires grade 4 in both English and Maths)")

    if st.button("🔮 Predict Completion Chance", use_container_width=True):
        student = pd.DataFrame([{
            'english_grade': eng,
            'maths_grade': maths,
            'attendance_pct': attendance,
            'enrolled_level3': level3
        }])
        prob = model.predict_proba(student)[0][1]
        pct = prob * 100

        if pct >= 70:
            colour = "#2d6a4f"
            emoji = "✅"
            verdict = "ON TRACK"
            advice = "This student shows strong indicators of completion."
        elif pct >= 50:
            colour = "#f4a261"
            emoji = "🟡"
            verdict = "MONITOR"
            advice = "Some risk factors present. Worth keeping an eye on."
        else:
            colour = "#e63946"
            emoji = "⚠️"
            verdict = "AT RISK"
            advice = "This student would benefit from early intervention and pastoral support."

        st.markdown(f"""
        <div style='background:white;border-radius:16px;padding:2rem;box-shadow:0 4px 16px rgba(0,0,0,0.08);border-top:6px solid {colour};margin-top:1rem;'>
            <div style='color:{colour};font-size:0.9rem;font-weight:600;letter-spacing:2px;'>{emoji} {verdict}</div>
            <div class='big-prob' style='color:{colour};'>{pct:.0f}%</div>
            <div style='color:#666;font-size:1rem;margin-top:0.3rem;'>chance of completing the course</div>
            <div style='margin-top:1rem;padding:0.8rem 1rem;background:#f7f5f0;border-radius:8px;color:#333;'>{advice}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 3 — At-Risk Report
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ⚠️ Full At-Risk Student Report")
    st.caption("All students with less than 50% predicted completion probability, sorted by highest risk first.")

    df['completion_probability'] = model.predict_proba(df[features])[:, 1]
    at_risk_df = df[df['completion_probability'] < 0.5].copy()
    at_risk_df = at_risk_df.sort_values('completion_probability')
    at_risk_df['risk_%'] = (at_risk_df['completion_probability'] * 100).round(1).astype(str) + '%'
    at_risk_df['attendance'] = at_risk_df['attendance_pct'].round(1).astype(str) + '%'

    display_cols = ['student_id', 'english_grade', 'maths_grade', 'attendance', 'enrolled_level3', 'risk_%']
    rename = {
        'student_id': 'Student ID',
        'english_grade': 'English',
        'maths_grade': 'Maths',
        'attendance': 'Attendance',
        'enrolled_level3': 'Level 3',
        'risk_%': 'Completion Chance'
    }

    st.dataframe(
        at_risk_df[display_cols].rename(columns=rename),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(f"**{len(at_risk_df)} students flagged** out of {len(df)} total ({len(at_risk_df)/len(df)*100:.1f}%)")

    csv = at_risk_df[display_cols].rename(columns=rename).to_csv(index=False)
    st.download_button(
        label="⬇️ Download At-Risk List as CSV",
        data=csv,
        file_name="at_risk_students.csv",
        mime="text/csv",
        use_container_width=True
    )
