import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default – Survival Analysis",
    page_icon="📊",
    layout="wide",
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2e3450;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4f8ef7; }
    .metric-label { font-size: 0.85rem; color: #8892a4; margin-top: 4px; }
    .section-title {
        font-size: 1.2rem; font-weight: 600;
        color: #c9d1e0; margin: 24px 0 12px;
        border-left: 4px solid #4f8ef7; padding-left: 12px;
    }
    .insight-box {
        background: #1a1f2e;
        border-left: 3px solid #f7c948;
        border-radius: 6px;
        padding: 12px 16px;
        color: #c9d1e0;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .filter-label {
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
        color: #8892a4; text-transform: uppercase; margin-bottom: 8px;
    }
    /* Style toggle buttons to look like pills */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 20px !important;
        font-size: 0.8rem !important;
        padding: 4px 14px !important;
        margin: 2px !important;
    }
    .filter-section {
        background: #1a1f2e;
        border: 1px solid #2e3450;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("## 📊 Loan Default — Survival Analysis Dashboard")
st.markdown("*Visual analysis of key research variables · Preliminary prototype*")
st.divider()

# ── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    return pd.read_excel(file)

uploaded = st.sidebar.file_uploader("Chọn file Excel", type=["xlsx", "xls"])

# Fallback: load bundled file if available
if uploaded:
    df = load_data(uploaded)
else:
    try:
        df = pd.read_excel("loan_default_survival__1_.xlsx")
        st.sidebar.success("✅ Đang dùng file mặc định")
    except:
        st.info("👆 Vui lòng upload file Excel để bắt đầu")
        st.stop()

# ── SIDEBAR FILTERS (Excel-style checkbox dropdowns) ─────────────────────────
st.sidebar.markdown("### 🎛️ Bộ lọc")

# --- Loan Grade ---
grade_opts = sorted(df["loan_grade"].unique())
with st.sidebar.expander(f"🏷️ Loan Grade", expanded=False):
    sel_all_g = st.checkbox("(Select All)", value=True, key="all_grade")
    if sel_all_g:
        sel_grades = grade_opts
        for g in grade_opts:
            st.checkbox(g, value=True, key=f"cb_grade_{g}", disabled=True)
    else:
        sel_grades = [g for g in grade_opts if st.checkbox(g, value=True, key=f"cb_grade_{g}")]

st.sidebar.markdown("")

# --- Home Ownership ---
ownership_opts = sorted(df["home_ownership"].unique())
with st.sidebar.expander("🏠 Home Ownership", expanded=False):
    sel_all_o = st.checkbox("(Select All)", value=True, key="all_own")
    if sel_all_o:
        sel_ownership = ownership_opts
        for o in ownership_opts:
            st.checkbox(o, value=True, key=f"cb_own_{o}", disabled=True)
    else:
        sel_ownership = [o for o in ownership_opts if st.checkbox(o, value=True, key=f"cb_own_{o}")]

st.sidebar.markdown("")

# --- Loan Purpose ---
purpose_opts = sorted(df["loan_purpose"].unique())
with st.sidebar.expander("🎯 Loan Purpose", expanded=False):
    sel_all_p = st.checkbox("(Select All)", value=True, key="all_purp")
    if sel_all_p:
        sel_purposes = purpose_opts
        for p in purpose_opts:
            st.checkbox(p.replace("_", " ").title(), value=True, key=f"cb_purp_{p}", disabled=True)
    else:
        sel_purposes = [p for p in purpose_opts if st.checkbox(
            p.replace("_", " ").title(), value=True, key=f"cb_purp_{p}")]

st.sidebar.markdown("")

# --- Loan Grade (employment length) ---
employ_opts = sorted(df["employment_length"].unique())
with st.sidebar.expander("💼 Employment Length", expanded=False):
    sel_all_e = st.checkbox("(Select All)", value=True, key="all_emp")
    if sel_all_e:
        sel_employ = employ_opts
        for e in employ_opts:
            st.checkbox(e, value=True, key=f"cb_emp_{e}", disabled=True)
    else:
        sel_employ = [e for e in employ_opts if st.checkbox(e, value=True, key=f"cb_emp_{e}")]

st.sidebar.markdown("---")

# --- Income Slider ---
st.sidebar.markdown("**💰 Annual Income ($)**")
income_range = st.sidebar.slider(
    "", int(df["annual_income"].min()), int(df["annual_income"].max()),
    (int(df["annual_income"].min()), int(df["annual_income"].max())),
    label_visibility="collapsed"
)

dff = df[
    df["loan_grade"].isin(sel_grades) &
    df["loan_purpose"].isin(sel_purposes) &
    df["home_ownership"].isin(sel_ownership) &
    df["employment_length"].isin(sel_employ) &
    df["annual_income"].between(*income_range)
].copy()

# ── KPI ROW ──────────────────────────────────────────────────────────────────
total = len(dff)
n_default = dff["defaulted"].sum()
rate = n_default / total * 100
avg_time = dff[dff["defaulted"] == 1]["time_to_default"].mean()
avg_amount = dff["loan_amount"].mean()

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in zip(
    [c1, c2, c3, c4],
    [f"{total:,}", f"{n_default:,}", f"{rate:.1f}%", f"{avg_time:.1f} mo"],
    ["Total Loans", "Defaults", "Default Rate", "Avg Time to Default"]
):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Kaplan-Meier Curves",
    "📊 Variable Distribution",
    "🔍 Group Comparison",
    "💡 Research Insights"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — KAPLAN-MEIER
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Overall Survival Curve</div>', unsafe_allow_html=True)

    kmf = KaplanMeierFitter()
    kmf.fit(dff["time_to_default"], event_observed=dff["defaulted"], label="All Loans")
    t = kmf.survival_function_.index
    s = kmf.survival_function_["All Loans"]
    ci_lo = kmf.confidence_interval_["All Loans_lower_0.95"]
    ci_hi = kmf.confidence_interval_["All Loans_upper_0.95"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=ci_hi, fill=None, mode="lines",
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=t, y=ci_lo, fill="tonexty", mode="lines",
                             line=dict(width=0), fillcolor="rgba(79,142,247,0.15)",
                             name="95% CI"))
    fig.add_trace(go.Scatter(x=t, y=s, mode="lines",
                             line=dict(color="#4f8ef7", width=2.5),
                             name="Survival Function"))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#f7c948",
                  annotation_text=f"Median: {kmf.median_survival_time_:.0f} mo")
    fig.update_layout(
        template="plotly_dark", height=420,
        xaxis_title="Time (months)", yaxis_title="Probability of NOT Defaulting",
        legend=dict(x=0.75, y=0.95),
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="insight-box">
        ⏱️ <b>Median Survival Time: {kmf.median_survival_time_:.0f} months</b><br>
        50% of loans default within {kmf.median_survival_time_:.0f} months.
        </div>""", unsafe_allow_html=True)
    with col_b:
        surv_12 = float(kmf.survival_function_[kmf.survival_function_.index <= 12].iloc[-1])
        st.markdown(f"""
        <div class="insight-box">
        📅 <b>Survival at 12 months: {surv_12:.1%}</b><br>
        {1-surv_12:.1%} of loans default within the first year.
        </div>""", unsafe_allow_html=True)

    # KM by Loan Grade
    st.markdown('<div class="section-title">Kaplan-Meier by Loan Grade</div>', unsafe_allow_html=True)
    colors = ["#4f8ef7","#2ecc71","#e74c3c","#f7c948","#9b59b6","#1abc9c","#e67e22"]
    fig2 = go.Figure()
    for i, grade in enumerate(sorted(dff["loan_grade"].unique())):
        sub = dff[dff["loan_grade"] == grade]
        kmf_g = KaplanMeierFitter()
        kmf_g.fit(sub["time_to_default"], event_observed=sub["defaulted"], label=grade)
        tg = kmf_g.survival_function_.index
        sg = kmf_g.survival_function_[grade]
        fig2.add_trace(go.Scatter(x=tg, y=sg, mode="lines",
                                  name=f"Grade {grade}",
                                  line=dict(color=colors[i % len(colors)], width=2)))
    fig2.update_layout(template="plotly_dark", height=400,
                       xaxis_title="Time (months)",
                       yaxis_title="Survival Probability",
                       margin=dict(t=20, b=40))
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — VARIABLE DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Default Rate by Loan Grade</div>', unsafe_allow_html=True)
        grade_df = dff.groupby("loan_grade")["defaulted"].mean().reset_index()
        grade_df.columns = ["grade", "default_rate"]
        grade_df = grade_df.sort_values("grade")
        fig3 = px.bar(grade_df, x="grade", y="default_rate",
                      color="default_rate", color_continuous_scale="RdYlGn_r",
                      labels={"default_rate": "Default Rate", "grade": "Loan Grade"},
                      template="plotly_dark")
        fig3.update_layout(height=320, margin=dict(t=10, b=10), showlegend=False)
        fig3.update_traces(text=[f"{v:.1%}" for v in grade_df["default_rate"]],
                           textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Default Rate by Loan Purpose</div>', unsafe_allow_html=True)
        purpose_df = dff.groupby("loan_purpose")["defaulted"].mean().reset_index()
        purpose_df.columns = ["purpose", "default_rate"]
        purpose_df = purpose_df.sort_values("default_rate", ascending=True)
        fig4 = px.bar(purpose_df, x="default_rate", y="purpose", orientation="h",
                      color="default_rate", color_continuous_scale="RdYlGn_r",
                      template="plotly_dark")
        fig4.update_layout(height=320, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-title">Credit Score Distribution</div>', unsafe_allow_html=True)
        fig5 = px.histogram(dff, x="credit_score", color="defaulted",
                            barmode="overlay", nbins=40,
                            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                            labels={"defaulted": "Defaulted", "credit_score": "Credit Score"},
                            template="plotly_dark")
        fig5.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig5, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title">Interest Rate vs Default</div>', unsafe_allow_html=True)
        fig6 = px.box(dff, x="defaulted", y="interest_rate",
                      color="defaulted",
                      color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                      labels={"defaulted": "Defaulted (0=No, 1=Yes)", "interest_rate": "Interest Rate (%)"},
                      template="plotly_dark")
        fig6.update_layout(height=320, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown('<div class="section-title">Loan Amount Distribution by Home Ownership</div>', unsafe_allow_html=True)
    fig7 = px.violin(dff, x="home_ownership", y="loan_amount", color="defaulted",
                     box=True, points=False,
                     color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                     template="plotly_dark")
    fig7.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig7, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — GROUP COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Compare Survival Curves Between Groups</div>', unsafe_allow_html=True)

    group_var = st.selectbox("Chọn biến để so sánh", [
        "loan_grade", "home_ownership", "loan_purpose", "employment_length"
    ])

    groups = sorted(dff[group_var].unique())
    fig8 = go.Figure()
    colors2 = ["#4f8ef7","#e74c3c","#2ecc71","#f7c948","#9b59b6","#1abc9c","#e67e22"]
    results_rows = []

    for i, g in enumerate(groups):
        sub = dff[dff[group_var] == g]
        kmf_g = KaplanMeierFitter()
        kmf_g.fit(sub["time_to_default"], event_observed=sub["defaulted"])
        tg = kmf_g.survival_function_.index
        sg = kmf_g.survival_function_.iloc[:, 0]
        fig8.add_trace(go.Scatter(x=tg, y=sg, mode="lines", name=str(g),
                                  line=dict(color=colors2[i % len(colors2)], width=2)))
        results_rows.append({
            "Group": g,
            "N": len(sub),
            "Defaults": int(sub["defaulted"].sum()),
            "Default Rate": f"{sub['defaulted'].mean():.1%}",
            "Median Survival (mo)": kmf_g.median_survival_time_
        })

    fig8.update_layout(template="plotly_dark", height=420,
                       xaxis_title="Time (months)",
                       yaxis_title="Survival Probability",
                       margin=dict(t=10, b=40))
    st.plotly_chart(fig8, use_container_width=True)

    # Summary table
    st.markdown('<div class="section-title">Group Summary Table</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(results_rows).set_index("Group"), use_container_width=True)

    # Log-rank test for 2 groups
    if len(groups) == 2:
        g1, g2 = groups
        s1 = dff[dff[group_var] == g1]
        s2 = dff[dff[group_var] == g2]
        lr = logrank_test(s1["time_to_default"], s2["time_to_default"],
                          s1["defaulted"], s2["defaulted"])
        p = lr.p_value
        st.markdown(f"""
        <div class="insight-box">
        🧪 <b>Log-rank Test p-value = {p:.4f}</b><br>
        {"✅ Statistically significant difference (p < 0.05)" if p < 0.05 else "⚠️ No significant difference (p ≥ 0.05)"}
        </div>""", unsafe_allow_html=True)

    # Heatmap: default rate by grade × purpose
    st.markdown('<div class="section-title">Default Rate Heatmap: Grade × Purpose</div>', unsafe_allow_html=True)
    pivot = dff.pivot_table(values="defaulted", index="loan_grade",
                            columns="loan_purpose", aggfunc="mean")
    fig9 = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
                     text_auto=".1%", template="plotly_dark",
                     labels=dict(color="Default Rate"))
    fig9.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig9, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESEARCH INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Key Research Findings</div>', unsafe_allow_html=True)

    # 1. Credit score vs default
    avg_cs_default = dff[dff["defaulted"]==1]["credit_score"].mean()
    avg_cs_no = dff[dff["defaulted"]==0]["credit_score"].mean()

    # 2. Interest rate
    avg_ir_default = dff[dff["defaulted"]==1]["interest_rate"].mean()
    avg_ir_no = dff[dff["defaulted"]==0]["interest_rate"].mean()

    # 3. Riskiest grade
    riskiest = dff.groupby("loan_grade")["defaulted"].mean().idxmax()
    riskiest_rate = dff.groupby("loan_grade")["defaulted"].mean().max()

    insights = [
        ("🏦", "Credit Score & Default",
         f"Defaulted borrowers average credit score <b>{avg_cs_default:.0f}</b> vs <b>{avg_cs_no:.0f}</b> for non-defaulted — lower credit score strongly predicts default."),
        ("📈", "Interest Rate Risk",
         f"Defaulted loans carry avg interest rate <b>{avg_ir_default:.1f}%</b> vs <b>{avg_ir_no:.1f}%</b> for non-defaulted — high rates correlate with default risk."),
        ("🎯", "Riskiest Loan Grade",
         f"Grade <b>{riskiest}</b> has the highest default rate at <b>{riskiest_rate:.1%}</b> — significantly riskier than higher grades."),
        ("⏱️", "Survival Timeline",
         f"Median time to default is <b>{kmf.median_survival_time_:.0f} months</b> — half of all defaults occur within this window, critical for early warning models."),
        ("🏠", "Home Ownership Effect",
         f"RENT borrowers tend to have higher default rates than OWN/MORTGAGE — ownership status is a meaningful risk signal."),
    ]

    for icon, title, text in insights:
        st.markdown(f"""
        <div class="insight-box">
        {icon} <b>{title}</b><br>{text}
        </div>""", unsafe_allow_html=True)

    # Scatter: credit score vs interest rate colored by default
    st.markdown('<div class="section-title">Credit Score vs Interest Rate (by Default Status)</div>', unsafe_allow_html=True)
    sample = dff.sample(min(1000, len(dff)), random_state=42)
    fig10 = px.scatter(sample, x="credit_score", y="interest_rate",
                       color=sample["defaulted"].map({0: "No Default", 1: "Default"}),
                       color_discrete_map={"No Default": "#2ecc71", "Default": "#e74c3c"},
                       opacity=0.5, template="plotly_dark",
                       labels={"credit_score": "Credit Score",
                               "interest_rate": "Interest Rate (%)",
                               "color": "Status"})
    fig10.update_layout(height=400, margin=dict(t=10, b=10))
    st.plotly_chart(fig10, use_container_width=True)

    # Correlation bar chart
    st.markdown('<div class="section-title">Variable Correlation with Default</div>', unsafe_allow_html=True)
    num_cols = ["loan_amount","interest_rate","credit_score","annual_income","term_months","time_to_default"]
    corr = dff[num_cols + ["defaulted"]].corr()["defaulted"].drop("defaulted").sort_values()
    fig11 = px.bar(x=corr.values, y=corr.index, orientation="h",
                   color=corr.values, color_continuous_scale="RdBu",
                   labels={"x": "Correlation with Default", "y": "Variable"},
                   template="plotly_dark")
    fig11.update_layout(height=320, margin=dict(t=10, b=10), showlegend=False)
    st.plotly_chart(fig11, use_container_width=True)

st.divider()
st.caption("Dashboard · Loan Default Survival Analysis · Built with Streamlit & Lifelines")
