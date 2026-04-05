import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from itertools import combinations

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Loan Default — Survival Analysis", page_icon="📊", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #f0f2f6; }
.kpi-card {
    background: white;
    border: 1.5px solid #d0dce8;
    border-radius: 10px;
    padding: 18px 10px 14px;
    text-align: center;
}
.kpi-label {
    font-size: 11px; font-weight: 700; letter-spacing: 0.1em;
    color: #5a7a96; text-transform: uppercase; margin-bottom: 6px;
}
.kpi-value { font-size: 28px; font-weight: 700; color: #1a3a5c; }
.rq-box {
    border-left: 5px solid #1a6fa6;
    background: #f4f9ff;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.rq-title { font-size: 18px; font-weight: 700; color: #1a3a5c; margin-bottom: 6px; }
.rq-question { font-size: 14px; font-style: italic; color: #4a6a86; }
.info-box {
    background: #f8f9fa; border: 1px solid #dee2e6;
    border-radius: 8px; padding: 12px 16px;
    font-size: 13px; color: #444; margin-bottom: 16px;
}
.sig-yes { color: #c0392b; font-weight: 700; }
.sig-no  { color: #888; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(f):
    return pd.read_excel(f)

uploaded = st.sidebar.file_uploader("Upload file Excel", type=["xlsx","xls"])
if uploaded:
    df = load_data(uploaded)
else:
    try:
        df = pd.read_excel("loan_default_survival_1.xlsx")
    except:
        st.info("👆 Vui lòng upload file Excel")
        st.stop()

# Credit score groups
def cs_group(v):
    if v < 580: return "Poor (<580)"
    elif v < 670: return "Fair (580-669)"
    elif v < 740: return "Good (670-739)"
    else: return "Excellent (740+)"
df["cs_group"] = df["credit_score"].apply(cs_group)
CS_ORDER = ["Poor (<580)","Fair (580-669)","Good (670-739)","Excellent (740+)"]

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────────
st.sidebar.markdown("## Filters")

grade_opts   = sorted(df["loan_grade"].unique())
own_opts     = sorted(df["home_ownership"].unique())
purp_opts    = sorted(df["loan_purpose"].unique())
employ_opts  = sorted(df["employment_length"].unique())
term_opts    = sorted(df["term_months"].unique())

sel_grade  = st.sidebar.multiselect("Loan Grade",        grade_opts,  default=grade_opts,  placeholder="Choose options")
sel_own    = st.sidebar.multiselect("Home Ownership",    own_opts,    default=own_opts,    placeholder="Choose options")
sel_purp   = st.sidebar.multiselect("Loan Purpose",      purp_opts,   default=purp_opts,   placeholder="Choose options")
sel_employ = st.sidebar.multiselect("Employment Length", employ_opts, default=employ_opts, placeholder="Choose options")
sel_term   = st.sidebar.multiselect("Term",              term_opts,   default=term_opts,   placeholder="Choose options")

st.sidebar.markdown("---")
cs_min, cs_max = int(df["credit_score"].min()), int(df["credit_score"].max())
sel_cs = st.sidebar.slider("Credit Score", cs_min, cs_max, (cs_min, cs_max))

la_min, la_max = int(df["loan_amount"].min()), int(df["loan_amount"].max())
sel_la = st.sidebar.slider("Loan Amount ($)", la_min, la_max, (la_min, la_max))

ir_min, ir_max = float(df["interest_rate"].min()), float(df["interest_rate"].max())
sel_ir = st.sidebar.slider("Interest Rate (%)", ir_min, ir_max, (ir_min, ir_max))

# Apply filters
dff = df[
    df["loan_grade"].isin(sel_grade) &
    df["home_ownership"].isin(sel_own) &
    df["loan_purpose"].isin(sel_purp) &
    df["employment_length"].isin(sel_employ) &
    df["term_months"].isin(sel_term) &
    df["credit_score"].between(*sel_cs) &
    df["loan_amount"].between(*sel_la) &
    df["interest_rate"].between(*sel_ir)
].copy()

st.sidebar.markdown(f"**{len(dff):,} / {len(df):,} loans selected**")

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("## Loan Default — Survival Analysis Dashboard 🔗")
st.markdown(f"*{len(df):,} loan records · Kaplan-Meier · Cox PH · Hazard estimation*")

# ── KPI ROW ───────────────────────────────────────────────────────────────────
n       = len(dff)
def_rate = dff["defaulted"].mean() * 100
avg_time = dff[dff["defaulted"]==1]["time_to_default"].mean()
med_cs   = dff["credit_score"].median()
avg_ir   = dff["interest_rate"].mean()

k1,k2,k3,k4,k5 = st.columns(5)
for col, label, val in zip(
    [k1,k2,k3,k4,k5],
    ["LOANS","DEFAULT RATE","AVG TIME","MEDIAN CREDIT","AVG RATE"],
    [f"{n:,}", f"{def_rate:.1f}%", f"{avg_time:.1f} mo", f"{med_cs:.0f}", f"{avg_ir:.1f}%"]
):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "RQ1: Survival Probabilities",
    "RQ2: Risk Factors (Cox)",
    "RQ3: Segment Comparison",
    "RQ4: Hazard & Timing",
    "Data"
])

OVERALL_RATE = dff["defaulted"].mean()

def bar_chart_vs_avg(data_series, title, order=None):
    """Horizontal bar chart colored red if above average, blue if below."""
    rates = data_series.sort_values(ascending=False)
    if order:
        rates = rates.reindex([o for o in order if o in rates.index])
    colors = ["#c0392b" if v > OVERALL_RATE else "#2980b9" for v in rates.values]
    counts = dff.groupby(data_series.index.name if hasattr(data_series,'name') else "tmp").size()

    fig = go.Figure()
    fig.add_vline(x=OVERALL_RATE, line_dash="dash", line_color="#888",
                  annotation_text=f"Overall {OVERALL_RATE:.1%}", annotation_position="top right")
    fig.add_trace(go.Bar(
        x=rates.values, y=rates.index.tolist(),
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in rates.values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title, template="plotly_white", height=300,
        xaxis=dict(tickformat=".0%", title="Default Rate"),
        margin=dict(t=40,b=20,l=120,r=80),
        showlegend=False
    )
    return fig

# ════════════════════════════════════════════════════
# TAB 1 — RQ1: Survival Probabilities
# ════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="rq-box">
        <div class="rq-title">RQ1 — Survival Probabilities Over Time</div>
        <div class="rq-question">What is the probability that a loan survives (does not default) over time, and how does this vary across loan grades?</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
    The <b>Kaplan-Meier estimator</b> computes the probability of a loan remaining default-free at each point in time.
    Censored loans (those that didn't default during the observation window) are handled correctly — they contribute
    information up to the point they leave the study.
    </div>""", unsafe_allow_html=True)

    # Overall KM
    kmf = KaplanMeierFitter()
    kmf.fit(dff["time_to_default"], event_observed=dff["defaulted"])
    t = kmf.survival_function_.index
    s = kmf.survival_function_.iloc[:,0]
    ci_lo = kmf.confidence_interval_.iloc[:,0]
    ci_hi = kmf.confidence_interval_.iloc[:,1]

    col_a, col_b = st.columns([2,1])
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(t)+list(t[::-1]),
                                  y=list(ci_hi)+list(ci_lo[::-1]),
                                  fill="toself", fillcolor="rgba(26,111,166,0.12)",
                                  line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=t, y=s, mode="lines",
                                  line=dict(color="#1a6fa6", width=2.5),
                                  name="Survival Function"))
        fig.add_hline(y=0.5, line_dash="dot", line_color="#c0392b",
                      annotation_text=f"Median: {kmf.median_survival_time_:.0f} mo")
        fig.update_layout(template="plotly_white", height=380,
                          title="Overall Kaplan-Meier Survival Curve",
                          xaxis_title="Time (months)", yaxis_title="Survival Probability",
                          yaxis=dict(range=[0,1]), margin=dict(t=50,b=40))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Key Statistics")
        subset_12 = kmf.survival_function_[kmf.survival_function_.index <= 12]
        surv_12 = float(subset_12.values[-1][0]) if len(subset_12) > 0 else None
        subset_24 = kmf.survival_function_[kmf.survival_function_.index <= 24]
        surv_24 = float(subset_24.values[-1][0]) if len(subset_24) > 0 else None
        subset_36 = kmf.survival_function_[kmf.survival_function_.index <= 36]
        surv_36 = float(subset_36.values[-1][0]) if len(subset_36) > 0 else None

        stats = {
            "Median survival time": f"{kmf.median_survival_time_:.0f} months",
            "Survival at 12 mo": f"{surv_12:.1%}" if surv_12 else "N/A",
            "Survival at 24 mo": f"{surv_24:.1%}" if surv_24 else "N/A",
            "Survival at 36 mo": f"{surv_36:.1%}" if surv_36 else "N/A",
            "Total loans": f"{len(dff):,}",
            "Defaults": f"{int(dff['defaulted'].sum()):,}",
            "Censored": f"{int((dff['defaulted']==0).sum()):,}",
        }
        for k, v in stats.items():
            st.markdown(f"**{k}:** {v}")

    # KM by Loan Grade
    st.markdown("#### Survival by Loan Grade")
    colors_grade = {"A":"#2ecc71","B":"#27ae60","C":"#f39c12","D":"#e67e22",
                    "E":"#e74c3c","F":"#c0392b","G":"#8e1010"}
    fig2 = go.Figure()
    for grade in sorted(dff["loan_grade"].unique()):
        sub = dff[dff["loan_grade"]==grade]
        kmf_g = KaplanMeierFitter()
        kmf_g.fit(sub["time_to_default"], event_observed=sub["defaulted"])
        tg = kmf_g.survival_function_.index
        sg = kmf_g.survival_function_.iloc[:,0]
        fig2.add_trace(go.Scatter(x=tg, y=sg, mode="lines",
                                   name=f"{grade} (n={len(sub):,})",
                                   line=dict(color=colors_grade.get(grade,"#888"), width=2)))
    fig2.update_layout(template="plotly_white", height=400,
                       title="Kaplan-Meier by Loan Grade",
                       xaxis_title="Time (months)", yaxis_title="Survival Probability",
                       yaxis=dict(range=[0,1]), margin=dict(t=50,b=40))
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════
# TAB 2 — RQ2: Cox PH
# ════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="rq-box">
        <div class="rq-title">RQ2 — Risk Factors (Cox Proportional Hazards)</div>
        <div class="rq-question">Which borrower and loan characteristics are the strongest predictors of default risk, controlling for all other variables simultaneously?</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
    The <b>Cox PH model</b> estimates a <b>Hazard Ratio (HR)</b> for each variable.
    HR > 1 means higher default risk; HR &lt; 1 means lower risk.
    Only numeric variables are included below.
    </div>""", unsafe_allow_html=True)

    cox_cols = ["loan_amount","interest_rate","credit_score","annual_income","term_months","time_to_default","defaulted"]
    cox_df = dff[cox_cols].dropna().copy()

    try:
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="time_to_default", event_col="defaulted")
        summary = cph.summary[["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].copy()
        summary.columns = ["HR","HR_lo","HR_hi","p_value"]
        summary["Significant"] = summary["p_value"] < 0.05
        summary = summary.sort_values("HR", ascending=True)

        fig_cox = go.Figure()
        colors_cox = ["#c0392b" if sig else "#2980b9" for sig in summary["Significant"]]
        fig_cox.add_trace(go.Scatter(
            x=summary["HR"], y=summary.index,
            mode="markers",
            marker=dict(color=colors_cox, size=10),
            error_x=dict(
                type="data",
                symmetric=False,
                array=summary["HR_hi"]-summary["HR"],
                arrayminus=summary["HR"]-summary["HR_lo"],
                color="#aaa"
            ),
            name="Hazard Ratio"
        ))
        fig_cox.add_vline(x=1, line_dash="dash", line_color="#888")
        fig_cox.update_layout(template="plotly_white", height=380,
                               title="Hazard Ratios with 95% CI (red = significant)",
                               xaxis_title="Hazard Ratio", margin=dict(t=50,l=150))
        st.plotly_chart(fig_cox, use_container_width=True)

        st.markdown("#### Cox PH Model Summary")
        disp = summary.copy()
        disp["HR"] = disp["HR"].round(4)
        disp["p_value"] = disp["p_value"].round(4)
        disp["Sig."] = disp["Significant"].map({True:"✅ Yes", False:"No"})
        st.dataframe(disp[["HR","HR_lo","HR_hi","p_value","Sig."]].rename(
            columns={"HR_lo":"HR Lower 95%","HR_hi":"HR Upper 95%","p_value":"p-value"}
        ), use_container_width=True)
    except Exception as e:
        st.warning(f"Cox model could not be fitted: {e}")

# ════════════════════════════════════════════════════
# TAB 3 — RQ3: Segment Comparison
# ════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="rq-box">
        <div class="rq-title">RQ3 — Default Risk Comparison Across Segments</div>
        <div class="rq-question">How does default risk (measured by survival probabilities and hazard rates) differ across different borrower segments defined by loan grade, loan purpose, home ownership status, and credit score groups?</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
    <b>Default Rate</b> is simply the percentage of loans in each group that eventually defaulted.
    Bars highlighted in <b style="color:#c0392b">red</b> exceed the overall average — these segments carry higher risk.
    The dashed line marks the overall default rate across all loans.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        gr = dff.groupby("loan_grade")["defaulted"].mean().sort_values(ascending=False)
        st.plotly_chart(bar_chart_vs_avg(gr, "Default Rate by Loan Grade"), use_container_width=True)
    with col2:
        csg = dff.groupby("cs_group")["defaulted"].mean().reindex(CS_ORDER)
        st.plotly_chart(bar_chart_vs_avg(csg, "Default Rate by Credit Score Group"), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        pg = dff.groupby("loan_purpose")["defaulted"].mean().sort_values(ascending=False)
        st.plotly_chart(bar_chart_vs_avg(pg, "Default Rate by Loan Purpose"), use_container_width=True)
    with col4:
        og = dff.groupby("home_ownership")["defaulted"].mean().sort_values(ascending=False)
        st.plotly_chart(bar_chart_vs_avg(og, "Default Rate by Home Ownership"), use_container_width=True)

    # Stratified KM + Log-rank pairwise
    st.markdown("---")
    st.markdown("### Stratified Survival Curves — Segment Deep Dive")
    st.markdown("""<div class="info-box">
    Here we overlay <b style="color:#1a6fa6">Kaplan-Meier curves</b> for each segment on the same chart.
    If one group's curve drops faster than another, that group defaults sooner.
    The <b style="color:#1a6fa6">Log-Rank test</b> on the right tells us whether the gap is statistically real.
    </div>""", unsafe_allow_html=True)

    seg_var = st.selectbox("Choose segment variable:", {
        "loan_grade":"Loan Grade",
        "home_ownership":"Home Ownership",
        "loan_purpose":"Loan Purpose",
        "employment_length":"Employment Length",
        "cs_group":"Credit Score Group"
    })
    seg_label = {"loan_grade":"Loan Grade","home_ownership":"Home Ownership",
                 "loan_purpose":"Loan Purpose","employment_length":"Employment Length",
                 "cs_group":"Credit Score Group"}[seg_var]

    groups = sorted(dff[seg_var].unique())
    palette = px.colors.qualitative.Set2

    col_km, col_lr = st.columns([3,2])
    with col_km:
        fig_seg = go.Figure()
        for i, g in enumerate(groups):
            sub = dff[dff[seg_var]==g]
            kmf_s = KaplanMeierFitter()
            kmf_s.fit(sub["time_to_default"], event_observed=sub["defaulted"])
            ts = kmf_s.survival_function_.index
            ss = kmf_s.survival_function_.iloc[:,0]
            fig_seg.add_trace(go.Scatter(x=ts, y=ss, mode="lines",
                                          name=f"{g} (n={len(sub):,})",
                                          line=dict(color=palette[i%len(palette)], width=2)))
        fig_seg.update_layout(template="plotly_white", height=420,
                               title=f"Survival by {seg_label}",
                               xaxis_title="Time (months)", yaxis_title="Survival Probability",
                               yaxis=dict(range=[0,1]), margin=dict(t=50,b=40))
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_lr:
        st.markdown("#### Log-Rank Pairwise Tests")
        rows = []
        for g1, g2 in combinations(groups, 2):
            s1 = dff[dff[seg_var]==g1]
            s2 = dff[dff[seg_var]==g2]
            try:
                lr = logrank_test(s1["time_to_default"], s2["time_to_default"],
                                  s1["defaulted"], s2["defaulted"])
                rows.append({"Group A":g1,"Group B":g2,
                              "Test Stat":round(lr.test_statistic,3),
                              "p-value":round(lr.p_value,4),
                              "Sig.":"Yes" if lr.p_value<0.05 else "No"})
            except:
                pass
        if rows:
            lr_df = pd.DataFrame(rows)
            st.dataframe(lr_df, use_container_width=True, height=380)

# ════════════════════════════════════════════════════
# TAB 4 — RQ4: Hazard & Timing
# ════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="rq-box">
        <div class="rq-title">RQ4 — Hazard Rates & Default Timing</div>
        <div class="rq-question">When are loans most at risk of defaulting? Are there specific time windows where the hazard rate spikes?</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
    The <b>hazard rate</b> is the instantaneous risk of defaulting at time t, given survival until t.
    Peaks in the hazard curve indicate high-risk periods. Below we also show the distribution of
    actual default timing.
    </div>""", unsafe_allow_html=True)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        # Smoothed hazard via KM derivative
        kmf2 = KaplanMeierFitter()
        kmf2.fit(dff["time_to_default"], event_observed=dff["defaulted"])
        sf = kmf2.survival_function_.iloc[:,0]
        t_vals = sf.index.tolist()
        hazard = []
        for i in range(1, len(t_vals)):
            dt = t_vals[i] - t_vals[i-1]
            if dt > 0 and sf.iloc[i-1] > 0:
                h = (sf.iloc[i-1] - sf.iloc[i]) / (dt * sf.iloc[i-1])
            else:
                h = 0
            hazard.append(h)
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=t_vals[1:], y=hazard, mode="lines",
                                    fill="tozeroy", fillcolor="rgba(192,57,43,0.15)",
                                    line=dict(color="#c0392b", width=2), name="Hazard Rate"))
        fig_h.update_layout(template="plotly_white", height=350,
                             title="Estimated Hazard Rate Over Time",
                             xaxis_title="Time (months)", yaxis_title="Hazard Rate",
                             margin=dict(t=50,b=40))
        st.plotly_chart(fig_h, use_container_width=True)

    with col_h2:
        # Default timing histogram
        defaulted_times = dff[dff["defaulted"]==1]["time_to_default"]
        fig_dist = px.histogram(defaulted_times, nbins=30, template="plotly_white",
                                title="Distribution of Default Timing",
                                labels={"value":"Time to Default (months)","count":"Count"},
                                color_discrete_sequence=["#1a6fa6"])
        fig_dist.update_layout(height=350, margin=dict(t=50,b=40), showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Default timing by grade
    st.markdown("#### Default Timing by Loan Grade")
    fig_box = px.box(dff[dff["defaulted"]==1], x="loan_grade", y="time_to_default",
                     category_orders={"loan_grade":sorted(dff["loan_grade"].unique())},
                     color="loan_grade", template="plotly_white",
                     labels={"loan_grade":"Loan Grade","time_to_default":"Time to Default (months)"},
                     title="Time to Default Distribution by Grade")
    fig_box.update_layout(height=380, showlegend=False, margin=dict(t=50,b=40))
    st.plotly_chart(fig_box, use_container_width=True)

# ════════════════════════════════════════════════════
# TAB 5 — Data
# ════════════════════════════════════════════════════
with tab5:
    st.markdown("### Raw Data")
    st.markdown(f"Showing **{len(dff):,}** records after filters")
    st.dataframe(dff.reset_index(drop=True), use_container_width=True, height=500)

    st.markdown("### Descriptive Statistics")
    st.dataframe(dff.describe().round(2), use_container_width=True)

st.markdown("---")
st.caption("Loan Default Survival Analysis Dashboard · Kaplan-Meier · Cox PH · Built with Streamlit")
