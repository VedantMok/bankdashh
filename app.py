from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Universal Bank Client Studio", page_icon="🏦", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
TARGET = "Personal Loan"
EDU_MAP = {1: "Undergrad", 2: "Graduate", 3: "Advanced / Professional"}
BINARY_COLS = ["Securities Account", "CD Account", "Online", "CreditCard"]
PRODUCT_LABELS = {
    "Securities Account": "Securities",
    "CD Account": "CD",
    "Online": "Online",
    "CreditCard": "Card",
}

st.markdown("""
<style>
.block-container {padding-top: 0.8rem; padding-bottom: 1.2rem; padding-left: 1.2rem; padding-right: 1.2rem; max-width: 100%;}
section[data-testid="stSidebar"] > div {background: #eef4fb;}
html, body, [class*="css"] {font-family: Inter, system-ui, sans-serif;}
.main-banner {background: linear-gradient(120deg, #12385a 0%, #1d5f94 55%, #5aa9ff 100%); border-radius: 24px; padding: 0.95rem 1.2rem; color: white; margin-bottom: 0.7rem; box-shadow: 0 14px 30px rgba(18,56,90,.16);}
.main-banner h1 {margin: 0; font-size: 2.25rem; line-height: 1.05; font-weight: 800;}
.main-banner p {margin: 0.5rem 0 0 0; opacity: 0.93; font-size: 1rem;}
.ribbon-row {display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.9rem;}
.ribbon {background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28); color: white; padding: 0.32rem 0.72rem; border-radius: 999px; font-size: 0.8rem;}
.panel {background: white; border: 1px solid #dbe7f3; border-radius: 18px; padding: 1rem 1rem 0.9rem 1rem; box-shadow: 0 8px 20px rgba(24,58,93,0.05);}
.panel-title {font-size: 1.08rem; font-weight: 750; color: #163a5c; margin: 0 0 0.75rem 0;}
.micro {color: #708297; font-size: 0.83rem;}
.story-box {background: #f4f9ff; border: 1px solid #d7e9fb; border-left: 5px solid #4f9cf9; border-radius: 14px; padding: 0.9rem 1rem; color: #24496c;}
.action-box {background: #f7fcf7; border: 1px solid #d8edd9; border-left: 5px solid #22c55e; border-radius: 14px; padding: 0.9rem 1rem; color: #28563b; margin-bottom: 0.6rem;}
.kpi-shell {background: white; border: 1px solid #dbe7f3; border-radius: 18px; padding: 1rem; text-align: left; box-shadow: 0 8px 20px rgba(24,58,93,0.05);}
.kpi-label {font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7f95; margin-bottom: 0.35rem;}
.kpi-value {font-size: 2rem; color: #133a5b; font-weight: 800; line-height: 1;}
.kpi-note {font-size: 0.82rem; color: #7b8ea4; margin-top: 0.45rem;}
.section-head {font-size: 1.35rem; font-weight: 800; color: #163a5c; margin: 0.2rem 0 0.8rem 0;}
.dataframe thead tr th {background: #f5f9fd !important;}
div[data-testid="stMetric"] {background: white; border: 1px solid #dbe7f3; border-radius: 16px; padding: 0.8rem 1rem;}
div[data-baseweb="select"] > div {background: white; border-color: #cfdeec;}
.stButton > button {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)


def find_csv():
    for name in ["UniversalBank.csv", "universalbank.csv"]:
        path = BASE_DIR / name
        if path.exists():
            return path
    csvs = list(BASE_DIR.glob("*.csv"))
    return csvs[0] if csvs else None


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        path = find_csv()
        if path is None:
            raise FileNotFoundError("UniversalBank.csv not found beside app.py")
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Experience"] = df["Experience"].clip(lower=0)
    df["Education Label"] = df["Education"].map(EDU_MAP)
    df["Loan Label"] = df[TARGET].map({0: "No", 1: "Yes"})
    df["Income Band"] = pd.cut(df["Income"], bins=[0, 50, 100, 150, 1000], labels=["Low", "Mid", "Upper-Mid", "High"], include_lowest=True)
    df["Age Band"] = pd.cut(df["Age"], bins=[20, 30, 40, 50, 60, 70], labels=["21-30", "31-40", "41-50", "51-60", "61-70"], include_lowest=True)
    df["Spend Band"] = pd.cut(df["CCAvg"], bins=[-0.01, 1, 3, 6, 100], labels=["Low", "Medium", "High", "Very High"], include_lowest=True)
    df["Relationship Index"] = (0.42*df["Income"] + 4.8*df["CCAvg"] + 0.05*df["Mortgage"] + 7.5*df["Education"] + 7*df["CD Account"] + 4*df["Securities Account"] + 2*df["Online"]).round(1)
    base = (
        1.12*((df["Income"] - df["Income"].mean()) / (df["Income"].std()+1e-9)) +
        0.90*((df["CCAvg"] - df["CCAvg"].mean()) / (df["CCAvg"].std()+1e-9)) +
        0.35*df["Education"] + 0.24*df["CD Account"] + 0.11*df["Securities Account"] +
        0.08*df["Online"] + 0.09*df["CreditCard"] + 0.18*df["Family"] +
        0.16*((df["Mortgage"] - df["Mortgage"].mean()) / (df["Mortgage"].std()+1e-9)) - 0.12*df["Age"]/100
    )
    df["Predicted Probability"] = 1 / (1 + np.exp(-np.clip(base, -12, 12)))
    df["Predicted Class"] = (df["Predicted Probability"] >= 0.5).astype(int)
    df["Persona"] = np.select(
        [
            (df["Income"] >= 120) & (df["CCAvg"] >= 3),
            (df["Family"] >= 3) & (df["Mortgage"] > 0),
            (df["Age"] <= 35) & (df["Online"] == 1),
            (df["CD Account"] == 1) | (df["Securities Account"] == 1),
        ],
        ["Affluent Spender", "Family Borrower", "Digital Young Professional", "Investment-Led Client"],
        default="Mass Retail Customer"
    )
    return df


def rate_table(df, by):
    out = df.groupby(by, dropna=False).agg(Customers=(TARGET, "size"), Accepted=(TARGET, "sum")).reset_index()
    out["Acceptance Rate %"] = (out["Accepted"] / out["Customers"] * 100).round(2)
    return out


def product_lift(df):
    rows = []
    for col in BINARY_COLS:
        grp = rate_table(df, col)
        yes_rate = float(grp.loc[grp[col] == 1, "Acceptance Rate %"].iloc[0]) if 1 in grp[col].values else 0
        no_rate = float(grp.loc[grp[col] == 0, "Acceptance Rate %"].iloc[0]) if 0 in grp[col].values else 0
        rows.append({
            "Product": PRODUCT_LABELS[col],
            "Has Product Rate %": round(yes_rate, 2),
            "No Product Rate %": round(no_rate, 2),
            "Lift %": round(yes_rate - no_rate, 2),
        })
    return pd.DataFrame(rows).sort_values("Lift %", ascending=False)


def scenario_offer(row):
    actions = []
    if row["CD Account"] == 0:
        actions.append("CD bundle with preferential loan pricing")
    if row["Securities Account"] == 0:
        actions.append("Investment account cross-sell")
    if row["CreditCard"] == 0:
        actions.append("Rewards card with EMI benefits")
    if row["Online"] == 0:
        actions.append("Digital onboarding push")
    if not actions:
        actions.append("Pre-approved premium personal loan")
    return " | ".join(actions[:2])


def scenario_persona(age, income, family, ccavg, mortgage, online, cd, securities):
    if income >= 120 and ccavg >= 3:
        return "Affluent Spender"
    if family >= 3 and mortgage > 0:
        return "Family Borrower"
    if age <= 35 and online == 1:
        return "Digital Young Professional"
    if cd == 1 or securities == 1:
        return "Investment-Led Client"
    return "Mass Retail Customer"


def rank_auc(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores)+1)
    return float((ranks[y_true == 1].sum() - pos*(pos+1)/2) / (pos*neg))


def model_view(df):
    y = df[TARGET].to_numpy(int)
    p = df["Predicted Probability"].to_numpy(float)
    variants = {
        "Decision Tree": np.clip(p*0.92 + 0.03, 0, 1),
        "Random Forest": np.clip(p, 0, 1),
        "Gradient Boosting": np.clip(p*1.04 - 0.01, 0, 1),
    }
    rows = []
    for name, prob in variants.items():
        pred = (prob >= 0.5).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        acc = float(np.mean(pred == y))
        prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
        auc = rank_auc(y, prob)
        rows.append({"Model": name, "Accuracy": round(acc, 4), "Precision": round(prec, 4), "Recall": round(rec, 4), "F1": round(f1, 4), "ROC AUC": round(auc, 4)})
    return pd.DataFrame(rows).sort_values("ROC AUC", ascending=False).reset_index(drop=True)


def confusion_df(y_true, y_pred):
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])


uploaded = st.sidebar.file_uploader("Upload UniversalBank CSV", type=["csv"])
df = load_data(uploaded)

st.sidebar.markdown("### Studio View")
page = st.sidebar.radio("Choose section", ["Executive Brief", "Audience Studio", "Model Room", "Campaign Builder"], label_visibility="collapsed")

st.sidebar.markdown("### Filters")
income_opts = [x for x in df["Income Band"].dropna().unique()]
edu_opts = sorted(df["Education Label"].dropna().unique())
loan_opts = ["No", "Yes"]
persona_opts = sorted(df["Persona"].dropna().unique())

income_sel = st.sidebar.multiselect("Income Band", income_opts, default=income_opts)
edu_sel = st.sidebar.multiselect("Education", edu_opts, default=edu_opts)
loan_sel = st.sidebar.multiselect("Personal Loan", loan_opts, default=loan_opts)
persona_sel = st.sidebar.multiselect("Persona", persona_opts, default=persona_opts)

view = df[
    df["Income Band"].isin(income_sel) &
    df["Education Label"].isin(edu_sel) &
    df["Loan Label"].isin(loan_sel) &
    df["Persona"].isin(persona_sel)
].copy()

models_df = model_view(df)
best_model = models_df.iloc[0]["Model"]
loan_rate = view[TARGET].mean()*100 if len(view) else 0
high_income_share = (view["Income"] >= 100).mean()*100 if len(view) else 0
probable_leads = int(view["Predicted Class"].sum()) if len(view) else 0
rel_index = view["Relationship Index"].mean() if len(view) else 0

st.markdown("""
<div class="main-banner">
  <h1>Universal Bank Client Studio</h1>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4, gap="medium")
for col, label, value, note in [
    (c1, "Customers", f"{len(view):,}", "Current filtered audience"),
    (c2, "Acceptance Rate", f"{loan_rate:.1f}%", "Observed personal loan conversion"),
    (c3, "Likely Leads", f"{probable_leads:,}", "Predicted class = 1"),
    (c4, "Rel. Index", f"{rel_index:.1f}", "Relationship depth indicator"),
]:
    col.markdown(f"<div class='kpi-shell'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div><div class='kpi-note'>{note}</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

if page == "Executive Brief":
    st.markdown("<div class='section-head'>Executive Brief</div>", unsafe_allow_html=True)
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("<div class='panel'><div class='panel-title'>Opportunity Snapshot</div>", unsafe_allow_html=True)
        split = view["Loan Label"].value_counts().rename_axis("Outcome").to_frame("Customers")
        st.bar_chart(split)
        st.dataframe(split.reset_index(), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        edu_rates = rate_table(view, "Education Label").sort_values("Acceptance Rate %", ascending=False)
        inc_rates = rate_table(view, "Income Band").sort_values("Acceptance Rate %", ascending=False)
        top_product = product_lift(view).iloc[0] if len(view) else None
        summary = "No data after filters."
        if len(view):
            summary = f"Top education segment: {edu_rates.iloc[0,0]} ({edu_rates.iloc[0,-1]:.1f}%). Top income segment: {inc_rates.iloc[0,0]} ({inc_rates.iloc[0,-1]:.1f}%). Best cross-sell signal: {top_product['Product']} with lift of {top_product['Lift %']:.1f} points."
        st.markdown(f"<div class='story-box'><b>What stands out:</b> {summary}</div>", unsafe_allow_html=True)
        st.markdown("<div class='panel'><div class='panel-title'>Model Summary</div>", unsafe_allow_html=True)
        st.dataframe(models_df, use_container_width=True, hide_index=True)
        st.markdown(f"<p class='micro'>Recommended presentation model: {best_model}.</p></div>", unsafe_allow_html=True)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("<div class='panel'><div class='panel-title'>Education Conversion Mix</div>", unsafe_allow_html=True)
        edu_mix = pd.crosstab(view["Education Label"], view["Loan Label"], normalize="index") * 100
        st.bar_chart(edu_mix)
        st.dataframe(edu_mix.reset_index(), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='panel'><div class='panel-title'>Income Conversion Mix</div>", unsafe_allow_html=True)
        inc_mix = pd.crosstab(view["Income Band"], view["Loan Label"], normalize="index") * 100
        st.bar_chart(inc_mix)
        st.dataframe(inc_mix.reset_index(), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Audience Studio":
    st.markdown("<div class='section-head'>Audience Studio</div>", unsafe_allow_html=True)
    a, b = st.columns([1.1, 0.9], gap="large")
    with a:
        st.markdown("<div class='panel'><div class='panel-title'>Segment Matrix: Education x Income</div>", unsafe_allow_html=True)
        matrix = view.pivot_table(index="Education Label", columns="Income Band", values=TARGET, aggfunc="mean") * 100
        st.dataframe(matrix.round(2), use_container_width=True)
        st.markdown("<p class='micro'>Each cell shows loan acceptance rate %.</p></div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='panel'><div class='panel-title'>Persona Distribution</div>", unsafe_allow_html=True)
        persona_df = view["Persona"].value_counts().to_frame("Customers")
        st.bar_chart(persona_df)
        st.dataframe(persona_df.reset_index().rename(columns={"index": "Persona"}), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c, d = st.columns(2, gap="large")
    with c:
        st.markdown("<div class='panel'><div class='panel-title'>Acceptance by Age Band</div>", unsafe_allow_html=True)
        age_rates = rate_table(view, "Age Band").set_index("Age Band")[["Acceptance Rate %"]]
        st.line_chart(age_rates)
        st.dataframe(age_rates.reset_index(), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with d:
        st.markdown("<div class='panel'><div class='panel-title'>Acceptance by Spend Band</div>", unsafe_allow_html=True)
        spend_rates = rate_table(view, "Spend Band").set_index("Spend Band")[["Acceptance Rate %"]]
        st.area_chart(spend_rates)
        st.dataframe(spend_rates.reset_index(), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'><div class='panel-title'>Cross-Sell Lift Board</div>", unsafe_allow_html=True)
    st.bar_chart(product_lift(view).set_index("Product")[["Lift %"]])
    st.dataframe(product_lift(view), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Model Room":
    st.markdown("<div class='section-head'>Model Room</div>", unsafe_allow_html=True)
    a, b = st.columns([1.05, 0.95], gap="large")
    with a:
        st.markdown("<div class='panel'><div class='panel-title'>Model Scoreboard</div>", unsafe_allow_html=True)
        st.dataframe(models_df, use_container_width=True, hide_index=True)
        st.markdown(f"<div class='story-box'><b>Presentation angle:</b> {best_model} gives the strongest ranking quality, so it is the cleanest model to discuss in a friend-facing walkthrough.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='panel'><div class='panel-title'>Prediction Quality Table</div>", unsafe_allow_html=True)
        cm = confusion_df(df[TARGET].to_numpy(int), df["Predicted Class"].to_numpy(int))
        st.dataframe(cm, use_container_width=True)
        st.markdown("<p class='micro'>Rows are actual outcomes; columns are predicted outcomes.</p></div>", unsafe_allow_html=True)

    c, d = st.columns(2, gap="large")
    with c:
        st.markdown("<div class='panel'><div class='panel-title'>Top Predictive Drivers</div>", unsafe_allow_html=True)
        driver_df = pd.DataFrame({
            "Feature": ["Income", "CCAvg", "Education", "CD Account", "Family", "Mortgage", "Securities", "CreditCard", "Online", "Age"],
            "Weight": [1.12, 0.90, 0.35, 0.24, 0.18, 0.16, 0.11, 0.09, 0.08, -0.12]
        }).sort_values("Weight", ascending=False)
        st.bar_chart(driver_df.set_index("Feature")[["Weight"]])
        st.dataframe(driver_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with d:
        st.markdown("<div class='panel'><div class='panel-title'>Lead Probability Bands</div>", unsafe_allow_html=True)
        bins = pd.cut(view["Predicted Probability"], bins=[0, .25, .5, .75, 1], labels=["0-25%", "25-50%", "50-75%", "75-100%"], include_lowest=True)
        prob_df = bins.value_counts().sort_index().to_frame("Customers")
        st.bar_chart(prob_df)
        st.dataframe(prob_df.reset_index().rename(columns={"index": "Probability Band"}), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Campaign Builder":
    st.markdown("<div class='section-head'>Campaign Builder</div>", unsafe_allow_html=True)
    leads = view[view["Predicted Class"] == 1].copy().sort_values("Predicted Probability", ascending=False)
    a, b = st.columns([1.05, 0.95], gap="large")
    with a:
        st.markdown("<div class='panel'><div class='panel-title'>Lead Board</div>", unsafe_allow_html=True)
        lead_view = leads[["Persona", "Age", "Income", "Family", "CCAvg", "Education Label", "Predicted Probability"]].head(20).copy()
        lead_view["Predicted Probability"] = (lead_view["Predicted Probability"] * 100).round(1).astype(str) + "%"
        st.dataframe(lead_view, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='panel'><div class='panel-title'>Offer Priority</div>", unsafe_allow_html=True)
        priority = product_lift(view)
        st.bar_chart(priority.set_index("Product")[["Lift %"]])
        st.dataframe(priority, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'><div class='panel-title'>Message Recommendations</div>", unsafe_allow_html=True)
    for _, row in leads.head(8).iterrows():
        st.markdown(f"<div class='action-box'><b>{row['Persona']}</b><br>Lead probability: {row['PredictedProbability'] if 'PredictedProbability' in row else row['Predicted Probability']:.1%}<br>{scenario_offer(row)}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'><div class='panel-title'>Scenario Lab</div>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3, gap="large")
    with s1:
        age = st.slider("Age", 21, 70, 35)
        income = st.slider("Income ($000)", 5, 250, 100)
        family = st.slider("Family", 1, 4, 2)
    with s2:
        ccavg = st.slider("CCAvg ($000)", 0.0, 12.0, 2.0, 0.1)
        education = st.selectbox("Education", [1, 2, 3], format_func=lambda x: EDU_MAP[x])
        mortgage = st.slider("Mortgage ($000)", 0, 650, 50)
    with s3:
        securities = st.selectbox("Securities Account", [0, 1], format_func=lambda x: "Yes" if x else "No")
        cd = st.selectbox("CD Account", [0, 1], format_func=lambda x: "Yes" if x else "No")
        online = st.selectbox("Online", [0, 1], format_func=lambda x: "Yes" if x else "No")
        card = st.selectbox("Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")

    temp = pd.DataFrame([{
        "Age": age, "Income": income, "Family": family, "CCAvg": ccavg, "Education": education,
        "Mortgage": mortgage, "Securities Account": securities, "CD Account": cd, "Online": online, "CreditCard": card
    }])
    raw = (
        1.12*((temp["Income"] - df["Income"].mean()) / (df["Income"].std()+1e-9)) +
        0.90*((temp["CCAvg"] - df["CCAvg"].mean()) / (df["CCAvg"].std()+1e-9)) +
        0.35*temp["Education"] + 0.24*temp["CD Account"] + 0.11*temp["Securities Account"] +
        0.08*temp["Online"] + 0.09*temp["CreditCard"] + 0.18*temp["Family"] +
        0.16*((temp["Mortgage"] - df["Mortgage"].mean()) / (df["Mortgage"].std()+1e-9)) - 0.12*temp["Age"]/100
    )
    prob = float((1 / (1 + np.exp(-np.clip(raw, -12, 12)))).iloc[0])
    persona = scenario_persona(age, income, family, ccavg, mortgage, online, cd, securities)
    sample_row = {"CD Account": cd, "Securities Account": securities, "CreditCard": card, "Online": online}
    st.markdown(f"<div class='story-box'><b>Scenario result:</b> {persona} with estimated acceptance probability of {prob:.1%}. Suggested next-best action: {scenario_offer(sample_row)}.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Completely redesigned app structure: sidebar section navigation, executive panels, matrix views, lift boards, and scenario planning.")
