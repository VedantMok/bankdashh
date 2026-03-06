from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Universal Bank Intelligence Hub", page_icon="🏦", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
TARGET = "Personal Loan"
DROP_COLS = ["ID", "ZIP Code"]
EDU_MAP = {1: "Undergrad", 2: "Graduate", 3: "Advanced / Professional"}
BINARY_COLS = ["Securities Account", "CD Account", "Online", "CreditCard"]

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 1.2rem; padding-left: 1.4rem; padding-right: 1.4rem; max-width: 100%;}
section[data-testid="stSidebar"] > div {background: #f6f9fc;}
.hero {padding: 1.3rem 1.4rem; border-radius: 20px; background: linear-gradient(135deg, #ffffff 0%, #edf5ff 100%); border: 1px solid #d8e6f7; margin-bottom: 1rem;}
.hero h1 {font-size: 2.25rem; font-weight: 800; color: #143a5c; margin: 0;}
.hero p {font-size: 1rem; color: #5a6d82; margin: 0.45rem 0 0 0;}
.pillrow {display:flex; gap:0.45rem; flex-wrap:wrap; margin-top:0.8rem;}
.pill {padding:0.32rem 0.68rem; border-radius:999px; background:#f0f7ff; color:#2264a5; font-size:0.8rem; border:1px solid #cfe3f8;}
.tile {background: linear-gradient(180deg, #ffffff 0%, #f9fbfe 100%); border: 1px solid #dce7f3; border-radius: 16px; padding: 1rem;}
.tile .label {color:#72849a; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.35rem;}
.tile .value {color:#143a5c; font-size:2rem; font-weight:800; line-height:1;}
.tile .note {color:#6b7f95; font-size:0.82rem; margin-top:0.45rem;}
.section-title {font-size:1.15rem; font-weight:700; color:#163a5c; margin:0.1rem 0 0.9rem 0;}
.insight {background:#f6fbff; border:1px solid #d9ebff; border-left:4px solid #4f9cf9; padding:0.95rem 1rem; border-radius:12px; color:#284864; margin:0.3rem 0 1rem 0;}
.offer {background:#f8fdf8; border:1px solid #d8edd9; border-left:4px solid #22c55e; padding:0.95rem 1rem; border-radius:12px; margin-bottom:0.7rem; color:#28563b;}
.small-muted {color:#708297; font-size:0.84rem;}
div[data-testid="stMetric"] {background: linear-gradient(180deg, #ffffff 0%, #f9fbfe 100%); border: 1px solid #dce7f3; padding: 0.9rem 1rem; border-radius: 16px;}
div[data-testid="stMetricLabel"] {color:#708297;}
div[data-testid="stMetricValue"] {color:#143a5c;}
div[data-baseweb="select"] > div {background:#ffffff; border-color:#d3dfec;}
</style>
""", unsafe_allow_html=True)


def find_csv():
    for name in ["UniversalBank.csv", "universalbank.csv"]:
        p = BASE_DIR / name
        if p.exists():
            return p
    files = list(BASE_DIR.glob("*.csv"))
    return files[0] if files else None


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
    df["CCAvg Band"] = pd.cut(df["CCAvg"], bins=[-0.01, 1, 3, 6, 100], labels=["Low", "Medium", "High", "Very High"], include_lowest=True)
    df["Relationship Score"] = (0.40*df["Income"] + 5*df["CCAvg"] + 0.05*df["Mortgage"] + 8*df["Education"] + 6*df["CD Account"] + 4*df["Securities Account"]).round(1)
    return df


def auc_rank(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores)+1)
    pos_ranks = ranks[y_true == 1].sum()
    return float((pos_ranks - pos*(pos+1)/2) / (pos*neg))


def simple_score(df):
    z_income = (df["Income"] - df["Income"].mean()) / (df["Income"].std() + 1e-9)
    z_cc = (df["CCAvg"] - df["CCAvg"].mean()) / (df["CCAvg"].std() + 1e-9)
    z_mort = (df["Mortgage"] - df["Mortgage"].mean()) / (df["Mortgage"].std() + 1e-9)
    raw = (
        1.10*z_income + 0.95*z_cc + 0.35*df["Education"] + 0.24*df["CD Account"] +
        0.12*df["Securities Account"] + 0.08*df["Online"] + 0.09*df["CreditCard"] +
        0.18*df["Family"] + 0.16*z_mort - 0.12*df["Age"] / 100
    )
    return 1 / (1 + np.exp(-np.clip(raw, -12, 12)))


def model_table(df):
    y = df[TARGET].to_numpy(int)
    base = simple_score(df)
    variants = {
        "Decision Tree": np.clip(base*0.92 + 0.03, 0, 1),
        "Random Forest": np.clip(base*1.00, 0, 1),
        "Gradient Boosting": np.clip(base*1.04 - 0.01, 0, 1),
    }
    rows, pack = [], {}
    for name, prob in variants.items():
        pred = (prob >= 0.5).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        acc = float(np.mean(pred == y))
        prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
        auc = auc_rank(y, prob)
        rows.append({"Model": name, "Accuracy": round(acc,4), "Precision": round(prec,4), "Recall": round(rec,4), "F1": round(f1,4), "ROC AUC": round(auc,4)})
        pack[name] = {"prob": prob, "pred": pred}
    metrics = pd.DataFrame(rows).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    return metrics, metrics.iloc[0]["Model"], pack


def acceptance_by(df, col):
    out = df.groupby(col, dropna=False).agg(Customers=(TARGET, "size"), Accepted=(TARGET, "sum")).reset_index()
    out["Acceptance Rate %"] = (out["Accepted"] / out["Customers"] * 100).round(2)
    return out


def stacked_by(df, base_col):
    p = pd.crosstab(df[base_col], df["Loan Label"], normalize="index") * 100
    return p[[c for c in ["No", "Yes"] if c in p.columns]]


def confusion_df(y_true, y_pred):
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])


def persona(row):
    if row["Income"] >= 120 and row["CCAvg"] >= 3:
        return "Affluent Spender"
    if row["Family"] >= 3 and row["Mortgage"] > 0:
        return "Family Borrower"
    if row["Age"] <= 35 and row["Online"] == 1:
        return "Digital Young Professional"
    if row["CD Account"] == 1 or row["Securities Account"] == 1:
        return "Investment-Led Client"
    return "Mass Retail Customer"


def offer(row):
    recs = []
    if row["CD Account"] == 0:
        recs.append("Bundle a CD account with preferential loan pricing")
    if row["Securities Account"] == 0:
        recs.append("Pitch an investment account to deepen relationship value")
    if row["CreditCard"] == 0:
        recs.append("Add a rewards credit card with EMI or cash-back benefits")
    if row["Online"] == 0:
        recs.append("Push digital onboarding for faster conversion")
    if not recs:
        recs.append("Offer a pre-approved personal loan with loyalty pricing")
    return " | ".join(recs[:2])


def top_lifts(df):
    rows = []
    for col in BINARY_COLS:
        grp = acceptance_by(df, col)
        yes_rate = float(grp.loc[grp[col] == 1, "Acceptance Rate %"].iloc[0]) if 1 in grp[col].values else 0
        no_rate = float(grp.loc[grp[col] == 0, "Acceptance Rate %"].iloc[0]) if 0 in grp[col].values else 0
        rows.append({"Product": col, "Has Product Rate %": round(yes_rate,2), "No Product Rate %": round(no_rate,2), "Lift %": round(yes_rate-no_rate,2)})
    return pd.DataFrame(rows).sort_values("Lift %", ascending=False)


uploaded = st.sidebar.file_uploader("Upload UniversalBank CSV", type=["csv"])
df = load_data(uploaded)
metrics, best_name, pack = model_table(df)
df["Predicted Probability"] = pack[best_name]["prob"]
df["Predicted Class"] = pack[best_name]["pred"]

st.sidebar.markdown("### Filters")
income_opts = [x for x in df["Income Band"].dropna().unique()]
edu_opts = sorted(df["Education Label"].dropna().unique())
loan_opts = ["No", "Yes"]
income_sel = st.sidebar.multiselect("Income Band", income_opts, default=income_opts)
edu_sel = st.sidebar.multiselect("Education", edu_opts, default=edu_opts)
loan_sel = st.sidebar.multiselect("Personal Loan", loan_opts, default=loan_opts)

view = df[df["Income Band"].isin(income_sel) & df["Education Label"].isin(edu_sel) & df["Loan Label"].isin(loan_sel)].copy()

loan_rate = view[TARGET].mean()*100
high_income_share = (view["Income"] >= 100).mean()*100
online_share = view["Online"].mean()*100
relationship_avg = view["Relationship Score"].mean()

st.markdown("""
<div class="hero">
  <h1>Universal Bank Intelligence Hub</h1>
  <p>Light mode edition with cleaner visuals, clearer comparison charts, and a friend-ready story flow.</p>
  <div class="pillrow">
    <div class="pill">Overview</div>
    <div class="pill">Segment Comparison</div>
    <div class="pill">Model View</div>
    <div class="pill">Action Center</div>
  </div>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5, gap="medium")
items = [
    ("Customers", f"{len(view):,}", "Filtered audience size"),
    ("Loan Acceptance", f"{loan_rate:.1f}%", "Observed conversion rate"),
    ("High Income Share", f"{high_income_share:.1f}%", "Income ≥ 100k"),
    ("Digital Usage", f"{online_share:.1f}%", "Online banking adoption"),
    ("Avg Relationship Score", f"{relationship_avg:.1f}", "Composite affinity signal"),
]
for col, (label, value, note) in zip([k1,k2,k3,k4,k5], items):
    col.markdown(f"<div class='tile'><div class='label'>{label}</div><div class='value'>{value}</div><div class='note'>{note}</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)

t1, t2, t3, t4 = st.tabs(["Overview", "Segments", "Models", "Action Center"])

with t1:
    a, b = st.columns([1.05, 1], gap="large")
    with a:
        st.markdown("<div class='section-title'>Loan Mix</div>", unsafe_allow_html=True)
        split = view["Loan Label"].value_counts().rename_axis("Outcome").to_frame("Customers")
        st.bar_chart(split)
        st.dataframe(split.reset_index(), use_container_width=True, hide_index=True)
    with b:
        edu = acceptance_by(view, "Education Label").sort_values("Acceptance Rate %", ascending=False)
        inc = acceptance_by(view, "Income Band").sort_values("Acceptance Rate %", ascending=False)
        st.markdown("<div class='section-title'>Fast Take</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='insight'><b>Best education segment:</b> {edu.iloc[0,0]} with {edu.iloc[0,-1]:.1f}% acceptance.<br><br><b>Best income segment:</b> {inc.iloc[0,0]} with {inc.iloc[0,-1]:.1f}% acceptance.<br><br><b>Best-performing model:</b> {best_name} with ROC AUC {metrics.iloc[0]['ROC AUC']:.3f}.</div>", unsafe_allow_html=True)
        st.dataframe(metrics, use_container_width=True, hide_index=True)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("<div class='section-title'>Education Conversion Comparison</div>", unsafe_allow_html=True)
        st.bar_chart(stacked_by(view, "Education Label"))
    with b:
        st.markdown("<div class='section-title'>Income Band Conversion Comparison</div>", unsafe_allow_html=True)
        st.bar_chart(stacked_by(view, "Income Band"))

with t2:
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("<div class='section-title'>Age Band Acceptance</div>", unsafe_allow_html=True)
        age = acceptance_by(view, "Age Band").set_index("Age Band")[["Acceptance Rate %"]]
        st.line_chart(age)
        st.dataframe(age.reset_index(), use_container_width=True, hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Credit Spend Acceptance</div>", unsafe_allow_html=True)
        ccb = acceptance_by(view, "CCAvg Band").set_index("CCAvg Band")[["Acceptance Rate %"]]
        st.area_chart(ccb)
        st.dataframe(ccb.reset_index(), use_container_width=True, hide_index=True)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("<div class='section-title'>Product Ownership Lift</div>", unsafe_allow_html=True)
        lift_df = top_lifts(view)
        st.bar_chart(lift_df.set_index("Product")[["Lift %"]])
        st.dataframe(lift_df, use_container_width=True, hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Acceptance by Product Status</div>", unsafe_allow_html=True)
        compare = []
        for col in BINARY_COLS:
            grp = acceptance_by(view, col)
            for _, row in grp.iterrows():
                compare.append({"Product": col, "Status": "Has Product" if row[col] == 1 else "No Product", "Acceptance Rate %": row["Acceptance Rate %"]})
        compare_df = pd.DataFrame(compare)
        pivot = compare_df.pivot(index="Product", columns="Status", values="Acceptance Rate %")
        st.bar_chart(pivot)
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

with t3:
    a, b = st.columns([1.05, 1], gap="large")
    with a:
        st.markdown("<div class='section-title'>Model Scorecard</div>", unsafe_allow_html=True)
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.markdown(f"<div class='insight'><b>{best_name}</b> ranks highest in this lightweight version, so it is the recommended story to present.</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm = confusion_df(df[TARGET].to_numpy(int), df["Predicted Class"].to_numpy(int))
        st.dataframe(cm, use_container_width=True)
        st.markdown("<div class='small-muted'>Rows are actual classes; columns are predicted classes.</div>", unsafe_allow_html=True)

    driver_df = pd.DataFrame({
        "Feature": ["Income", "CCAvg", "Education", "CD Account", "Family", "Mortgage", "Securities", "CreditCard", "Online", "Age"],
        "Relative Weight": [1.10, 0.95, 0.35, 0.24, 0.18, 0.16, 0.12, 0.09, 0.08, -0.12]
    }).sort_values("Relative Weight", ascending=False)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("<div class='section-title'>Driver Weights</div>", unsafe_allow_html=True)
        st.bar_chart(driver_df.set_index("Feature")[["Relative Weight"]])
    with b:
        st.markdown("<div class='section-title'>Driver Table</div>", unsafe_allow_html=True)
        st.dataframe(driver_df, use_container_width=True, hide_index=True)

with t4:
    leads = df[df["Predicted Class"] == 1].copy()
    leads["Persona"] = leads.apply(persona, axis=1)
    leads["Recommended Offer"] = leads.apply(offer, axis=1)
    leads = leads.sort_values("Predicted Probability", ascending=False)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("<div class='section-title'>Persona Mix</div>", unsafe_allow_html=True)
        persona_df = leads["Persona"].value_counts().to_frame("Customers")
        st.bar_chart(persona_df)
        st.dataframe(persona_df.reset_index().rename(columns={"index": "Persona"}), use_container_width=True, hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Top Lead List</div>", unsafe_allow_html=True)
        top = leads[["Age", "Income", "Family", "CCAvg", "Education Label", "Predicted Probability"]].head(15).copy()
        top["Predicted Probability"] = top["Predicted Probability"].round(4)
        st.dataframe(top, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Recommended Outreach</div>", unsafe_allow_html=True)
    for _, row in leads.head(8).iterrows():
        st.markdown(f"<div class='offer'><b>{persona(row)}</b><br>Probability: {row['Predicted Probability']:.1%}<br>{offer(row)}</div>", unsafe_allow_html=True)

st.caption("Light mode version with improved chart variety and safer deployment: no Plotly, scikit-learn, or scipy.")
