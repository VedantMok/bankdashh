from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.stats import chi2_contingency

st.set_page_config(page_title="Universal Bank Loan Dashboard", page_icon="🏦", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

st.markdown("""
<style>
.main-title {font-size: 2.1rem; font-weight: 800; color: #123b5d; margin-bottom: 0.2rem;}
.subtle {color: #5b6b7a; margin-bottom: 1rem;}
.kpi-card {background: #f7fafc; border: 1px solid #d8e2eb; border-radius: 14px; padding: 16px; text-align: center;}
.kpi-label {font-size: 0.82rem; color: #617180; text-transform: uppercase; letter-spacing: 0.5px;}
.kpi-value {font-size: 1.9rem; font-weight: 800; color: #123b5d;}
.box-note {background: #f8fbff; border-left: 4px solid #2f80ed; padding: 14px; border-radius: 8px;}
.offer-card {background: #f8fcf8; border: 1px solid #d6ecd7; border-radius: 12px; padding: 14px; margin-bottom: 12px;}
.small {font-size: 0.88rem; color: #66788a;}
</style>
""", unsafe_allow_html=True)

TARGET = "Personal Loan"
DROP_COLS = ["ID", "ZIP Code"]
BINARY_COLS = ["Securities Account", "CD Account", "Online", "CreditCard"]
NUMERIC_COLS = ["Age", "Experience", "Income", "Family", "CCAvg", "Education", "Mortgage"]
EDU_MAP = {1: "Undergrad", 2: "Graduate", 3: "Advanced/Prof"}


def resolve_default_csv():
    candidates = ["UniversalBank.csv", "universalbank.csv"]
    for name in candidates:
        path = BASE_DIR / name
        if path.exists():
            return path
    csvs = list(BASE_DIR.glob("*.csv"))
    if csvs:
        return csvs[0]
    return None


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        path = resolve_default_csv()
        if path is None:
            raise FileNotFoundError("Upload UniversalBank.csv or place it beside app.py")
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Experience"] = df["Experience"].clip(lower=0)
    df["Education Label"] = df["Education"].map(EDU_MAP)
    for col in BINARY_COLS + [TARGET]:
        df[f"{col} Label"] = df[col].map({1: "Yes", 0: "No"})
    df["Income Band"] = pd.cut(
        df["Income"],
        bins=[0, 50, 100, 150, 250],
        labels=["Low", "Mid", "Upper-Mid", "High"],
        include_lowest=True
    )
    df["Age Band"] = pd.cut(
        df["Age"],
        bins=[20, 30, 40, 50, 60, 70],
        labels=["21-30", "31-40", "41-50", "51-60", "61-70"],
        include_lowest=True
    )
    df["CCAvg Band"] = pd.cut(
        df["CCAvg"],
        bins=[-0.01, 1, 3, 6, 12],
        labels=["Low", "Medium", "High", "Very High"],
        include_lowest=True
    )
    return df


@st.cache_resource
def train_models(df):
    features = [c for c in df.columns if c not in DROP_COLS + [TARGET] and not c.endswith("Label") and c not in ["Income Band", "Age Band", "CCAvg Band"]]
    X = df[features].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=25, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=12, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42),
    }
    metrics_rows = []
    scored = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        metrics_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, proba),
        })
        scored[name] = {
            "model": model,
            "pred": pred,
            "proba": proba,
            "fpr": roc_curve(y_test, proba)[0],
            "tpr": roc_curve(y_test, proba)[1],
            "cm": confusion_matrix(y_test, pred),
        }
    metrics_df = pd.DataFrame(metrics_rows).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    best_name = metrics_df.iloc[0]["Model"]
    best_model = scored[best_name]["model"]
    full_probs = best_model.predict_proba(X)[:, 1]
    scored_full = df.copy()
    scored_full["Predicted Probability"] = full_probs
    scored_full["Predicted Class"] = (scored_full["Predicted Probability"] >= 0.5).astype(int)
    return features, X_train, X_test, y_train, y_test, models, metrics_df, scored, best_name, best_model, scored_full


def style_fig(fig, height=420):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf2f7")
    fig.update_yaxes(showgrid=True, gridcolor="#edf2f7")
    return fig


def acceptance_by_group(df, col):
    out = df.groupby(col, dropna=False).agg(Customers=(TARGET, "size"), Accepted=(TARGET, "sum")).reset_index()
    out["Acceptance Rate"] = (out["Accepted"] / out["Customers"] * 100).round(2)
    return out


def cross_sell_recommendation(row):
    recs = []
    if row["CD Account"] == 0:
        recs.append("Bundle a CD account with preferential loan pricing")
    if row["Securities Account"] == 0:
        recs.append("Pitch an investment account for affluent borrowers")
    if row["CreditCard"] == 0:
        recs.append("Offer a rewards credit card with EMI benefits")
    if row["Online"] == 0:
        recs.append("Push digital onboarding to simplify loan fulfilment")
    if not recs:
        recs.append("Offer a pre-approved personal loan upgrade and loyalty pricing")
    return " | ".join(recs[:2])


def build_persona(row):
    if row["Income"] >= 120 and row["CCAvg"] >= 3:
        return "Affluent Spender"
    if row["Family"] >= 3 and row["Mortgage"] > 0:
        return "Family Borrower"
    if row["Age"] <= 35 and row["Online"] == 1:
        return "Digital Young Professional"
    if row["CD Account"] == 1 or row["Securities Account"] == 1:
        return "Investment-Oriented Customer"
    return "Mass Retail Customer"


uploaded = st.sidebar.file_uploader("Upload UniversalBank CSV", type=["csv"])
df = load_data(uploaded)
features, X_train, X_test, y_train, y_test, models, metrics_df, scored, best_name, best_model, scored_full = train_models(df)

st.sidebar.markdown("### Filters")
income_filter = st.sidebar.multiselect("Income Band", options=[x for x in df["Income Band"].dropna().unique()], default=[x for x in df["Income Band"].dropna().unique()])
edu_filter = st.sidebar.multiselect("Education", options=sorted(df["Education Label"].dropna().unique()), default=sorted(df["Education Label"].dropna().unique()))
loan_filter = st.sidebar.multiselect("Personal Loan", options=["No", "Yes"], default=["No", "Yes"])

view = df[
    df["Income Band"].isin(income_filter) &
    df["Education Label"].isin(edu_filter) &
    df["Personal Loan Label"].isin(loan_filter)
].copy()

st.markdown('<div class="main-title">🏦 Universal Bank Personal Loan Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Descriptive, diagnostic, predictive, and prescriptive analytics for personal loan targeting and cross-sell opportunities.</div>', unsafe_allow_html=True)

loan_rate = view[TARGET].mean() * 100
avg_income = view["Income"].mean()
avg_cc = view["CCAvg"].mean()
cd_share = view["CD Account"].mean() * 100
online_share = view["Online"].mean() * 100

k1, k2, k3, k4, k5 = st.columns(5)
for col, label, value in [
    (k1, "Customers", f"{len(view):,}"),
    (k2, "Loan Acceptance", f"{loan_rate:.1f}%"),
    (k3, "Avg Income ($000)", f"{avg_income:.1f}"),
    (k4, "Avg CC Spend ($000)", f"{avg_cc:.2f}"),
    (k5, "CD Account Share", f"{cd_share:.1f}%"),
]:
    col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

st.markdown("")
tab1, tab2, tab3, tab4 = st.tabs(["Descriptive", "Diagnostic", "Predictive", "Prescriptive"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        split = view[TARGET].value_counts().rename(index={0: "No", 1: "Yes"})
        fig = go.Figure(data=[go.Pie(labels=split.index, values=split.values, hole=0.55, marker_colors=["#cbd5e1", "#2f80ed"])])
        fig.update_layout(title="Personal Loan Acceptance Split")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with c2:
        edu_rates = acceptance_by_group(view, "Education Label")
        fig = px.bar(edu_rates, x="Education Label", y="Acceptance Rate", text="Acceptance Rate", color="Acceptance Rate", color_continuous_scale="Blues", title="Acceptance Rate by Education")
        fig.update_traces(texttemplate="%{text:.1f}%")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(view, x="Personal Loan Label", y="Income", color="Personal Loan Label", color_discrete_map={"No": "#94a3b8", "Yes": "#2f80ed"}, title="Income Distribution by Loan Outcome")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with c4:
        age_rates = acceptance_by_group(view, "Age Band")
        fig = px.line(age_rates, x="Age Band", y="Acceptance Rate", markers=True, title="Acceptance Rate by Age Band")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    prod_rows = []
    for col in BINARY_COLS:
        tmp = view.groupby(col)[TARGET].mean().reset_index()
        tmp["Product"] = col
        tmp["Has Product"] = tmp[col].map({0: "No", 1: "Yes"})
        tmp["Acceptance Rate"] = tmp[TARGET] * 100
        prod_rows.append(tmp[["Product", "Has Product", "Acceptance Rate"]])
    prod_df = pd.concat(prod_rows, ignore_index=True)
    fig = px.bar(prod_df, x="Product", y="Acceptance Rate", color="Has Product", barmode="group", title="Acceptance Rate by Existing Product Ownership", color_discrete_map={"No": "#cbd5e1", "Yes": "#2f80ed"})
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    st.markdown('<div class="box-note">Use this tab to explain who accepts loans more often by income, education, age, and existing product ownership.</div>', unsafe_allow_html=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        corr_cols = ["Age", "Experience", "Income", "Family", "CCAvg", "Education", "Mortgage", "Securities Account", "CD Account", "Online", "CreditCard", TARGET]
        corr = view[corr_cols].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()
        fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h", marker_color=["#d9534f" if v < 0 else "#2f80ed" for v in corr.values]))
        fig.update_layout(title="Correlation with Personal Loan")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)
    with c2:
        means = view.groupby("Personal Loan Label")[["Income", "CCAvg", "Mortgage", "Family"]].mean().T.reset_index().rename(columns={"index": "Feature"})
        fig = px.bar(means, x="Feature", y=["No", "Yes"], barmode="group", title="Feature Means by Loan Outcome")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        bands = acceptance_by_group(view, "Income Band")
        fig = px.bar(bands, x="Income Band", y="Acceptance Rate", text="Acceptance Rate", color="Acceptance Rate", color_continuous_scale="Blues", title="Acceptance Rate by Income Band")
        fig.update_traces(texttemplate="%{text:.1f}%")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with c4:
        ccb = acceptance_by_group(view, "CCAvg Band")
        fig = px.bar(ccb, x="CCAvg Band", y="Acceptance Rate", text="Acceptance Rate", color="Acceptance Rate", color_continuous_scale="Teal", title="Acceptance Rate by Credit Card Spend")
        fig.update_traces(texttemplate="%{text:.1f}%")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    chi_rows = []
    for col in BINARY_COLS + ["Education"]:
        table = pd.crosstab(view[col], view[TARGET])
        chi2, p, _, _ = chi2_contingency(table)
        chi_rows.append({"Feature": col, "Chi-Square": round(chi2, 2), "p-value": round(p, 5)})
    chi_df = pd.DataFrame(chi_rows).sort_values("Chi-Square", ascending=False)

    cross_rows = []
    for col in BINARY_COLS:
        out = acceptance_by_group(view, col)
        yes_rate = float(out.loc[out[col] == 1, "Acceptance Rate"].iloc[0]) if 1 in out[col].values else 0.0
        no_rate = float(out.loc[out[col] == 0, "Acceptance Rate"].iloc[0]) if 0 in out[col].values else 0.0
        cross_rows.append({"Product": col, "Has Product Rate %": round(yes_rate, 2), "No Product Rate %": round(no_rate, 2), "Lift": round(yes_rate - no_rate, 2)})
    cross_df = pd.DataFrame(cross_rows).sort_values("Lift", ascending=False)

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Cross-Sell Opportunity Table")
        st.dataframe(cross_df, use_container_width=True)
    with c6:
        st.subheader("Chi-Square Significance")
        st.dataframe(chi_df, use_container_width=True)

with tab3:
    st.subheader("Model Comparison")
    show_metrics = metrics_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]:
        show_metrics[col] = show_metrics[col].map(lambda x: f"{x:.3f}")
    st.dataframe(show_metrics, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for name, color in zip(scored.keys(), ["#2f80ed", "#22c55e", "#8b5cf6"]):
            fig.add_trace(go.Scatter(x=scored[name]["fpr"], y=scored[name]["tpr"], mode="lines", name=name, line=dict(width=3, color=color)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash", color="#94a3b8")))
        fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with c2:
        best_cm = scored[best_name]["cm"]
        cm_df = pd.DataFrame(best_cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", title=f"Confusion Matrix: {best_name}")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    st.subheader(f"Feature Importance — {best_name}")
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False).head(10)
        fig = px.bar(imp.sort_values(), orientation='h', title="Top Drivers of Loan Acceptance")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    st.markdown(f'<div class="box-note">Best model selected by ROC AUC: <b>{best_name}</b>. Use this in your report as the final prediction model.</div>', unsafe_allow_html=True)

with tab4:
    st.subheader("Lead Scoring and Personalized Offers")
    leads = scored_full[scored_full["Predicted Class"] == 1].copy()
    leads["Persona"] = leads.apply(build_persona, axis=1)
    leads["Recommended Offer"] = leads.apply(cross_sell_recommendation, axis=1)
    leads = leads.sort_values("Predicted Probability", ascending=False)

    c1, c2 = st.columns([1.1, 1.2])
    with c1:
        top_personas = leads["Persona"].value_counts().reset_index()
        top_personas.columns = ["Persona", "Customers"]
        fig = px.bar(top_personas, x="Persona", y="Customers", color="Customers", color_continuous_scale="Blues", title="Predicted Acceptors by Persona")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with c2:
        offer_mix = leads["Recommended Offer"].value_counts().reset_index().head(10)
        offer_mix.columns = ["Offer", "Count"]
        fig = px.bar(offer_mix, x="Count", y="Offer", orientation="h", color="Count", color_continuous_scale="Teal", title="Most Useful Offer Actions")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    st.subheader("Top Likely Acceptors")
    show_cols = ["Age", "Income", "Family", "CCAvg", "Education Label", "Mortgage", "CD Account", "Securities Account", "Online", "CreditCard", "Predicted Probability", "Persona", "Recommended Offer"]
    top_leads = leads[show_cols].head(25).copy()
    top_leads["Predicted Probability"] = top_leads["Predicted Probability"].map(lambda x: f"{x:.1%}")
    st.dataframe(top_leads, use_container_width=True)

    st.subheader("Interactive Offer Simulator")
    s1, s2, s3 = st.columns(3)
    with s1:
        age = st.slider("Age", 21, 70, 35)
        experience = st.slider("Experience", 0, 45, 10)
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
        credit = st.selectbox("CreditCard", [0, 1], format_func=lambda x: "Yes" if x else "No")

    sample = pd.DataFrame([{
        "Age": age,
        "Experience": experience,
        "Income": income,
        "Family": family,
        "CCAvg": ccavg,
        "Education": education,
        "Mortgage": mortgage,
        "Securities Account": securities,
        "CD Account": cd,
        "Online": online,
        "CreditCard": credit,
    }])
    sample_prob = float(best_model.predict_proba(sample[features])[0, 1])
    sample_row = sample.iloc[0].copy()
    sample_row["Predicted Probability"] = sample_prob
    sample_row["Personal Loan"] = 1 if sample_prob >= 0.5 else 0
    sample_row["Recommended Offer"] = cross_sell_recommendation(sample_row)
    sample_row["Persona"] = build_persona(sample_row)

    st.markdown(f"""
    <div class="offer-card">
    <b>Predicted acceptance probability:</b> {sample_prob:.1%}<br>
    <b>Persona:</b> {sample_row['Persona']}<br>
    <b>Recommended action:</b> {sample_row['Recommended Offer']}
    </div>
    """, unsafe_allow_html=True)

st.caption("Assignment note: model predicts Personal Loan, excludes ZIP Code, and highlights cross-sell opportunities for likely acceptors.")
