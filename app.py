from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.stats import chi2_contingency

st.set_page_config(page_title="Universal Bank Loan Dashboard", page_icon="🏦", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
TARGET = "Personal Loan"
DROP_COLS = ["ID", "ZIP Code"]
BINARY_COLS = ["Securities Account", "CD Account", "Online", "CreditCard"]
EDU_MAP = {1: "Undergrad", 2: "Graduate", 3: "Advanced/Professional"}

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
.big-title {font-size: 2rem; font-weight: 800; color: #123b5d;}
.subtle {color: #5f6f7f; margin-bottom: 1rem;}
.note {background:#f6fbff; border-left:4px solid #2f80ed; padding:0.9rem; border-radius:8px;}
.offer {background:#f7fcf7; border:1px solid #d7ead8; padding:0.9rem; border-radius:10px; margin-bottom:0.6rem;}
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
        p = find_csv()
        if p is None:
            raise FileNotFoundError("UniversalBank.csv not found beside app.py")
        df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    df["Experience"] = df["Experience"].clip(lower=0)
    df["Education Label"] = df["Education"].map(EDU_MAP)
    df["Loan Label"] = df[TARGET].map({0: "No", 1: "Yes"})
    df["Income Band"] = pd.cut(df["Income"], bins=[0, 50, 100, 150, 1000], labels=["Low", "Mid", "Upper-Mid", "High"], include_lowest=True)
    df["Age Band"] = pd.cut(df["Age"], bins=[20, 30, 40, 50, 60, 70], labels=["21-30", "31-40", "41-50", "51-60", "61-70"], include_lowest=True)
    df["CCAvg Band"] = pd.cut(df["CCAvg"], bins=[-0.01, 1, 3, 6, 50], labels=["Low", "Medium", "High", "Very High"], include_lowest=True)
    return df


@st.cache_resource
def fit_models(df):
    feature_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET, "Education Label", "Loan Label", "Income Band", "Age Band", "CCAvg Band"]]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=25, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=12, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42),
    }
    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "F1": round(f1_score(y_test, pred, zero_division=0), 4),
            "ROC AUC": round(roc_auc_score(y_test, proba), 4),
        })
        fitted[name] = {"model": model, "pred": pred, "proba": proba, "cm": confusion_matrix(y_test, pred)}
    metrics = pd.DataFrame(rows).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    best_model = fitted[best_name]["model"]
    full = df.copy()
    full["Predicted Probability"] = best_model.predict_proba(X)[:, 1]
    full["Predicted Class"] = (full["Predicted Probability"] >= 0.5).astype(int)
    return feature_cols, metrics, fitted, best_name, best_model, full


def grp_rate(df, col):
    out = df.groupby(col, dropna=False).agg(Customers=(TARGET, "size"), Accepted=(TARGET, "sum")).reset_index()
    out["Acceptance Rate %"] = (out["Accepted"] / out["Customers"] * 100).round(2)
    return out


def persona(row):
    if row["Income"] >= 120 and row["CCAvg"] >= 3:
        return "Affluent Spender"
    if row["Family"] >= 3 and row["Mortgage"] > 0:
        return "Family Borrower"
    if row["Age"] <= 35 and row["Online"] == 1:
        return "Digital Young Professional"
    if row["CD Account"] == 1 or row["Securities Account"] == 1:
        return "Investment-Oriented Customer"
    return "Mass Retail Customer"


def offer(row):
    recs = []
    if row["CD Account"] == 0:
        recs.append("CD bundle with better loan pricing")
    if row["Securities Account"] == 0:
        recs.append("Investment account cross-sell")
    if row["CreditCard"] == 0:
        recs.append("Rewards credit card with EMI perks")
    if row["Online"] == 0:
        recs.append("Digital onboarding push")
    if not recs:
        recs.append("Pre-approved premium loan offer")
    return " | ".join(recs[:2])


uploaded = st.sidebar.file_uploader("Upload UniversalBank CSV", type=["csv"])
df = load_data(uploaded)
feature_cols, metrics, fitted, best_name, best_model, scored = fit_models(df)

st.sidebar.markdown("### Filters")
income_sel = st.sidebar.multiselect("Income Band", [x for x in df["Income Band"].dropna().unique()], default=[x for x in df["Income Band"].dropna().unique()])
edu_sel = st.sidebar.multiselect("Education", sorted(df["Education Label"].dropna().unique()), default=sorted(df["Education Label"].dropna().unique()))
loan_sel = st.sidebar.multiselect("Personal Loan", ["No", "Yes"], default=["No", "Yes"])

view = df[df["Income Band"].isin(income_sel) & df["Education Label"].isin(edu_sel) & df["Loan Label"].isin(loan_sel)].copy()

st.markdown('<div class="big-title">🏦 Universal Bank Personal Loan Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Alternative version with no Plotly dependency. Uses only Streamlit, pandas, scikit-learn, and scipy.</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Customers", f"{len(view):,}")
k2.metric("Loan Acceptance", f"{view[TARGET].mean()*100:.1f}%")
k3.metric("Avg Income ($000)", f"{view['Income'].mean():.1f}")
k4.metric("Avg CCAvg ($000)", f"{view['CCAvg'].mean():.2f}")
k5.metric("Online Share", f"{view['Online'].mean()*100:.1f}%")

t1, t2, t3, t4 = st.tabs(["Descriptive", "Diagnostic", "Predictive", "Prescriptive"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Loan Split")
        split = view["Loan Label"].value_counts().rename_axis("Outcome").to_frame("Customers")
        st.bar_chart(split)
        st.dataframe(split, use_container_width=True)
    with c2:
        st.subheader("Acceptance by Education")
        edu = grp_rate(view, "Education Label").set_index("Education Label")[["Acceptance Rate %"]]
        st.bar_chart(edu)
        st.dataframe(edu.reset_index(), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Acceptance by Income Band")
        inc = grp_rate(view, "Income Band").set_index("Income Band")[["Acceptance Rate %"]]
        st.bar_chart(inc)
        st.dataframe(inc.reset_index(), use_container_width=True)
    with c4:
        st.subheader("Acceptance by Age Band")
        age = grp_rate(view, "Age Band").set_index("Age Band")[["Acceptance Rate %"]]
        st.line_chart(age)
        st.dataframe(age.reset_index(), use_container_width=True)

    st.subheader("Existing Product Ownership vs Loan Acceptance")
    prod_tables = []
    for col in BINARY_COLS:
        g = grp_rate(view, col)
        g[col] = g[col].map({0: "No", 1: "Yes"})
        g["Product"] = col
        prod_tables.append(g[["Product", col, "Acceptance Rate %", "Customers"]])
    prod_df = pd.concat(prod_tables, ignore_index=True)
    st.dataframe(prod_df, use_container_width=True)
    pivot = prod_df.pivot(index="Product", columns=BINARY_COLS[0] if False else prod_df.columns[1], values="Acceptance Rate %")
    st.markdown('<div class="note">Use this tab to explain which segments accept the loan more often by education, income, age, and product ownership.</div>', unsafe_allow_html=True)

with t2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Correlation with Personal Loan")
        corr_cols = ["Age", "Experience", "Income", "Family", "CCAvg", "Education", "Mortgage", "Securities Account", "CD Account", "Online", "CreditCard", TARGET]
        corr = view[corr_cols].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(ascending=False)
        st.bar_chart(corr)
        st.dataframe(corr.reset_index().rename(columns={"index": "Feature", TARGET: "Correlation"}), use_container_width=True)
    with c2:
        st.subheader("Mean Profile by Outcome")
        mean_df = view.groupby("Loan Label")[["Income", "CCAvg", "Mortgage", "Family"]].mean().round(2)
        st.dataframe(mean_df, use_container_width=True)

    st.subheader("Cross-Sell Opportunity")
    cross_rows = []
    for col in BINARY_COLS:
        g = grp_rate(view, col)
        yes_rate = float(g.loc[g[col] == 1, "Acceptance Rate %"].iloc[0]) if 1 in g[col].values else 0
        no_rate = float(g.loc[g[col] == 0, "Acceptance Rate %"].iloc[0]) if 0 in g[col].values else 0
        cross_rows.append({"Product": col, "Has Product Rate %": round(yes_rate, 2), "No Product Rate %": round(no_rate, 2), "Lift %": round(yes_rate - no_rate, 2)})
    cross_df = pd.DataFrame(cross_rows).sort_values("Lift %", ascending=False)
    st.dataframe(cross_df, use_container_width=True)
    st.bar_chart(cross_df.set_index("Product")[["Lift %"]])

    st.subheader("Chi-Square Significance")
    chi_rows = []
    for col in BINARY_COLS + ["Education"]:
        tab = pd.crosstab(view[col], view[TARGET])
        chi2, p, _, _ = chi2_contingency(tab)
        chi_rows.append({"Feature": col, "Chi-Square": round(chi2, 2), "p-value": round(p, 5)})
    st.dataframe(pd.DataFrame(chi_rows).sort_values("Chi-Square", ascending=False), use_container_width=True)

with t3:
    st.subheader("Model Comparison")
    st.dataframe(metrics, use_container_width=True)

    st.subheader(f"Confusion Matrix — {best_name}")
    cm = pd.DataFrame(fitted[best_name]["cm"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm, use_container_width=True)

    st.subheader(f"Feature Importance — {best_name}")
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False).round(4)
        st.bar_chart(imp.head(12))
        st.dataframe(imp.reset_index().rename(columns={"index": "Feature", 0: "Importance"}).head(12), use_container_width=True)
    st.markdown(f'<div class="note">Best model by ROC AUC: <b>{best_name}</b>.</div>', unsafe_allow_html=True)

with t4:
    leads = scored[scored["Predicted Class"] == 1].copy()
    leads["Persona"] = leads.apply(persona, axis=1)
    leads["Recommended Offer"] = leads.apply(offer, axis=1)
    leads = leads.sort_values("Predicted Probability", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Predicted Acceptors by Persona")
        p = leads["Persona"].value_counts().to_frame("Customers")
        st.bar_chart(p)
        st.dataframe(p.reset_index().rename(columns={"index": "Persona"}), use_container_width=True)
    with c2:
        st.subheader("Top Leads")
        quick = leads[["Age", "Income", "Family", "CCAvg", "Education Label", "Predicted Probability"]].head(10).copy()
        quick["Predicted Probability"] = quick["Predicted Probability"].round(4)
        st.dataframe(quick, use_container_width=True)

    st.subheader("Recommended Campaign Actions")
    for _, row in leads.head(8).iterrows():
        st.markdown(f"<div class='offer'><b>{persona(row)}</b><br>Probability: {row['Predicted Probability']:.1%}<br>{offer(row)}</div>", unsafe_allow_html=True)

    st.subheader("Offer Simulator")
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
        "Age": age, "Experience": experience, "Income": income, "Family": family,
        "CCAvg": ccavg, "Education": education, "Mortgage": mortgage,
        "Securities Account": securities, "CD Account": cd, "Online": online, "CreditCard": credit,
    }])
    prob = float(best_model.predict_proba(sample[feature_cols])[0, 1])
    row = sample.iloc[0].copy()
    st.markdown(f"<div class='offer'><b>Predicted acceptance probability:</b> {prob:.1%}<br><b>Persona:</b> {persona(row)}<br><b>Recommended action:</b> {offer(row)}</div>", unsafe_allow_html=True)

st.caption("This version avoids Plotly completely and keeps the assignment logic: predict Personal Loan, exclude ZIP Code, and recommend cross-sell actions.")
