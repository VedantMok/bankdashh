from pathlib import Path
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Universal Bank Loan Dashboard", page_icon="🏦", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
TARGET = "Personal Loan"
DROP_COLS = ["ID", "ZIP Code"]
BINARY_COLS = ["Securities Account", "CD Account", "Online", "CreditCard"]
EDU_MAP = {1: "Undergrad", 2: "Graduate", 3: "Advanced/Professional"}

st.markdown("""
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 1rem;}
.big-title {font-size: 2rem; font-weight: 800; color: #123b5d;}
.subtle {color: #5f6f7f; margin-bottom: 1rem;}
.note {background:#f6fbff; border-left:4px solid #2f80ed; padding:0.9rem; border-radius:8px;}
.offer {background:#f7fcf7; border:1px solid #d7ead8; padding:0.9rem; border-radius:10px; margin-bottom:0.6rem;}
.small {color:#5f6f7f; font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def stratified_split(X, y, test_size=0.25, random_state=42):
    rng = np.random.default_rng(random_state)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n0 = max(1, int(len(idx0) * test_size))
    n1 = max(1, int(len(idx1) * test_size))
    test_idx = np.concatenate([idx0[:n0], idx1[:n1]])
    train_idx = np.concatenate([idx0[n0:], idx1[n1:]])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tp / (tp + fp)) if (tp + fp) else 0.0


def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(tp / (tp + fn)) if (tp + fn) else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_ranks = ranks[y_true == 1].sum()
    auc = (pos_ranks - pos * (pos + 1) / 2) / (pos * neg)
    return float(auc)


def confusion(y_true, y_pred):
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def roc_curve_points(y_true, y_score):
    thresholds = np.unique(np.round(y_score, 6))[::-1]
    thresholds = np.concatenate([[1.01], thresholds, [-0.01]])
    fprs, tprs = [], []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        cm = confusion(y_true, pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0
        tpr = tp / (tp + fn) if (tp + fn) else 0
        fprs.append(fpr)
        tprs.append(tpr)
    return pd.DataFrame({"FPR": fprs, "TPR": tprs})


def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    p = np.mean(y)
    return 1.0 - p * p - (1 - p) * (1 - p)


def mse_impurity(y):
    if len(y) == 0:
        return 0.0
    m = np.mean(y)
    return float(np.mean((y - m) ** 2))


def candidate_thresholds(col):
    vals = np.unique(col)
    if len(vals) <= 12:
        mids = (vals[:-1] + vals[1:]) / 2
        return mids
    qs = np.unique(np.quantile(col, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
    return qs


class TreeNode:
    def __init__(self, value=None, prob=None, feature=None, threshold=None, left=None, right=None):
        self.value = value
        self.prob = prob
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class DecisionTreeCustom:
    def __init__(self, max_depth=4, min_samples_leaf=20, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.n_features_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self._rng = np.random.default_rng(self.random_state)
        self.root = self._build(X, y, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ = self.feature_importances_ / total
        return self

    def _feature_subset(self):
        idx = np.arange(self.n_features_)
        if self.max_features is None or self.max_features >= self.n_features_:
            return idx
        return self._rng.choice(idx, size=self.max_features, replace=False)

    def _best_split(self, X, y):
        parent_imp = gini_impurity(y)
        best_gain, best_f, best_t = 0.0, None, None
        feats = self._feature_subset()
        for f in feats:
            col = X[:, f]
            for t in candidate_thresholds(col):
                left = col <= t
                right = ~left
                if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
                    continue
                gain = parent_imp - (left.mean() * gini_impurity(y[left]) + right.mean() * gini_impurity(y[right]))
                if gain > best_gain:
                    best_gain, best_f, best_t = gain, f, float(t)
        return best_gain, best_f, best_t

    def _build(self, X, y, depth):
        prob = float(np.mean(y)) if len(y) else 0.0
        value = int(prob >= 0.5)
        if depth >= self.max_depth or len(y) < self.min_samples_leaf * 2 or len(np.unique(y)) == 1:
            return TreeNode(value=value, prob=prob)
        gain, feat, thr = self._best_split(X, y)
        if feat is None or gain <= 1e-10:
            return TreeNode(value=value, prob=prob)
        self.feature_importances_[feat] += gain * len(y)
        mask = X[:, feat] <= thr
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return TreeNode(value=value, prob=prob, feature=feat, threshold=thr, left=left, right=right)

    def _predict_one(self, x, node):
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.prob

    def predict_proba(self, X):
        probs = np.array([self._predict_one(x, self.root) for x in X])
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RegressionTreeCustom:
    def __init__(self, max_depth=2, min_samples_leaf=20, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.n_features_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self._rng = np.random.default_rng(self.random_state)
        self.root = self._build(X, y, 0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ = self.feature_importances_ / total
        return self

    def _feature_subset(self):
        idx = np.arange(self.n_features_)
        if self.max_features is None or self.max_features >= self.n_features_:
            return idx
        return self._rng.choice(idx, size=self.max_features, replace=False)

    def _best_split(self, X, y):
        parent_imp = mse_impurity(y)
        best_gain, best_f, best_t = 0.0, None, None
        feats = self._feature_subset()
        for f in feats:
            col = X[:, f]
            for t in candidate_thresholds(col):
                left = col <= t
                right = ~left
                if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
                    continue
                gain = parent_imp - (left.mean() * mse_impurity(y[left]) + right.mean() * mse_impurity(y[right]))
                if gain > best_gain:
                    best_gain, best_f, best_t = gain, f, float(t)
        return best_gain, best_f, best_t

    def _build(self, X, y, depth):
        val = float(np.mean(y)) if len(y) else 0.0
        if depth >= self.max_depth or len(y) < self.min_samples_leaf * 2:
            return TreeNode(value=val)
        gain, feat, thr = self._best_split(X, y)
        if feat is None or gain <= 1e-10:
            return TreeNode(value=val)
        self.feature_importances_[feat] += gain * len(y)
        mask = X[:, feat] <= thr
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return TreeNode(value=val, feature=feat, threshold=thr, left=left, right=right)

    def _predict_one(self, x, node):
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


class RandomForestCustom:
    def __init__(self, n_estimators=25, max_depth=5, min_samples_leaf=20, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, m = X.shape
        self.trees = []
        importances = np.zeros(m)
        max_feats = max(1, int(np.sqrt(m)))
        for i in range(self.n_estimators):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            tree = DecisionTreeCustom(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, max_features=max_feats, random_state=self.random_state + i + 1)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
            importances += tree.feature_importances_
        self.feature_importances_ = importances / self.n_estimators
        return self

    def predict_proba(self, X):
        probs = np.mean([t.predict_proba(X)[:, 1] for t in self.trees], axis=0)
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class GradientBoostingCustom:
    def __init__(self, n_estimators=35, learning_rate=0.08, max_depth=2, min_samples_leaf=25, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []
        self.base_score = 0.0
        self.feature_importances_ = None

    def fit(self, X, y):
        p = np.clip(np.mean(y), 1e-5, 1 - 1e-5)
        self.base_score = math.log(p / (1 - p))
        F = np.full(len(y), self.base_score)
        self.trees = []
        importances = np.zeros(X.shape[1])
        for i in range(self.n_estimators):
            prob = sigmoid(F)
            residual = y - prob
            tree = RegressionTreeCustom(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=self.random_state + i + 1)
            tree.fit(X, residual)
            update = tree.predict(X)
            F += self.learning_rate * update
            self.trees.append(tree)
            importances += tree.feature_importances_
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.base_score)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        probs = sigmoid(F)
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def chi_square_stat(df, feature, target):
    tab = pd.crosstab(df[feature], df[target]).values.astype(float)
    total = tab.sum()
    row_sum = tab.sum(axis=1, keepdims=True)
    col_sum = tab.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    expected[expected == 0] = 1e-9
    chi2 = float(((tab - expected) ** 2 / expected).sum())
    return round(chi2, 2)


def find_csv():
    for name in ["UniversalBank.csv", "universalbank.csv"]:
        p = BASE_DIR / name
        if p.exists():
            return p
    csvs = list(BASE_DIR.glob("*.csv"))
    return csvs[0] if csvs else None


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
def train_all(df):
    feature_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET, "Education Label", "Loan Label", "Income Band", "Age Band", "CCAvg Band"]]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test, train_idx, test_idx = stratified_split(X, y, test_size=0.25, random_state=42)

    models = {
        "Decision Tree": DecisionTreeCustom(max_depth=4, min_samples_leaf=30, random_state=42),
        "Random Forest": RandomForestCustom(n_estimators=25, max_depth=5, min_samples_leaf=25, random_state=42),
        "Gradient Boosting": GradientBoostingCustom(n_estimators=35, learning_rate=0.08, max_depth=2, min_samples_leaf=30, random_state=42),
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy(y_test, pred), 4),
            "Precision": round(precision(y_test, pred), 4),
            "Recall": round(recall(y_test, pred), 4),
            "F1": round(f1_score(y_test, pred), 4),
            "ROC AUC": round(roc_auc(y_test, proba), 4),
        })
        fitted[name] = {
            "model": model,
            "pred": pred,
            "proba": proba,
            "cm": confusion(y_test, pred),
            "roc": roc_curve_points(y_test, proba),
        }

    metrics = pd.DataFrame(rows).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    best_model = fitted[best_name]["model"]
    full = df.copy()
    full_probs = best_model.predict_proba(X)[:, 1]
    full["Predicted Probability"] = full_probs
    full["Predicted Class"] = (full_probs >= 0.5).astype(int)
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
feature_cols, metrics, fitted, best_name, best_model, scored = train_all(df)

st.sidebar.markdown("### Filters")
income_opts = [x for x in df["Income Band"].dropna().unique()]
edu_opts = sorted(df["Education Label"].dropna().unique())
loan_opts = ["No", "Yes"]
income_sel = st.sidebar.multiselect("Income Band", income_opts, default=income_opts)
edu_sel = st.sidebar.multiselect("Education", edu_opts, default=edu_opts)
loan_sel = st.sidebar.multiselect("Personal Loan", loan_opts, default=loan_opts)

view = df[df["Income Band"].isin(income_sel) & df["Education Label"].isin(edu_sel) & df["Loan Label"].isin(loan_sel)].copy()

st.markdown('<div class="big-title">🏦 Universal Bank Personal Loan Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Dependency-light version: no Plotly, no scikit-learn, no scipy. Everything runs with only Streamlit, pandas, and numpy.</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Customers", f"{len(view):,}")
k2.metric("Loan Acceptance", f"{view[TARGET].mean()*100:.1f}%")
k3.metric("Avg Income ($000)", f"{view['Income'].mean():.1f}")
k4.metric("Avg CCAvg ($000)", f"{view['CCAvg'].mean():.2f}")
k5.metric("CD Account Share", f"{view['CD Account'].mean()*100:.1f}%")

t1, t2, t3, t4 = st.tabs(["Descriptive", "Diagnostic", "Predictive", "Prescriptive"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Loan Split")
        split = view["Loan Label"].value_counts().rename_axis("Outcome").to_frame("Customers")
        st.bar_chart(split)
        st.dataframe(split.reset_index(), use_container_width=True)
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
    st.markdown('<div class="note">Use this tab to explain who is more likely to accept the loan by age, income, education, and existing products.</div>', unsafe_allow_html=True)

with t2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Correlation with Personal Loan")
        corr_cols = ["Age", "Experience", "Income", "Family", "CCAvg", "Education", "Mortgage", "Securities Account", "CD Account", "Online", "CreditCard", TARGET]
        corr = view[corr_cols].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(ascending=False)
        st.bar_chart(corr)
        corr_df = corr.reset_index()
        corr_df.columns = ["Feature", "Correlation"]
        st.dataframe(corr_df, use_container_width=True)
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

    st.subheader("Chi-Square Scores")
    chi_rows = []
    for col in BINARY_COLS + ["Education"]:
        chi_rows.append({"Feature": col, "Chi-Square": chi_square_stat(view, col, TARGET)})
    st.dataframe(pd.DataFrame(chi_rows).sort_values("Chi-Square", ascending=False), use_container_width=True)

with t3:
    st.subheader("Model Comparison")
    st.dataframe(metrics, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Confusion Matrix — {best_name}")
        cm = pd.DataFrame(fitted[best_name]["cm"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm, use_container_width=True)
    with c2:
        st.subheader("ROC Curve Points")
        st.line_chart(fitted[best_name]["roc"].set_index("FPR"))

    st.subheader(f"Feature Importance — {best_name}")
    imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False).round(4)
    st.bar_chart(imp.head(12))
    imp_df = imp.reset_index()
    imp_df.columns = ["Feature", "Importance"]
    st.dataframe(imp_df.head(12), use_container_width=True)
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
        st.subheader("Top Likely Acceptors")
        quick = leads[["Age", "Income", "Family", "CCAvg", "Education Label", "Predicted Probability"]].head(15).copy()
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
    prob = float(best_model.predict_proba(sample[feature_cols].to_numpy(dtype=float))[:, 1][0])
    row = sample.iloc[0].copy()
    st.markdown(f"<div class='offer'><b>Predicted acceptance probability:</b> {prob:.1%}<br><b>Persona:</b> {persona(row)}<br><b>Recommended action:</b> {offer(row)}</div>", unsafe_allow_html=True)

st.caption("This version avoids Plotly, scikit-learn, and scipy. It still covers descriptive, diagnostic, predictive, and prescriptive analytics for the assignment.")
