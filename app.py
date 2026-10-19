
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, roc_auc_score
)

st.set_page_config(page_title="FraudShield | Transaction Analytics", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.main {background-color: #f7f9fc;}
.block-container {padding-top: 1.6rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; color: white;
       background: linear-gradient(120deg,#12233f,#245b87);}
.hero h1 {margin: 0; font-size: 2.15rem;}
.hero p {margin: .4rem 0 0 0; opacity: .9;}
div[data-testid="stMetric"] {background: white; padding: 14px; border: 1px solid #e6eaf0; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🛡️ FraudShield</h1>
<p>Banking Transaction Fraud Detection & Analytics · Educational portfolio project</p>
</div>
""", unsafe_allow_html=True)

st.caption("Demo data is synthetic. Predictions are for learning and must not be used to make real banking decisions.")

@st.cache_data
def make_demo_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    amount = np.round(rng.lognormal(mean=3.3, sigma=1.1, size=n), 2)
    hour = rng.integers(0, 24, n)
    channel = rng.choice(["POS", "Online", "ATM", "Mobile"], n, p=[.30, .35, .15, .20])
    country_mismatch = rng.binomial(1, .035, n)
    failed_attempts = rng.poisson(.22, n)
    new_device = rng.binomial(1, .10, n)
    # Create a small, intentionally imbalanced synthetic fraud label.
    logit = (-5.0 + 0.0008 * amount + 1.25 * (hour <= 4) +
             2.0 * country_mismatch + .7 * failed_attempts + 1.0 * new_device +
             .35 * (channel == "Online"))
    probability = 1 / (1 + np.exp(-logit))
    fraud = rng.binomial(1, probability)
    return pd.DataFrame({
        "amount": amount, "hour": hour, "channel": channel,
        "country_mismatch": country_mismatch, "failed_attempts": failed_attempts,
        "new_device": new_device, "is_fraud": fraud
    })

def clean_dataset(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    required = {"amount", "hour", "channel", "country_mismatch",
                "failed_attempts", "new_device", "is_fraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    for c in ["amount", "hour", "country_mismatch", "failed_attempts", "new_device", "is_fraud"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(required))
    df = df[df["is_fraud"].isin([0, 1])]
    if len(df) < 20:
        raise ValueError("Please provide at least 20 valid rows.")
    df["channel"] = df["channel"].astype(str)
    return df

with st.sidebar:
    st.header("Data source")
    upload = st.file_uploader("Upload transaction CSV", type=["csv"])
    st.markdown("**Required columns**")
    st.code("amount, hour, channel,\ncountry_mismatch, failed_attempts,\nnew_device, is_fraud")
    st.markdown("`is_fraud`: 0 = genuine, 1 = fraud")
    st.divider()
    test_size = st.slider("Test data (%)", 10, 40, 25, 5)
    trees = st.slider("Random forest trees", 50, 300, 120, 10)
    threshold = st.slider("Fraud alert threshold", 0.10, 0.90, 0.50, 0.05)

try:
    raw = pd.read_csv(upload) if upload is not None else make_demo_data()
    data = clean_dataset(raw)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

if data["is_fraud"].nunique() < 2:
    st.error("The dataset must contain both genuine (0) and fraud (1) examples.")
    st.stop()

st.subheader("Transaction overview")
total = len(data)
frauds = int(data["is_fraud"].sum())
fraud_rate = frauds / total * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total transactions", f"{total:,}")
c2.metric("Fraud cases", f"{frauds:,}")
c3.metric("Fraud rate", f"{fraud_rate:.2f}%")
c4.metric("Total transaction value", f"{data['amount'].sum():,.2f}")

left, right = st.columns(2)
with left:
    st.markdown("#### Genuine vs fraud")
    counts = data["is_fraud"].map({0:"Genuine", 1:"Fraud"}).value_counts()
    st.bar_chart(counts)
with right:
    st.markdown("#### Transaction amount distribution")
    st.bar_chart(data.groupby(pd.cut(data["amount"], bins=10, duplicates="drop"), observed=False)["is_fraud"].mean().rename("fraud_rate"))

features = ["amount", "hour", "channel", "country_mismatch", "failed_attempts", "new_device"]
X = data[features]
y = data["is_fraud"].astype(int)
stratify = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42, stratify=stratify
)
numeric = ["amount", "hour", "country_mismatch", "failed_attempts", "new_device"]
categorical = ["channel"]
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(n_estimators=trees, class_weight="balanced",
                                         random_state=42, n_jobs=-1))
])
with st.spinner("Training model and evaluating test data..."):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

st.subheader("Model performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Precision", f"{precision_score(y_test, preds, zero_division=0):.3f}")
m2.metric("Recall", f"{recall_score(y_test, preds, zero_division=0):.3f}")
m3.metric("F1 score", f"{f1_score(y_test, preds, zero_division=0):.3f}")
try:
    auc = roc_auc_score(y_test, probs)
    m4.metric("ROC-AUC", f"{auc:.3f}")
except ValueError:
    m4.metric("ROC-AUC", "N/A")

st.caption("Precision, recall, F1 and ROC-AUC are more informative than accuracy when fraud cases are rare. Metrics depend on the uploaded data and selected threshold.")
cm = confusion_matrix(y_test, preds, labels=[0, 1])
fig, ax = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Genuine", "Fraud"]).plot(ax=ax, colorbar=False)
ax.set_title("Confusion matrix (test set)")
st.pyplot(fig, clear_figure=True)

st.subheader("Score a transaction")
with st.form("transaction_form"):
    a, b, c = st.columns(3)
    amount_in = a.number_input("Transaction amount", min_value=0.0, value=2500.0, step=100.0)
    hour_in = b.slider("Hour of day", 0, 23, 22)
    channel_in = c.selectbox("Channel", ["POS", "Online", "ATM", "Mobile"])
    d, e, f = st.columns(3)
    mismatch_in = d.selectbox("Country mismatch?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    failed_in = e.number_input("Failed attempts", min_value=0, max_value=20, value=1)
    device_in = f.selectbox("New device?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    submitted = st.form_submit_button("Analyze transaction", type="primary")

if submitted:
    sample = pd.DataFrame([{
        "amount": amount_in, "hour": hour_in, "channel": channel_in,
        "country_mismatch": mismatch_in, "failed_attempts": failed_in, "new_device": device_in
    }])
    score = float(model.predict_proba(sample)[0, 1])
    st.metric("Estimated fraud risk score", f"{score:.1%}")
    if score >= threshold:
        st.error("⚠️ Flagged for review by the selected model threshold.")
    else:
        st.success("No alert at the selected threshold. This does not guarantee the transaction is genuine.")

st.subheader("Test-set predictions")
results = X_test.copy().reset_index(drop=True)
results["actual"] = y_test.reset_index(drop=True).map({0:"Genuine", 1:"Fraud"})
results["fraud_probability"] = probs.round(4)
results["predicted"] = pd.Series(preds).map({0:"Genuine", 1:"Fraud"})
results["alert"] = np.where(preds == 1, "Review", "No alert")
st.dataframe(results, use_container_width=True, hide_index=True)
st.download_button("Download scored test transactions", results.to_csv(index=False).encode("utf-8"),
                   file_name="fraud_detection_results.csv", mime="text/csv")

st.markdown("---")
st.caption("Portfolio project only. This demo uses simplified synthetic features and is not validated for real financial transactions.")
