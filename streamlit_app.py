import streamlit as st
import requests

API_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🔮",
    layout="wide",
)

# ── styles ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-high   { background:#ff4b4b22; border-left:4px solid #ff4b4b;
                   padding:16px; border-radius:8px; }
    .risk-medium { background:#ffa50022; border-left:4px solid #ffa500;
                   padding:16px; border-radius:8px; }
    .risk-low    { background:#00c85322; border-left:4px solid #00c853;
                   padding:16px; border-radius:8px; }
    .metric-card { background:#1e1e2e; padding:20px; border-radius:12px;
                   text-align:center; }
    .section-title { font-size:13px; color:#888; text-transform:uppercase;
                     letter-spacing:1px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ── header ───────────────────────────────────────────────────────────────────
st.title("🔮 Customer Churn Predictor")
st.caption("fill in the customer details and find out if theyre gonna leave")

# check if api is up
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health.get("model_loaded"):
        st.success("API connected · model loaded", icon="✅")
    else:
        st.warning("API up but model not loaded yet — run train.py first", icon="⚠️")
except Exception:
    st.error("cant reach the API — make sure uvicorn is running on port 8000", icon="🚨")

st.divider()

# ── input form ───────────────────────────────────────────────────────────────
st.subheader("Customer Info")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p class="section-title">account</p>', unsafe_allow_html=True)
    tenure         = st.slider("Tenure (months)", 0, 72, 12)
    monthly        = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
    total          = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                     round(monthly * tenure, 2), step=10.0)

with col2:
    st.markdown('<p class="section-title">demographics</p>', unsafe_allow_html=True)
    gender         = st.selectbox("Gender", ["Female", "Male"])
    senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Has Partner", ["No", "Yes"])
    dependents     = st.selectbox("Has Dependents", ["No", "Yes"])

with col3:
    st.markdown('<p class="section-title">plan & contract</p>', unsafe_allow_html=True)
    contract       = st.selectbox("Contract Type",
                                  ["Month-to-month", "One year", "Two year"])
    internet       = st.selectbox("Internet Service",
                                  ["DSL", "Fiber optic", "No"])
    payment        = st.selectbox("Payment Method",
                                  ["Electronic check", "Mailed check",
                                   "Bank transfer (automatic)",
                                   "Credit card (automatic)"])
    phone          = st.selectbox("Phone Service", ["Yes", "No"])
    paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])

st.divider()

predict_btn = st.button("Predict Churn →", type="primary", use_container_width=True)

# ── prediction ───────────────────────────────────────────────────────────────
if predict_btn:
    payload = {
        "tenure":                        tenure,
        "MonthlyCharges":                monthly,
        "TotalCharges":                  total,
        "gender":                        1 if gender == "Male" else 0,
        "SeniorCitizen":                 1 if senior == "Yes" else 0,
        "Partner":                       1 if partner == "Yes" else 0,
        "Dependents":                    1 if dependents == "Yes" else 0,
        "PhoneService":                  1 if phone == "Yes" else 0,
        "PaperlessBilling":              1 if paperless == "Yes" else 0,
        "Contract_One_year":             1 if contract == "One year" else 0,
        "Contract_Two_year":             1 if contract == "Two year" else 0,
        "InternetService_Fiber_optic":   1 if internet == "Fiber optic" else 0,
        "InternetService_No":            1 if internet == "No" else 0,
        "PaymentMethod_Credit_card":     1 if "Credit card" in payment else 0,
        "PaymentMethod_Electronic_check":1 if payment == "Electronic check" else 0,
        "PaymentMethod_Mailed_check":    1 if payment == "Mailed check" else 0,
    }

    with st.spinner("running prediction..."):
        try:
            res = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            res.raise_for_status()
            data = res.json()
        except requests.exceptions.ConnectionError:
            st.error("lost connection to the API")
            st.stop()
        except Exception as e:
            st.error(f"something broke: {e}")
            st.stop()

    # ── results layout ───────────────────────────────────────────────────────
    st.subheader("Prediction Results")

    r1, r2, r3 = st.columns(3)

    prob_pct = round(data["churn_probability"] * 100, 1)
    risk     = data["risk_level"]
    churning = data["will_churn"]

    with r1:
        st.metric(
            label="Churn Probability",
            value=f"{prob_pct}%",
            delta="high risk" if churning else "low risk",
            delta_color="inverse",
        )

    with r2:
        st.metric(
            label="Will Churn?",
            value="Yes 🚨" if churning else "No ✅",
        )

    with r3:
        st.metric(
            label="Risk Level",
            value=risk,
        )

    # risk banner
    risk_class = f"risk-{risk.lower()}"
    risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[risk]
    risk_msg   = {
        "HIGH":   "this customer is very likely to leave — act now",
        "MEDIUM": "this customer might leave — worth keeping an eye on",
        "LOW":    "this customer is probably staying, looking good",
    }[risk]

    st.markdown(
        f'<div class="{risk_class}"><strong>{risk_emoji} {risk} RISK</strong>'
        f"<br>{risk_msg}</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # probability bar
    st.markdown("**Churn Probability**")
    bar_color = "#ff4b4b" if risk == "HIGH" else "#ffa500" if risk == "MEDIUM" else "#00c853"
    st.progress(data["churn_probability"])

    # top factors
    st.markdown("**Top Factors Driving This Prediction**")
    factors = data.get("top_churn_factors", [])
    if factors:
        max_score = factors[0][1] if factors else 1
        for name, score in factors:
            clean_name = name.replace("_", " ").replace("  ", " ").title()
            bar_val    = score / max_score if max_score > 0 else 0
            st.markdown(f"`{clean_name}`")
            st.progress(bar_val)

    st.caption(f"model used: {data.get('model_used', 'N/A')}")

# ── sidebar: model info ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Info")
    try:
        info = requests.get(f"{API_URL}/model-info", timeout=3).json()
        st.markdown(f"**Model:** {info['model_name']}")
        m = info["metrics"]
        st.markdown(f"**F1 Score:** `{m['f1']}`")
        st.markdown(f"**Precision:** `{m['precision']}`")
        st.markdown(f"**Recall:** `{m['recall']}`")
        st.markdown(f"**ROC-AUC:** `{m['roc_auc']}`")

        st.divider()
        st.markdown("**Top Churn Factors**")
        for feat, score in info["top_churn_factors"][:7]:
            clean = feat.replace("_", " ").title()
            st.markdown(f"• {clean} `{score}`")
    except Exception:
        st.info("start the API to see model info")

    st.divider()
    st.caption("Customer Churn Prediction Platform")
