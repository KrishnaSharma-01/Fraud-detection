import streamlit as st
import pandas as pd
import joblib
import os

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("Fraud Detection Prediction App")
st.write("Enter transaction details to predict if the transaction is fraudulent.")

# ----------------------------
# Load model & scaler
# ----------------------------
MODEL_PATH = "final_model.pkl"
SCALER_PATH = "scaler.pkl"
DATA_HEADER_PATH = "processed_fraud_data.csv"  # used only to discover column order

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ----------------------------
# Get training feature order
# Prefer reading from processed_fraud_data.csv header.
# Fall back to a known list if the CSV isn't available.
# ----------------------------
fallback_features = [
    'step','amount','oldbalanceOrg','newbalanceOrig',
    'oldbalanceDest','newbalanceDest',
    'type_TRANSFER','type_CASH_OUT','type_DEBIT','type_PAYMENT',
    'day','hour','error_balance_orig','error_balance_dest',
    'isMerchant','high_value'
]

if os.path.exists(DATA_HEADER_PATH):
    try:
        hdr = pd.read_csv(DATA_HEADER_PATH, nrows=1)
        feature_order = [c for c in hdr.columns if c != "isFraud"]
    except Exception:
        feature_order = fallback_features
else:
    feature_order = fallback_features

# ----------------------------
# Inputs
# ----------------------------
with st.form("fraud_form"):
    step = st.number_input("Step (1..744)", min_value=1, max_value=744, value=1)
    txn_type = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0, step=100.0)

    oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=5000.0, step=100.0)
    newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, value=4000.0, step=100.0)

    oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, value=0.0, step=100.0)
    newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, value=1000.0, step=100.0)

    # If your training created isMerchant from nameDest starting with 'M',
    # keep this as a user input; otherwise you can change the rule below.
    isMerchant = st.selectbox("Is Destination a Merchant? (dataset rule)", [0, 1])

    submitted = st.form_submit_button("Predict")

# ----------------------------
# Helpers: build features exactly like training
# ----------------------------
def build_features_row(step, txn_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isMerchant):
    # Time features from step
    hour = step % 24
    day = step // 24

    # Engineered balances (as in notebook)
    error_balance_orig = oldbalanceOrg - amount - newbalanceOrig
    error_balance_dest = newbalanceDest - oldbalanceDest - amount

    # High value flag
    high_value = int(amount > 200000)

    # One-hot for type (we dropped CASH_IN during training with drop_first)
    type_TRANSFER = int(txn_type == "TRANSFER")
    type_CASH_OUT = int(txn_type == "CASH_OUT")
    type_DEBIT = int(txn_type == "DEBIT")
    type_PAYMENT = int(txn_type == "PAYMENT")
    # No 'type_CASH_IN' column by design.

    row = {
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'type_TRANSFER': type_TRANSFER,
        'type_CASH_OUT': type_CASH_OUT,
        'type_DEBIT': type_DEBIT,
        'type_PAYMENT': type_PAYMENT,
        'day': day,
        'hour': hour,
        'error_balance_orig': error_balance_orig,
        'error_balance_dest': error_balance_dest,
        'isMerchant': int(isMerchant),
        'high_value': high_value
    }
    return pd.DataFrame([row])

# ----------------------------
# Predict
# ----------------------------
if submitted:
    try:
        input_df = build_features_row(
            step, txn_type, amount,
            oldbalanceOrg, newbalanceOrig,
            oldbalanceDest, newbalanceDest,
            isMerchant
        )

        # Align to training columns exactly: add missing as 0, drop extras, correct order
        input_df = input_df.reindex(columns=feature_order, fill_value=0)

        # Scale & predict
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader("Prediction Result")
        if pred == 1:
            if proba is not None:
                st.error(f"Fraudulent transaction detected. (Fraud probability: {proba:.2f})")
            else:
                st.error("Fraudulent transaction detected.")
        else:
            if proba is not None:
                st.success(f"Legitimate transaction. (Fraud probability: {proba:.2f})")
            else:
                st.success("Legitimate transaction.")

        with st.expander("Debug: features sent to model"):
            st.write("Column order used:", feature_order)
            st.dataframe(input_df)

    except Exception as e:
        st.error("An error occurred during prediction:")
        st.exception(e)


