import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Title
st.title("Hybrid PONV Machine Learning Algorithm [Alpha 2.0]")
st.subheader("Department of Pharmacology, MKCG Medical College and Hospital")
st.markdown("<small>Made by- Dr Nabyendu Biswas</small>", unsafe_allow_html=True)

# Initialize session state for logging
if 'log_data' not in st.session_state:
    st.session_state.log_data = []

# User Inputs
st.sidebar.header("This hybrid model is designed based on multiple PONV risk scores, including Apfel, Koivuranta, and Bellville scores, in alignment with the POTTER app developed by Massachusetts Medical School, in collaboration with the Department of Anaesthesiology, MKCG Medical College & Hospital.")

# Patient-Specific Risk Factors
st.sidebar.subheader("Patient-Specific Risk Factors")
gender = st.sidebar.selectbox("Female gender", ["Male", "Female"])
smoker = st.sidebar.selectbox("Smoker", ["Yes", "No"])
history_ponv = st.sidebar.selectbox("History of PONV or Motion Sickness", ["Yes", "No"])
age = st.sidebar.slider("Age", 18, 80, 40)
preop_anxiety = st.sidebar.selectbox("Preoperative Anxiety", ["Yes", "No"])

# Surgical & Anesthetic Risk Factors
st.sidebar.subheader("Surgical & Anesthetic Risk Factors")
abdominal_surgery = st.sidebar.selectbox("Abdominal or Laparoscopic Surgery", ["Yes", "No"])
volatile_anesthetics = st.sidebar.selectbox("Use of Volatile Anesthetics", ["Yes", "No"])
nitrous_oxide = st.sidebar.selectbox("Use of Nitrous Oxide", ["Yes", "No"])

# Drug-Related Risk Factors
st.sidebar.subheader("Drug-Related Risk Factors")
midazolam = st.sidebar.selectbox("Midazolam given at induction", ["Yes", "No"])
ondansetron = st.sidebar.selectbox("Ondansetron given", ["Yes", "No"])
dexamethasone = st.sidebar.selectbox("Dexamethasone given at induction", ["Yes", "No"])

# Opioids and Pain Management Drugs
st.sidebar.subheader("Opioids and Pain Management Drugs")
opioid = st.sidebar.selectbox("Opioid Used", ["None", "Nalbuphine", "Fentanyl", "Butorphanol", "Pentazocine"])

# Induction & Maintenance Agents
st.sidebar.subheader("Induction & Maintenance Agents")
induction_agent = st.sidebar.selectbox("Induction Agent", ["Propofol (TIVA used)", "Propofol (Induction Only, then Inhalational)", "Sevoflurane / Isoflurane / Desflurane"])
muscle_relaxant = st.sidebar.selectbox("Muscle Relaxant Used", ["None", "Atracurium", "Cisatracurium", "Vecuronium", "Succinylcholine"])

# Postoperative Symptoms
st.sidebar.subheader("Postoperative Symptoms (SAND Score Integration)")
nausea_duration = st.sidebar.slider("Nausea lasting >30 minutes", 0, 1, 0)
vomiting_episodes = st.sidebar.slider("More than 2 episodes of vomiting", 0, 1, 0)
abdominal_discomfort = st.sidebar.selectbox("Abdominal Discomfort", ["Yes", "No"])

# Convert inputs
def convert_to_numeric(value):
    return 1 if value == "Yes" else 0

gender = 1 if gender == "Female" else 0
smoker = 1 if smoker == "No" else 0
history_ponv = convert_to_numeric(history_ponv)
preop_anxiety = convert_to_numeric(preop_anxiety)
abdominal_surgery = convert_to_numeric(abdominal_surgery)
volatile_anesthetics = convert_to_numeric(volatile_anesthetics)
nitrous_oxide = convert_to_numeric(nitrous_oxide)
midazolam = convert_to_numeric(midazolam)
ondansetron = convert_to_numeric(ondansetron)
dexamethasone = convert_to_numeric(dexamethasone)
abdominal_discomfort = convert_to_numeric(abdominal_discomfort)

# Calculate Hybrid Score
hybrid_score = (
    gender + smoker + history_ponv * 2 + (1 if age < 50 else 0) + preop_anxiety +
    abdominal_surgery * 2 + volatile_anesthetics * 2 + nitrous_oxide * 3 +
    (-2 if midazolam else 0) + (-2 if ondansetron else 0) + (-1 if dexamethasone else 0) +
    (1 if opioid in ["Nalbuphine", "Butorphanol"] else 3 if opioid in ["Fentanyl", "Pentazocine"] else 0)
)

st.write(f"### Calculated Hybrid Score: {hybrid_score}")

# Log Entry
if st.button("📌 Log This Entry"):
    new_entry = {
        "Gender": "Female" if gender else "Male",
        "Smoker": "No" if smoker else "Yes",
        "History PONV": "Yes" if history_ponv else "No",
        "Age": age,
        "Anxiety": "Yes" if preop_anxiety else "No",
        "Abdominal Surgery": "Yes" if abdominal_surgery else "No",
        "Volatile": "Yes" if volatile_anesthetics else "No",
        "N2O": "Yes" if nitrous_oxide else "No",
        "Midazolam": "Yes" if midazolam else "No",
        "Ondansetron": "Yes" if ondansetron else "No",
        "Dexamethasone": "Yes" if dexamethasone else "No",
        "Opioid": opioid,
        "Hybrid Score": hybrid_score
    }
    st.session_state.log_data.append(new_entry)
    st.success("Entry logged successfully!")

# Show table of all entries
if st.session_state.log_data:
    st.markdown("### 📋 Logged Entries")
    df_log = pd.DataFrame(st.session_state.log_data)
    st.dataframe(df_log)

    # Download CSV
    csv = df_log.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download All Entries as CSV",
        data=csv,
        file_name='logged_ponv_entries.csv',
        mime='text/csv'
    )

# Model Training Info
st.markdown("<small><b>Model Training Information:</b> Training has been done by a synthetic dataset of 1000 entries...</small>", unsafe_allow_html=True)

# Generate Synthetic Data
np.random.seed(42)
X_synthetic = np.random.rand(1000, 10)
y_synthetic = np.random.randint(0, 2, 1000)
X_train, X_validate, y_train, y_validate = train_test_split(X_synthetic, y_synthetic, test_size=0.3, random_state=42)

# Models
svc_model = LinearSVC(max_iter=10000)
svc_calibrated = CalibratedClassifierCV(svc_model, method='sigmoid')
svc_calibrated.fit(X_train, y_train)

adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost_model.fit(X_train, y_train)

# Predictions
y_prob_svc_train = svc_calibrated.predict_proba(X_train)[:, 1]
y_prob_svc_validate = svc_calibrated.predict_proba(X_validate)[:, 1]
y_prob_adaboost_train = adaboost_model.predict_proba(X_train)[:, 1]
y_prob_adaboost_validate = adaboost_model.predict_proba(X_validate)[:, 1]

# Binary Predictions
y_pred_svc_validate = svc_calibrated.predict(X_validate)
y_pred_adaboost_validate = adaboost_model.predict(X_validate)

# Classification Metrics
acc_svc = accuracy_score(y_validate, y_pred_svc_validate)
prec_svc = precision_score(y_validate, y_pred_svc_validate)
rec_svc = recall_score(y_validate, y_pred_svc_validate)
f1_svc = f1_score(y_validate, y_pred_svc_validate)

acc_ada = accuracy_score(y_validate, y_pred_adaboost_validate)
prec_ada = precision_score(y_validate, y_pred_adaboost_validate)
rec_ada = recall_score(y_validate, y_pred_adaboost_validate)
f1_ada = f1_score(y_validate, y_pred_adaboost_validate)

# Display classification metrics
st.markdown("### 📊 Classification Performance (Validation Set)")
metrics_df = pd.DataFrame({
    "Model": ["LinearSVC", "AdaBoost"],
    "Accuracy": [acc_svc, acc_ada],
    "Precision": [prec_svc, prec_ada],
    "Recall": [rec_svc, rec_ada],
    "F1 Score": [f1_svc, f1_ada]
})
numeric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
st.dataframe(metrics_df.style.format({col: "{:.2f}" for col in numeric_cols}))

# ROC Curves
fpr_svc_train, tpr_svc_train, _ = roc_curve(y_train, y_prob_svc_train)
roc_auc_svc_train = auc(fpr_svc_train, tpr_svc_train)
fpr_adaboost_train, tpr_adaboost_train, _ = roc_curve(y_train, y_prob_adaboost_train)
roc_auc_adaboost_train = auc(fpr_adaboost_train, tpr_adaboost_train)
fpr_svc_validate, tpr_svc_validate, _ = roc_curve(y_validate, y_prob_svc_validate)
roc_auc_svc_validate = auc(fpr_svc_validate, tpr_svc_validate)
fpr_adaboost_validate, tpr_adaboost_validate, _ = roc_curve(y_validate, y_prob_adaboost_validate)
roc_auc_adaboost_validate = auc(fpr_adaboost_validate, tpr_adaboost_validate)

# AUC Scores Display
st.markdown(f"""
    <div style="display: flex; justify-content: space-around;">
        <div>
            <p style="font-size: 14px;"><b>Training AUC Scores</b></p>
            <p style="font-size: 12px;">LinearSVC: {roc_auc_svc_train:.3f}</p>
            <p style="font-size: 12px;">AdaBoost: {roc_auc_adaboost_train:.3f}</p>
        </div>
        <div>
            <p style="font-size: 14px;"><b>Validation AUC Scores</b></p>
            <p style="font-size: 12px;">LinearSVC: {roc_auc_svc_validate:.3f}</p>
            <p style="font-size: 12px;">AdaBoost: {roc_auc_adaboost_validate:.3f}</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ROC Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(fpr_svc_train, tpr_svc_train, label=f"LinearSVC (AUC = {roc_auc_svc_train:.3f})")
ax1.plot(fpr_adaboost_train, tpr_adaboost_train, label=f"AdaBoost (AUC = {roc_auc_adaboost_train:.3f})")
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Training ROC Curve')
ax1.legend(loc="lower right")

ax2.plot(fpr_svc_validate, tpr_svc_validate, label=f"LinearSVC (AUC = {roc_auc_svc_validate:.3f})")
ax2.plot(fpr_adaboost_validate, tpr_adaboost_validate, label=f"AdaBoost (AUC = {roc_auc_adaboost_validate:.3f})")
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Validation ROC Curve')
ax2.legend(loc="lower right")

st.pyplot(fig)
