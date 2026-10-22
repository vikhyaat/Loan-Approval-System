import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

DATA_PATH = "loan_approval_data.csv"
TARGET = "Loan_Approved"
NUMERIC_COLS = [
    "Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
    "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
    "Collateral_Value", "Loan_Amount", "Loan_Term",
]
CATEGORICAL_COLS = [
    "Employment_Status", "Marital_Status", "Loan_Purpose",
    "Property_Area", "Education_Level", "Gender", "Employer_Category",
]
FEATURE_COLS = NUMERIC_COLS + ["DTI_Ratio_sq", "Credit_Score_sq"] + CATEGORICAL_COLS

def add_features(frame):
    frame = frame.copy()
    frame["DTI_Ratio_sq"] = pd.to_numeric(frame["DTI_Ratio"], errors="coerce") ** 2
    frame["Credit_Score_sq"] = pd.to_numeric(frame["Credit_Score"], errors="coerce") ** 2
    return frame

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Do not train on rows where the target is missing.
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET].isin(["Yes", "No"])]
    return df

@st.cache_resource
def train_model():
    df = load_data()
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    X = add_features(X)
    y = (df[TARGET].str.lower() == "yes").astype(int)

    numeric_features = NUMERIC_COLS + ["DTI_Ratio_sq", "Credit_Score_sq"]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, CATEGORICAL_COLS),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "confusion": confusion_matrix(y_test, pred),
        "test_size": len(y_test),
    }
    # Train final predictor on all labelled rows after evaluation.
    model.fit(X, y)
    return model, metrics, df

try:
    model, metrics, df = train_model()
except Exception as e:
    st.error(f"Could not load the dataset or train the model: {e}")
    st.stop()

st.title("🏦 Loan Approval Prediction")
st.caption("A supervised machine-learning demo using Gaussian Naive Bayes.")
st.info("This app is for learning and demonstration only. Its output is not a real bank decision or a guarantee of approval.")

tab_predict, tab_dashboard, tab_about = st.tabs(["🔮 Predict a Loan", "📊 Dataset & Model", "ℹ️ About"])

with tab_predict:
    st.subheader("Applicant details")
    st.write("Fill in the details below and select **Predict loan approval**.")

    with st.form("loan_prediction_form"):
        st.markdown("#### Financial information")
        c1, c2, c3 = st.columns(3)
        with c1:
            applicant_income = st.number_input("Applicant monthly income", min_value=0.0, value=15000.0, step=500.0)
            coapplicant_income = st.number_input("Co-applicant monthly income", min_value=0.0, value=5000.0, step=500.0)
            loan_amount = st.number_input("Loan amount", min_value=0.0, value=30000.0, step=1000.0)
            loan_term = st.number_input("Loan term (months)", min_value=1, max_value=600, value=36)
        with c2:
            credit_score = st.number_input("Credit score", min_value=0, max_value=900, value=700)
            dti_ratio = st.number_input("DTI ratio (0 to 1)", min_value=0.0, max_value=1.0, value=0.25, step=0.01, format="%.2f")
            savings = st.number_input("Savings", min_value=0.0, value=20000.0, step=1000.0)
            collateral_value = st.number_input("Collateral value", min_value=0.0, value=50000.0, step=1000.0)
        with c3:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            dependents = st.number_input("Number of dependents", min_value=0, max_value=20, value=1)
            existing_loans = st.number_input("Number of existing loans", min_value=0, max_value=20, value=0)

        st.markdown("#### Personal and employment information")
        c4, c5 = st.columns(2)
        with c4:
            employment_status = st.selectbox("Employment status", sorted(df["Employment_Status"].dropna().astype(str).unique()))
            marital_status = st.selectbox("Marital status", sorted(df["Marital_Status"].dropna().astype(str).unique()))
            education_level = st.selectbox("Education level", sorted(df["Education_Level"].dropna().astype(str).unique()))
            gender = st.selectbox("Gender", sorted(df["Gender"].dropna().astype(str).unique()))
        with c5:
            loan_purpose = st.selectbox("Loan purpose", sorted(df["Loan_Purpose"].dropna().astype(str).unique()))
            property_area = st.selectbox("Property area", sorted(df["Property_Area"].dropna().astype(str).unique()))
            employer_category = st.selectbox("Employer category", sorted(df["Employer_Category"].dropna().astype(str).unique()))

        submitted = st.form_submit_button("🔍 Predict Loan Approval", type="primary", use_container_width=True)

    if submitted:
        applicant = pd.DataFrame([{
            "Applicant_Income": applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Age": age,
            "Dependents": dependents,
            "Credit_Score": credit_score,
            "Existing_Loans": existing_loans,
            "DTI_Ratio": dti_ratio,
            "Savings": savings,
            "Collateral_Value": collateral_value,
            "Loan_Amount": loan_amount,
            "Loan_Term": loan_term,
            "Employment_Status": employment_status,
            "Marital_Status": marital_status,
            "Loan_Purpose": loan_purpose,
            "Property_Area": property_area,
            "Education_Level": education_level,
            "Gender": gender,
            "Employer_Category": employer_category,
        }])
        applicant = add_features(applicant)
        prediction = int(model.predict(applicant[FEATURE_COLS])[0])
        probabilities = model.predict_proba(applicant[FEATURE_COLS])[0]
        class_index = list(model.named_steps["classifier"].classes_).index(1)
        probability = float(probabilities[class_index])

        st.divider()
        st.subheader("Prediction result")
        if prediction == 1:
            st.success("✅ Predicted status: LOAN APPROVED")
        else:
            st.error("❌ Predicted status: LOAN REJECTED")
        st.metric("Estimated approval probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption("Probability is the model's estimate, not a guarantee or an explanation of a bank's actual decision.")

with tab_dashboard:
    st.subheader("Dataset overview")
    total = len(df)
    yes_count = int((df[TARGET] == "Yes").sum())
    no_count = int((df[TARGET] == "No").sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Labelled applicants", f"{total:,}")
    m2.metric("Approved in dataset", f"{yes_count:,}")
    m3.metric("Rejected in dataset", f"{no_count:,}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Approval distribution")
        dist = df[TARGET].value_counts().rename_axis("Status").reset_index(name="Applicants")
        st.bar_chart(dist.set_index("Status"))
    with right:
        st.markdown("#### Model evaluation")
        metric_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 score"],
            "Score": [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]],
        }).set_index("Metric")
        st.bar_chart(metric_df)
        st.caption(f"Evaluation on a held-out test set of {metrics['test_size']} applicants.")

    st.markdown("#### Confusion matrix")
    cm = metrics["confusion"]
    st.dataframe(
        pd.DataFrame(cm, index=["Actual: Rejected", "Actual: Approved"],
                     columns=["Predicted: Rejected", "Predicted: Approved"]),
        use_container_width=True
    )
    st.markdown("#### Dataset preview")
    st.dataframe(df.drop(columns=["Applicant_ID"], errors="ignore").head(10), use_container_width=True)

with tab_about:
    st.subheader("About this project")
    st.markdown("""
    - **Problem:** Predict whether a loan application is labelled Yes or No.
    - **Learning type:** Supervised classification.
    - **Algorithm:** Gaussian Naive Bayes.
    - **Preprocessing:** Missing-value imputation, one-hot encoding, standard scaling.
    - **Feature engineering:** Squared DTI ratio and squared credit score.
    - **Evaluation:** Accuracy, precision, recall, F1 score, and confusion matrix.

    The app trains its model from the included CSV on first launch. Rows with missing loan-approval labels are excluded from training rather than guessing their labels.
    """)
    st.warning("This sample dataset may not represent real lending practices. Do not use this demo to make real credit decisions.")
