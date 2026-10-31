import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD DATA & MODEL
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_processed.csv")

@st.cache_data
def load_raw_data():
    return pd.read_csv(
        "data/raw.csv"
    )

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model_final.pkl")

@st.cache_resource
def load_preprocessing():
    return joblib.load("models/preprocessing.pkl")


df = load_data()
raw_df = load_raw_data()
model_package = load_model()

model = model_package["model"]
threshold = model_package["threshold"]
metrics = model_package["metrics"]

preprocessing = load_preprocessing()

scaler = load_preprocessing()

# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Churn Prediction")

st.markdown(
    """
    Dashboard untuk menganalisis karakteristik pelanggan dan
    memprediksi potensi customer churn menggunakan Logistic Regression.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Customer Analysis",
        "Model Performance",
        "Feature Insights",
        "Prediction",
        "Business Recommendations"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Customer Overview")

    total_customers = len(df)
    churned_customers = df["Churn"].sum()
    churn_rate = churned_customers / total_customers * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "Churned Customers",
            f"{churned_customers:,}"
        )

    with col3:
        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

    st.divider()

    st.subheader("Churn Distribution")

    churn_counts = df["Churn"].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        ["No Churn", "Churn"],
        [
            churn_counts.get(0, 0),
            churn_counts.get(1, 0)
        ]
    )

    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Churn Distribution")

    st.pyplot(fig)
    
# ============================================================
# CUSTOMER ANALYSIS
# ============================================================

elif page == "Customer Analysis":

    st.header("Customer Analysis")

    st.write(
        "Explore customer characteristics and their relationship "
        "with churn."
    )

    # --------------------------------------------------------
    # Contract
    # --------------------------------------------------------

    st.subheader("Churn Rate by Contract")

    contract_churn = (
        raw_df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        contract_churn.index,
        contract_churn.values
    )

    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Contract")
    ax.set_title("Churn Rate by Contract")

    plt.xticks(rotation=15)

    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------------
    # Internet Service
    # --------------------------------------------------------

    st.subheader("Churn Rate by Internet Service")

    internet_churn = (
        raw_df.groupby("InternetService")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        internet_churn.index,
        internet_churn.values
    )

    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Internet Service")
    ax.set_title("Churn Rate by Internet Service")

    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------------
    # Payment Method
    # --------------------------------------------------------

    st.subheader("Churn Rate by Payment Method")

    payment_churn = (
        raw_df.groupby("PaymentMethod")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        payment_churn.index,
        payment_churn.values
    )

    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Payment Method")
    ax.set_title("Churn Rate by Payment Method")

    plt.xticks(rotation=20)

    st.pyplot(fig)

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("Model Performance")

    st.write(
        "Final model: **Logistic Regression**"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            f"{metrics['accuracy']:.2%}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{metrics['precision']:.2%}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{metrics['recall']:.2%}"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{metrics['f1']:.2%}"
        )

    with col5:
        st.metric(
            "ROC-AUC",
            f"{metrics['roc_auc']:.2%}"
        )

    st.divider()

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = np.array([
        [799, 236],
        [103, 271]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"]
    )

    disp.plot(
        ax=ax,
        values_format="d",
        colorbar=False
    )

    ax.set_title("Final Model Confusion Matrix")

    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    st.subheader("Classification Threshold")

    st.write(
        f"""
        The final model uses a classification threshold of
        **{threshold:.2f}** instead of the default **0.50**.
        """
    )

    st.info(
        """
        The threshold was adjusted to improve the balance between
        precision and recall for the Churn class. The selected
        threshold maximized the F1-score during threshold tuning.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    st.subheader("Model Interpretation")

    st.markdown(
        """
        **Recall — Churn: 72.46%**

        The model successfully identifies approximately 72% of
        customers who actually churn.

        **Precision — Churn: 53.45%**

        Among customers predicted as churn, approximately 53%
        actually churned.

        **ROC-AUC: 83.82%**

        The model demonstrates good ability to distinguish between
        customers who churn and those who do not.
        """
    )

# ============================================================
# FEATURE INSIGHTS
# ============================================================

elif page == "Feature Insights":

    st.header("Feature Insights")

    coefficients = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Coefficient": model.coef_[0]
    })

    coefficients["Absolute Coefficient"] = (
        coefficients["Coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        "Absolute Coefficient",
        ascending=False
    )

    st.subheader("Top Features")

    top_features = coefficients.head(15).sort_values(
        "Coefficient"
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        top_features["Feature"],
        top_features["Coefficient"]
    )

    ax.axvline(0, linewidth=1)

    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_title("Top 15 Feature Coefficients")

    st.pyplot(fig)

    st.subheader("Coefficient Table")

    st.dataframe(
        coefficients.head(15),
        use_container_width=True
    )

# ============================================================
# PREDICTION
# ============================================================
    
elif page == "Prediction":
    st.header("Customer Churn Prediction")

    st.write(
        "Enter customer information below to estimate the probability "
        "of customer churn."
    )

    st.divider()

    # =========================
    # CUSTOMER INFORMATION
    # =========================
    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

    with col2:
        phone_service = st.selectbox(
            "Phone Service",
            ["No", "Yes"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No phone service", "No", "Yes"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check"
            ]
        )

    st.divider()

    # =========================
    # ADDITIONAL SERVICES
    # =========================
    st.subheader("Additional Services")

    col1, col2, col3 = st.columns(3)

    with col1:
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

    with col2:
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

    st.divider()

    # =========================
    # CREATE MODEL INPUT
    # =========================

    if st.button("Predict Churn", type="primary"):

        input_data = pd.DataFrame([{
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,

            "gender_Male": 1 if gender == "Male" else 0,
            "Partner_Yes": 1 if partner == "Yes" else 0,
            "Dependents_Yes": 1 if dependents == "Yes" else 0,

            "PhoneService_Yes": 1 if phone_service == "Yes" else 0,

            "MultipleLines_No phone service":
                1 if multiple_lines == "No phone service" else 0,

            "MultipleLines_Yes":
                1 if multiple_lines == "Yes" else 0,

            "InternetService_Fiber optic":
                1 if internet_service == "Fiber optic" else 0,

            "InternetService_No":
                1 if internet_service == "No" else 0,

            "OnlineSecurity_No internet service":
                1 if online_security == "No internet service" else 0,

            "OnlineSecurity_Yes":
                1 if online_security == "Yes" else 0,

            "OnlineBackup_No internet service":
                1 if online_backup == "No internet service" else 0,

            "OnlineBackup_Yes":
                1 if online_backup == "Yes" else 0,

            "DeviceProtection_No internet service":
                1 if device_protection == "No internet service" else 0,

            "DeviceProtection_Yes":
                1 if device_protection == "Yes" else 0,

            "TechSupport_No internet service":
                1 if tech_support == "No internet service" else 0,

            "TechSupport_Yes":
                1 if tech_support == "Yes" else 0,

            "StreamingTV_No internet service":
                1 if streaming_tv == "No internet service" else 0,

            "StreamingTV_Yes":
                1 if streaming_tv == "Yes" else 0,

            "StreamingMovies_No internet service":
                1 if streaming_movies == "No internet service" else 0,

            "StreamingMovies_Yes":
                1 if streaming_movies == "Yes" else 0,

            "Contract_One year":
                1 if contract == "One year" else 0,

            "Contract_Two year":
                1 if contract == "Two year" else 0,

            "PaperlessBilling_Yes":
                1 if paperless_billing == "Yes" else 0,

            "PaymentMethod_Credit card (automatic)":
                1 if payment_method == "Credit card (automatic)" else 0,

            "PaymentMethod_Electronic check":
                1 if payment_method == "Electronic check" else 0,

            "PaymentMethod_Mailed check":
                1 if payment_method == "Mailed check" else 0
        }])

        # Pastikan urutan fitur sama dengan saat training
        input_data = input_data[model_package["features"]]
        
        scaler_columns = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]

        numeric_model_features = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges"
        ]

        for feature in numeric_model_features:
            idx = scaler_columns.index(feature)

            input_data[feature] = (
                input_data[feature] - scaler.mean_[idx]
            ) / scaler.scale_[idx]

        # =========================
        # PREDICTION
        # =========================

        churn_probability = model.predict_proba(input_data)[0, 1]
        
        st.write("Raw probability:", churn_probability)

        prediction = (
            1 if churn_probability >= threshold else 0
        )

        st.divider()

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Churn Probability",
                f"{churn_probability:.2%}"
            )

        with col2:
            if prediction == 1:
                st.error("⚠️ High Churn Risk")
            else:
                st.success("✅ Low Churn Risk")

        # =========================
        # RISK LEVEL
        # =========================

        if churn_probability < 0.20:
            risk_level = "Low"
        elif churn_probability < 0.50:
            risk_level = "Medium"
        else:
            risk_level = "High"

        st.write(f"**Risk Level:** {risk_level}")

        # =========================
        # RECOMMENDATION
        # =========================

        if prediction == 1:
            st.warning(
                "This customer is predicted to have a higher likelihood "
                "of churn. Consider prioritizing this customer for "
                "retention activities."
            )
        else:
            st.info(
                "This customer is currently predicted as less likely "
                "to churn. Continue monitoring customer engagement "
                "and service experience."
            )
            
elif page == "Business Recommendations":
    st.header("Business Recommendations")

    st.markdown("""
    Halaman ini menyajikan rekomendasi retensi berdasarkan pola
    customer churn dan karakteristik pelanggan yang dianalisis.
    Rekomendasi berikut merupakan usulan tindakan bisnis yang perlu
    dievaluasi lebih lanjut, bukan bukti hubungan sebab-akibat.
    """)

    st.divider()

    # 1. Contract
    st.subheader("1. Improve Contract Retention")

    st.markdown("""
    **Temuan:** Pelanggan dengan kontrak bulanan menunjukkan
    kecenderungan churn yang lebih tinggi dibandingkan pelanggan
    dengan kontrak jangka panjang dalam analisis dataset.

    **Rekomendasi:**
    - Tawarkan pilihan kontrak satu tahun dengan manfaat yang jelas.
    - Berikan program loyalitas untuk mendorong perpanjangan kontrak.
    - Evaluasi alasan pelanggan memilih atau mempertahankan kontrak bulanan.

    **Indikator evaluasi:** Tingkat perpanjangan kontrak dan churn
    berdasarkan jenis kontrak.
    """)

    st.divider()

    # 2. Tenure
    st.subheader("2. Strengthen Early Customer Retention")

    st.markdown("""
    **Temuan:** Masa berlangganan (*tenure*) yang lebih pendek
    berkaitan dengan kecenderungan churn yang lebih tinggi.

    **Rekomendasi:**
    - Buat program onboarding bagi pelanggan baru.
    - Lakukan tindak lanjut setelah pelanggan mulai menggunakan layanan.
    - Sediakan panduan penggunaan dan kanal bantuan yang mudah diakses.

    **Indikator evaluasi:** Churn rate pelanggan baru berdasarkan
    kelompok masa berlangganan.
    """)

    st.divider()

    # 3. Internet service
    st.subheader("3. Evaluate Fiber Optic Customer Experience")

    st.markdown("""
    **Temuan:** Penggunaan layanan internet fiber optic berkaitan
    dengan probabilitas churn yang lebih tinggi dalam model.

    **Rekomendasi:**
    - Evaluasi keluhan terkait kecepatan, stabilitas, dan gangguan layanan.
    - Tinjau data pengaduan serta kepuasan pelanggan fiber optic.
    - Sesuaikan tindak lanjut berdasarkan masalah yang ditemukan.

    **Indikator evaluasi:** Churn rate, jumlah keluhan, dan tingkat
    kepuasan pelanggan berdasarkan jenis layanan internet.
    """)

    st.divider()

    # 4. Payment method
    st.subheader("4. Review Electronic Check Payment Experience")

    st.markdown("""
    **Temuan:** Metode pembayaran electronic check berkaitan dengan
    kecenderungan churn yang lebih tinggi pada analisis dataset.

    **Rekomendasi:**
    - Evaluasi kemudahan dan pengalaman pelanggan saat melakukan pembayaran.
    - Periksa apakah terdapat kendala pembayaran atau masalah transaksi.
    - Informasikan alternatif pembayaran otomatis kepada pelanggan
      yang menginginkannya.

    **Indikator evaluasi:** Churn rate berdasarkan metode pembayaran
    dan jumlah kendala pembayaran yang dilaporkan.
    """)

    st.divider()

    # 5. Model-assisted retention
    st.subheader("5. Prioritize Retention Outreach")

    st.markdown("""
    **Temuan:** Model Logistic Regression dapat membantu mengidentifikasi
    pelanggan dengan probabilitas churn yang relatif tinggi.

    **Rekomendasi:**
    - Gunakan hasil prediksi untuk membantu menentukan prioritas tindak lanjut.
    - Tinjau karakteristik dan riwayat layanan pelanggan sebelum mengambil tindakan.
    - Uji program retensi pada kelompok pelanggan terpilih dan bandingkan hasilnya
      dengan kelompok pembanding.

    **Indikator evaluasi:** Tingkat churn setelah program retensi,
    tingkat respons pelanggan, dan biaya retensi per pelanggan.
    """)

    st.divider()

    st.info("""
    **Important note**

    Recommendations are based on observed associations and model predictions.
    They do not establish causal relationships. Business actions should be
    evaluated using customer feedback and retention outcomes before
    large-scale implementation.
    """)