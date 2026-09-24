"""
HealthAI Virtual Hospital - Final Streamlit Application

Final UI integration:
- Home
- Doctor Portal
  - Chest X-ray
  - Length of Stay
  - Primary Diagnosis
  - Patient Sentiment
  - Medical NER
- Patient Portal
  - Health Risk
  - Medical Translation
  - HealthAI Assistant
  - Appointments
  - Hospital Services

All model inference is performed through FastAPI.
"""

from datetime import date, timedelta
import json
import requests
import streamlit as st


def find_first_value(data, keys):
    """Find a useful value in nested API responses without changing the API."""
    wanted = {str(k).lower() for k in keys}
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() in wanted and v not in (None, "", [], {}):
                    return v
            for v in obj.values():
                found = walk(v)
                if found not in (None, "", [], {}):
                    return found
        elif isinstance(obj, list):
            for v in obj:
                found = walk(v)
                if found not in (None, "", [], {}):
                    return found
        return None
    return walk(data)


def normalize_diagnosis(data):
    """Accept the common diagnosis field names used by the HealthAI API."""
    value = find_first_value(data, [
        "predicted_primary_diagnosis", "primary_diagnosis", "predicted_diagnosis",
        "diagnosis", "prediction", "predicted_class", "class", "label",
        "prediction_label", "result"
    ])
    if isinstance(value, dict):
        value = find_first_value(value, ["diagnosis", "label", "prediction", "class", "name"])
    return value


HEALTHAI_CSS = """
<style>
:root {
    --ha-blue: #1565C0;
    --ha-blue-dark: #0D47A1;
    --ha-blue-light: #EAF3FF;
    --ha-border: #D7E6F7;
    --ha-text: #16324F;
    --ha-muted: #60758A;
}
.stApp {
    background: linear-gradient(180deg, #F7FBFF 0%, #FFFFFF 48%, #F7FBFF 100%);
    color: var(--ha-text);
}
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D47A1 0%, #1565C0 55%, #1976D2 100%);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--ha-border);
    border-radius: 14px;
    padding: 14px;
    box-shadow: 0 5px 18px rgba(21,101,192,.07);
}
div[data-testid="stMetricLabel"] {
    font-size: .78rem !important;
}
div[data-testid="stMetricValue"] {
    font-size: 1.18rem !important;
}

div.stButton > button {
    border-radius: 10px;
    border: 1px solid #1565C0;
    background: #1565C0;
    color: white;
    font-weight: 600;
}
div.stButton > button:hover {
    background: #0D47A1;
    border-color: #0D47A1;
}
.ha-hero {
    background: linear-gradient(135deg, #0D47A1 0%, #1565C0 58%, #42A5F5 100%);
    border-radius: 24px;
    padding: 38px 42px;
    color: white;
    box-shadow: 0 14px 35px rgba(13,71,161,.18);
    margin-bottom: 26px;
}
.ha-hero h1 {
    color: white;
    font-size: 2.05rem;
    margin: 0 0 8px 0;
}
.ha-hero p {
    color: #EAF4FF;
    font-size: 0.98rem;
    margin: 0;
}
.ha-card {
    background: white;
    border: 1px solid var(--ha-border);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 7px 22px rgba(13,71,161,.06);
    margin-bottom: 16px;
}
.ha-doctor-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #F0F7FF 100%);
    border: 1px solid #CFE2F5;
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 10px 28px rgba(13,71,161,.08);
}
.ha-doctor-photo {
    max-width: 210px;
    background: linear-gradient(145deg, #E3F2FD, #FFFFFF);
    border-radius: 18px;
    padding: 12px;
    border: 1px solid #D1E5F7;
    text-align: center;
}
.ha-doctor-photo svg {
    max-width: 170px;
    height: auto;
}
.ha-badge {
    display: inline-block;
    background: #E3F2FD;
    color: #0D47A1;
    border: 1px solid #BBDEFB;
    border-radius: 999px;
    padding: 5px 11px;
    font-size: .82rem;
    font-weight: 700;
}
div[data-testid="stChatMessage"] {
    border-radius: 14px;
    border: 1px solid #DDEAF7;
}

/* Keep Tamil/Hindi patient-facing text readable. */
div[data-testid="stChatMessage"] p,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
    color: #16324F !important;
}

textarea, textarea:focus {
    color: #16324F !important;
}
textarea::placeholder {
    color: #60758A !important;
    opacity: 1 !important;
}
.stExpander {
    border: 1px solid #D7E6F7;
    border-radius: 14px;
    background: white;
}
.stExpander summary {
    font-weight: 700;
    color: #0D47A1;
}

</style>
"""


DOCTOR_SVG = """
<svg viewBox="0 0 360 360" xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="Professional doctor illustration">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#E3F2FD"/>
      <stop offset="100%" stop-color="#FFFFFF"/>
    </linearGradient>
  </defs>
  <rect width="360" height="360" rx="32" fill="url(#bg)"/>
  <circle cx="180" cy="126" r="67" fill="#F4C7A1"/>
  <path d="M113 119c6-48 34-75 68-75 39 0 67 28 68 76-20-18-37-25-61-26-24-1-48 8-75 25z" fill="#26384A"/>
  <path d="M127 139c7 31 27 47 53 47s46-16 53-47c-13 14-31 21-53 21s-40-7-53-21z" fill="#E7A97F"/>
  <circle cx="156" cy="124" r="5" fill="#26384A"/>
  <circle cx="204" cy="124" r="5" fill="#26384A"/>
  <path d="M166 150c9 7 19 7 28 0" fill="none" stroke="#8B4E35" stroke-width="4" stroke-linecap="round"/>
  <path d="M84 315c6-74 39-112 96-112s90 38 96 112z" fill="#FFFFFF"/>
  <path d="M126 217l54 60 54-60c20 15 31 36 37 98H89c6-62 17-83 37-98z" fill="#EAF3FF"/>
  <path d="M180 218v97" stroke="#1565C0" stroke-width="4"/>
  <path d="M158 232h44" stroke="#1565C0" stroke-width="4"/>
  <rect x="197" y="246" width="34" height="47" rx="4" fill="#FFFFFF" stroke="#B9D6EF" stroke-width="2"/>
  <path d="M214 254v31M204 269h20" stroke="#1565C0" stroke-width="4" stroke-linecap="round"/>
  <circle cx="117" cy="267" r="20" fill="#FFFFFF" stroke="#B9D6EF" stroke-width="3"/>
  <path d="M117 257v20M107 267h20" stroke="#1565C0" stroke-width="4" stroke-linecap="round"/>
</svg>
"""



st.set_page_config(
    page_title="HealthAI Virtual Hospital",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

API = "http://127.0.0.1:8000"
st.markdown(HEALTHAI_CSS, unsafe_allow_html=True)




def call_api(path, payload=None, method="POST", timeout=180, files=None):
    try:
        url = API + path

        if method == "GET":
            response = requests.get(url, timeout=timeout)
        else:
            response = requests.post(
                url,
                json=payload,
                files=files,
                timeout=timeout,
            )

        try:
            data = response.json()
        except Exception:
            data = {"message": response.text}

        return response.status_code, data

    except requests.exceptions.ConnectionError:
        return None, {
            "user_error": (
                "HealthAI API is not running. Start FastAPI first."
            )
        }

    except requests.exceptions.Timeout:
        return None, {
            "user_error": (
                "The AI service is taking longer than expected. "
                "Please try again."
            )
        }

    except Exception as exc:
        return None, {"user_error": str(exc)}


def error_message(code, data):
    if code is None:
        return data.get("user_error", "HealthAI is unavailable.")

    if isinstance(data, dict) and "detail" in data:
        detail = data["detail"]
        if isinstance(detail, dict):
            return detail.get("message", "The request could not be completed.")
        return str(detail)

    if code == 422:
        return "Please check the information entered and try again."

    return data.get(
        "message",
        "The request could not be completed. Please try again.",
    )


def render_error(code, data):
    st.error(error_message(code, data))


def show_prediction_result(code, data):
    if code != 200:
        render_error(code, data)
        return

    if data.get("status") not in ("success", "extracted"):
        render_error(code, data)
        return

    st.success("Analysis completed successfully.")
    return data


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def find_first_value(data, keys):
    """Find the first matching key recursively in an API response."""
    if isinstance(data, dict):
        for key in keys:
            if key in data and data[key] not in (None, ""):
                return data[key]
        for value in data.values():
            found = find_first_value(value, keys)
            if found not in (None, ""):
                return found
    elif isinstance(data, list):
        for item in data:
            found = find_first_value(item, keys)
            if found not in (None, ""):
                return found
    return None


def extract_xray_prediction(data):
    """Normalize common CNN response shapes into (label, probability)."""
    import json
    import re

    label_keys = {
        "prediction", "predicted_class", "predicted_label", "class",
        "label", "class_name", "predicted_class_name", "xray_prediction",
        "predicted", "diagnosis"
    }
    prob_keys = {
        "pneumonia_probability", "pneumonia_prob", "probability",
        "confidence", "pneumonia_score", "score", "positive_probability"
    }
    label = None
    probability = None

    def walk(obj):
        nonlocal label, probability
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k).strip().lower()
                if label is None and key in label_keys and v not in (None, ""):
                    if isinstance(v, (str, int, float)):
                        label = v
                if probability is None and key in prob_keys and isinstance(v, (int, float, str)):
                    try:
                        probability = float(v)
                    except Exception:
                        pass
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)
        elif isinstance(obj, str):
            s = obj.strip()
            if s.startswith("{") or s.startswith("["):
                try:
                    walk(json.loads(s))
                    return
                except Exception:
                    pass
            upper = s.upper()
            if label is None:
                if "NO PNEUMONIA" in upper or "NORMAL" in upper:
                    label = "NORMAL"
                elif "PNEUMONIA" in upper:
                    label = "PNEUMONIA"
            if probability is None:
                match = re.search(r"(?:pneumonia[^0-9]{0,20}|probability[^0-9]{0,20}|confidence[^0-9]{0,20})(0?\.\d+|1(?:\.0+)?)", s, re.I)
                if match:
                    try:
                        probability = float(match.group(1))
                    except Exception:
                        pass

    walk(data)
    if probability is not None:
        if probability > 1:
            probability /= 100.0
        probability = max(0.0, min(1.0, probability))
        if label is None:
            label = "PNEUMONIA" if probability >= 0.50 else "NORMAL"
    return label, probability


if "appointments" not in st.session_state:
    st.session_state.appointments = []

if "chat" not in st.session_state:
    st.session_state.chat = []


st.sidebar.title("🏥 HealthAI")
st.sidebar.caption("Intelligent Virtual Hospital")

portal = st.sidebar.radio(
    "Select Portal",
    ["🏠 Home", "👨‍⚕️ Doctor Portal", "👤 Patient Portal"],
)


# =====================================================================
# HOME
# =====================================================================

if portal == "🏠 Home":
    st.markdown(
        """
        <div class="ha-hero">
            <h1>🏥 HealthAI Virtual Hospital</h1>
            <p>End-to-end healthcare AI for clinical decision support and evidence-grounded health information.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    doctor, patient = st.columns(2)
    with doctor:
        st.markdown(
            '<div class="ha-card"><h3>👨‍⚕️ Doctor Portal</h3>'
            '<p>Clinical decision-support workspace.</p></div>',
            unsafe_allow_html=True,
        )
    with patient:
        st.markdown(
            '<div class="ha-card"><h3>👤 Patient Portal</h3>'
            '<p>Health assessment, appointments and AI assistance.</p></div>',
            unsafe_allow_html=True,
        )

    with st.expander("⚙️ Technical Overview", expanded=False):
        st.markdown(
            """
            **AI Capabilities**

            • **11 AI tools** across machine learning, deep learning,
              medical NLP, RAG and intelligent orchestration.  
            • **8 RAG sources** from the curated WHO and MedlinePlus
              healthcare knowledge base.  
            • **Agentic AI** for natural-language request routing and
              tool execution.  
            • **FastAPI** REST API layer connecting the interface to AI services.

            **Model Stack**

            • **Classical ML:** Diabetes Risk, Hospital Length of Stay,
              Primary Diagnosis, Patient Clustering and Association Analysis.  
            • **Deep Learning:** CNN for chest X-ray analysis and RNN/LSTM
              for clinical deterioration prediction.  
            • **Medical NLP:** BioBERT / ClinicalBERT for medical entity extraction.  
            • **Sentiment:** DistilBERT-based deep-learning text classification.

            **Architecture**

            `User → Streamlit → FastAPI → Agent Router → AI Tool → Model/RAG → Result`
            """
        )

    with st.expander("🔍 How HealthAI Works", expanded=False):
        st.markdown(
            """
            **1. User request** → **2. Agent Router** → **3. Appropriate AI tool**
            → **4. Model or RAG retrieval** → **5. Structured response**

            Different healthcare requests are handled by specialized AI
            capabilities. Unsupported requests are safely rejected instead
            of being forced through a medical model.
            """
        )

    st.caption("HealthAI Virtual Hospital — Portfolio Demonstration")

elif portal == "👨‍⚕️ Doctor Portal":

    page = st.sidebar.radio(
        "Doctor Services",
        [
            "Dashboard",
            "🫁 Chest X-ray",
            "🏨 Length of Stay",
            "🧬 Primary Diagnosis",
            "💬 Patient Sentiment",
            "🧪 Medical NER",
        ],
    )

    if page == "Dashboard":
        st.title("👨‍⚕️ Doctor Dashboard")
        st.caption("Clinical AI decision-support workspace")

        photo_col, intro_col = st.columns([1, 4])
        with photo_col:
            st.markdown(
                '<div class="ha-doctor-photo">' + DOCTOR_SVG + '</div>',
                unsafe_allow_html=True,
            )
        with intro_col:
            st.markdown(
                """
                <div class="ha-card">
                    <h3>Clinical Decision Support</h3>
                    <p>
                        AI-assisted tools for imaging, prediction, clinical
                        text extraction and patient experience analysis.
                    </p>
                    <p><b>Select a clinical AI service from the sidebar.</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif page == "🫁 Chest X-ray":

        st.title("🫁 Chest X-ray Analysis")

        st.info(
            "CNN Chest X-ray Prediction — AI prediction based on the "
            "synthetic chest X-ray dataset used for model development. "
            "This is a portfolio demonstration and is not intended for "
            "clinical diagnosis or medical decision-making."
        )

        st.caption("Upload a chest X-ray for CNN-based analysis.")


        uploaded = st.file_uploader(
            "Choose X-ray image",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded:

            st.image(
                uploaded,
                caption="Uploaded chest X-ray",
                use_container_width=True,
            )

            if st.button(
                "Analyze X-ray",
                type="primary",
            ):

                files = {
                    "file": (
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type,
                    )
                }

                code, data = call_api(
                    "/predict/xray",
                    files=files,
                    timeout=180,
                )

                if code != 200:
                    render_error(code, data)
                else:
                    label, p = extract_xray_prediction(data)

                    if label is not None:
                        normalized = str(label).strip().upper()
                        if normalized in {"PNEUMONIA", "PNEUMONIA DETECTED", "POSITIVE", "1"}:
                            st.error("🫁 PNEUMONIA DETECTED")
                        elif normalized in {"NORMAL", "NO PNEUMONIA", "NEGATIVE", "0"}:
                            st.success("🫁 NO PNEUMONIA DETECTED")
                        else:
                            st.info(f"🫁 CNN result: {normalized}")

                        if p is not None:
                            st.metric("Pneumonia probability", f"{p:.2%}")
                    else:
                        st.warning(
                            "The CNN service responded successfully, but the returned payload "
                            "did not contain a readable pneumonia/normal result. "
                            "Please make sure the latest X-ray adapter and FastAPI files are being used."
                        )
                        with st.expander("Technical response (for debugging)"):
                            st.json(data)

                    st.caption(
                        "CNN output is a portfolio/demo decision-support result "
                        "and is not a clinical diagnosis."
                    )

    # -----------------------------------------------------------------
    # LOS
    # -----------------------------------------------------------------

    elif page == "🏨 Length of Stay":

        st.title("🏨 Hospital Length of Stay")
        st.caption(
            "Estimate expected hospital stay using the trained XGBoost model."
        )

        a, b = st.columns(2)

        with a:
            rcount = st.number_input(
                "Diagnosis Count",
                min_value=0,
                max_value=20,
                value=3,
            )
            gender = st.selectbox(
                "Gender",
                ["M", "F"],
            )
            hematocrit = st.number_input(
                "Hematocrit",
                value=40.0,
            )
            neutrophils = st.number_input(
                "Neutrophils",
                value=65.0,
            )
            sodium = st.number_input(
                "Sodium",
                value=139.0,
            )
            glucose = st.number_input(
                "Glucose",
                value=110.0,
            )

        with b:
            creatinine = st.number_input(
                "Creatinine",
                value=1.0,
            )
            bmi = st.number_input(
                "BMI",
                value=25.0,
            )
            pulse = st.number_input(
                "Pulse",
                value=78.0,
            )
            respiration = st.number_input(
                "Respiration",
                value=18.0,
            )
            secondary = st.number_input(
                "Secondary Diagnosis Count",
                min_value=0,
                max_value=20,
                value=1,
            )

        # Facility is deliberately hidden. The original model contains
        # facid as an internal categorical feature; the user should not
        # need to know or provide a facility code.
        payload = {
            "rcount": rcount,
            "gender": gender,
            "dialysisrenalendstage": 0,
            "asthma": 0,
            "irondef": 0,
            "pneum": 0,
            "substancedependence": 0,
            "psychologicaldisordermajor": 0,
            "depress": 0,
            "psychother": 0,
            "fibrosisandother": 0,
            "malnutrition": 0,
            "hemo": 0,
            "hematocrit": hematocrit,
            "neutrophils": neutrophils,
            "sodium": sodium,
            "glucose": glucose,
            "bloodureanitro": 15.0,
            "creatinine": creatinine,
            "bmi": bmi,
            "pulse": pulse,
            "respiration": respiration,
            "secondarydiagnosisnonicd9": secondary,
            "facid": "A",
        }

        if st.button(
            "Predict Length of Stay",
            type="primary",
        ):

            code, data = call_api(
                "/predict/los",
                payload,
                timeout=180,
            )

            if code != 200:
                render_error(code, data)
            elif data.get("status") != "success":
                render_error(code, data)
            else:
                st.success("Length-of-stay prediction completed.")

                predicted_stay = find_first_value(
                    data,
                    ["predicted_length_of_stay_days", "predicted_stay", "prediction"],
                )

                if predicted_stay is not None:
                    st.markdown(
                        '<div class="result"><h2>Predicted Length of Stay</h2>'
                        f'<h1>{safe_float(predicted_stay):.2f} days</h1></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.error("The LOS service returned no predicted stay.")

    # -----------------------------------------------------------------
    # PRIMARY DIAGNOSIS
    # -----------------------------------------------------------------

    elif page == "🧬 Primary Diagnosis":

        st.title("🧬 Primary Diagnosis Prediction")
        st.caption(
            "Predict the most likely primary diagnosis from structured patient information."
        )

        a, b = st.columns(2)

        with a:
            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=52,
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
            )

            region = st.text_input(
                "Region",
                "Tamil Nadu",
            )

            socio = st.selectbox(
                "Socioeconomic Status",
                ["Low", "Middle", "High"],
            )

            symptoms = st.text_area(
                "Symptoms",
                "frequent urination, excessive thirst",
            )

        with b:
            glucose = st.number_input(
                "Blood Glucose (mg/dL)",
                value=165.0,
            )

            hba1c = st.number_input(
                "HbA1c (%)",
                value=7.8,
            )

            cholesterol = st.number_input(
                "Total Cholesterol (mg/dL)",
                value=220.0,
            )

            bmi = st.number_input(
                "BMI",
                value=29.5,
            )

        payload = {
            "Age": age,
            "Gender": gender,
            "Region": region,
            "Socioeconomic_Status": socio,
            "Symptoms": symptoms,
            "Blood_Glucose_mg_dL": glucose,
            "HbA1c_%": hba1c,
            "Total_Cholesterol_mg_dL": cholesterol,
            "BMI": bmi,
        }

        if st.button(
            "Predict Primary Diagnosis",
            type="primary",
        ):

            code, data = call_api(
                "/predict/diagnosis",
                payload,
                timeout=180,
            )

            if code != 200:
                render_error(code, data)
            else:
                diagnosis = normalize_diagnosis(data)
                confidence = find_first_value(
                    data,
                    ["confidence", "prediction_confidence", "probability", "score"],
                )

                if diagnosis:
                    st.success("Diagnosis prediction completed.")

                    st.markdown(
                        '<div class="result"><h2>Predicted Diagnosis</h2>'
                        f'<h1>{diagnosis}</h1></div>',
                        unsafe_allow_html=True,
                    )

                    if confidence is not None:
                        try:
                            c = float(confidence)
                            if c > 1:
                                c /= 100
                            st.metric(
                                "Model confidence",
                                f"{c:.1%}",
                            )
                        except Exception:
                            pass

                    st.warning(
                        "This is a machine-learning prediction, not a confirmed medical diagnosis."
                    )

                else:
                    st.error(
                        "The model returned a response without a diagnosis label."
                    )

    # -----------------------------------------------------------------
    # SENTIMENT
    # -----------------------------------------------------------------

    elif page == "💬 Patient Sentiment":

        st.title("💬 Patient Sentiment Analysis")
        st.caption(
            "Classify patient feedback as Positive or Negative."
        )

        text = st.text_area(
            "Patient feedback",
            "The staff were helpful and explained everything clearly.",
            height=150,
        )

        if st.button(
            "Analyze Sentiment",
            type="primary",
        ):

            code, data = call_api(
                "/predict/sentiment",
                {"text": text},
                timeout=120,
            )

            if code != 200:
                render_error(code, data)
            else:
                sentiment = data.get("sentiment")

                if sentiment:
                    if sentiment.lower() == "positive":
                        st.success("😊 POSITIVE")
                    else:
                        st.error("☹️ NEGATIVE")

                    p = data.get("probabilities", {})

                    if isinstance(p, dict):
                        a, b = st.columns(2)

                        with a:
                            if "positive" in p:
                                st.metric(
                                    "Positive probability",
                                    f"{safe_float(p['positive']):.1%}",
                                )

                        with b:
                            if "negative" in p:
                                st.metric(
                                    "Negative probability",
                                    f"{safe_float(p['negative']):.1%}",
                                )

                    st.caption(
                        "Sentiment model evaluation used a controlled dataset "
                        "with limited unique feedback texts."
                    )
                else:
                    st.error(
                        "The sentiment service returned no sentiment label."
                    )

    # -----------------------------------------------------------------
    # NER
    # -----------------------------------------------------------------

    elif page == "🧪 Medical NER":

        st.title("🧪 Medical NER")
        st.caption(
            "BioBERT extracts medical entities from clinical text."
        )

        text = st.text_area(
            "Clinical note",
            (
                "Patient has critical limb ischaemia with stump pain. "
                "Started metformin 500mg bd and apixaban."
            ),
            height=180,
        )

        if st.button(
            "Extract Medical Entities",
            type="primary",
        ):

            code, data = call_api(
                "/predict/ner",
                {"text": text},
                timeout=180,
            )

            if code != 200:
                render_error(code, data)
            else:
                entities = data.get("entities", [])

                st.success(
                    f"Medical NER completed — {len(entities)} entities found."
                )

                if entities:
                    display_rows = []
                    for entity in entities:
                        confidence = entity.get("confidence")
                        confidence_text = "N/A"
                        if confidence is not None:
                            confidence_text = f"{safe_float(confidence):.2%}"

                        display_rows.append(
                            {
                                "Medical Entity": entity.get("text", ""),
                                "Type": entity.get("label", ""),
                                "Model Confidence": confidence_text,
                            }
                        )

                    st.dataframe(
                        display_rows,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.caption(
                        "Model confidence reflects the NER model's confidence in the "
                        "extracted entity; it is not clinical certainty."
                    )
                else:
                    st.info(
                        "No entities passed the configured confidence threshold."
                    )




# PATIENT PORTAL
# =====================================================================

else:

    # -----------------------------------------------------------------
    # PATIENT PORTAL LANGUAGE GATE
    # -----------------------------------------------------------------
    if "patient_language" not in st.session_state:
        st.session_state["patient_language"] = None

    if st.session_state["patient_language"] is None:
        st.markdown(
            """
            <div style="text-align:center; padding:48px 20px 28px;">
                <div style="font-size:4rem;">🏥</div>
                <h1 style="font-size:2.4rem;">HealthAI Patient Portal</h1>
                <p style="font-size:1.25rem;">Choose your language / अपनी भाषा चुनें / உங்கள் மொழியைத் தேர்ந்தெடுக்கவும்</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🇬🇧  English", use_container_width=True, type="primary"):
                st.session_state["patient_language"] = "English"
                st.rerun()
        with c2:
            if st.button("🇮🇳  हिन्दी", use_container_width=True, type="primary"):
                st.session_state["patient_language"] = "हिन्दी"
                st.rerun()
        with c3:
            if st.button("🇮🇳  தமிழ்", use_container_width=True, type="primary"):
                st.session_state["patient_language"] = "தமிழ்"
                st.rerun()
        st.stop()

    language = st.session_state["patient_language"]
    is_hindi = language == "हिन्दी"
    is_tamil = language == "தமிழ்"

    PATIENT_TRANSLATIONS = {
        "हिन्दी": {
            "Patient Services": "रोगी सेवाएँ", "Dashboard": "डैशबोर्ड",
            "❤️ Health Risk": "❤️ स्वास्थ्य जोखिम", "🤖 HealthAI Assistant": "🤖 हेल्थAI सहायक",
            "📅 Appointments": "📅 अपॉइंटमेंट", "🏥 Hospital Services": "🏥 अस्पताल सेवाएँ",
            "Patient Dashboard": "रोगी डैशबोर्ड",
            "Personal Health Risk Assessment": "व्यक्तिगत स्वास्थ्य जोखिम आकलन",
            "HealthAI Assistant": "हेल्थAI सहायक", "Appointments": "अपॉइंटमेंट",
            "Hospital Services": "अस्पताल सेवाएँ", "Medical Information": "चिकित्सा जानकारी",
            "Enter Medical Information": "चिकित्सा जानकारी दर्ज करें", "Target Language": "लक्षित भाषा",
            "Translate Medical Information": "चिकित्सा जानकारी का अनुवाद करें",
        },
        "தமிழ்": {
            "Patient Services": "நோயாளர் சேவைகள்", "Dashboard": "டாஷ்போர்டு",
            "❤️ Health Risk": "❤️ உடல்நல ஆபத்து", "🤖 HealthAI Assistant": "🤖 HealthAI உதவியாளர்",
            "📅 Appointments": "📅 சந்திப்புகள்", "🏥 Hospital Services": "🏥 மருத்துவமனை சேவைகள்",
            "Patient Dashboard": "நோயாளர் டாஷ்போர்டு",
            "Personal Health Risk Assessment": "தனிப்பட்ட உடல்நல ஆபத்து மதிப்பீடு",
            "HealthAI Assistant": "HealthAI உதவியாளர்", "Appointments": "சந்திப்புகள்",
            "Hospital Services": "மருத்துவமனை சேவைகள்", "Medical Information": "மருத்துவ தகவல்",
            "Enter Medical Information": "மருத்துவ தகவலை உள்ளிடவும்", "Target Language": "இலக்கு மொழி",
            "Translate Medical Information": "மருத்துவ தகவலை மொழிபெயர்க்கவும்",
            "Appointments": "சந்திப்புகள்", "Health Assessment": "உடல்நல மதிப்பீடு",
            "Available": "கிடைக்கிறது", "AI Assistant": "AI உதவியாளர்", "Online": "ஆன்லைனில் உள்ளது",
            "Medical Knowledge": "மருத்துவ அறிவு", "8 sources": "8 ஆதாரங்கள்",
            "Use HealthAI Assistant for evidence-grounded healthcare information.": "ஆதார அடிப்படையிலான சுகாதார தகவல்களுக்கு HealthAI உதவியாளரைப் பயன்படுத்தவும்.",
            "Estimate diabetes risk using the trained machine-learning model.": "பயிற்சி பெற்ற இயந்திரக் கற்றல் மாதிரியைப் பயன்படுத்தி நீரிழிவு அபாயத்தை மதிப்பிடுங்கள்.",
            "Age": "வயது", "Gender": "பாலினம்", "Blood Glucose (mg/dL)": "இரத்த குளுக்கோஸ் (mg/dL)",
            "HbA1c (%)": "HbA1c (%)", "Total Cholesterol (mg/dL)": "மொத்த கொழுப்பு (mg/dL)",
            "BMI": "BMI", "Region": "பிராந்தியம்", "Socioeconomic Status": "சமூகப் பொருளாதார நிலை",
            "Low": "குறைவு", "Middle": "நடுத்தரம்", "High": "உயரம்", "Male": "ஆண்", "Female": "பெண்",
            "Assess My Risk": "என் அபாயத்தை மதிப்பிடுக", "Risk assessment completed.": "அபாய மதிப்பீடு முடிந்தது.",
            "Estimated diabetes probability": "மதிப்பிடப்பட்ட நீரிழிவு நிகழ்தகவு",
            "This is a machine-learning risk estimate, not a medical diagnosis.": "இது இயந்திரக் கற்றல் அடிப்படையிலான அபாய மதிப்பீடு மட்டுமே; மருத்துவ நோயறிதல் அல்ல.",
            "Ask a healthcare question in your own words.": "உங்கள் சொந்த வார்த்தைகளில் சுகாதாரக் கேள்வியைக் கேளுங்கள்.",
            "Choose a commonly asked question or type your own question below.": "பொதுவாக கேட்கப்படும் கேள்வியைத் தேர்ந்தெடுக்கவும் அல்லது கீழே உங்கள் கேள்வியைத் தட்டச்சு செய்யவும்.",
            "📚 What documents and topics can I ask about?": "📚 நான் எந்த ஆவணங்கள் மற்றும் தலைப்புகள் பற்றி கேட்கலாம்?",
            "Commonly asked questions": "பொதுவாக கேட்கப்படும் கேள்விகள்",
            "Type your healthcare question...": "உங்கள் சுகாதாரக் கேள்வியைத் தட்டச்சு செய்யவும்...",
            "My requests": "எனது கோரிக்கைகள்", "Patient name": "நோயாளர் பெயர்", "Department": "துறை",
            "Preferred date": "விருப்பமான தேதி", "Preferred time": "விருப்பமான நேரம்",
            "Reason for appointment": "சந்திப்பிற்கான காரணம்", "Request Appointment": "சந்திப்பைக் கோருக",
            "Please enter patient name.": "நோயாளர் பெயரை உள்ளிடவும்.", "Appointment request submitted.": "சந்திப்பு கோரிக்கை சமர்ப்பிக்கப்பட்டது.",
            "Requested": "கோரப்பட்டது", "Find a Doctor": "மருத்துவரைக் கண்டறியவும்", "Doctor and department discovery": "மருத்துவர் மற்றும் துறை விவரங்கள்",
            "Appointment requests": "சந்திப்பு கோரிக்கைகள்", "Laboratory": "ஆய்வகம்", "Laboratory services": "ஆய்வக சேவைகள்",
            "Radiology": "கதிரியக்கவியல்", "Imaging services": "படமெடுப்பு சேவைகள்", "Pharmacy": "மருந்தகம்",
            "Medication support": "மருந்து தொடர்பான உதவி", "Emergency": "அவசர சேவை", "Emergency-care information": "அவசர சிகிச்சை தகவல்",
            "Health Records": "சுகாதார பதிவுகள்", "Digital records integration": "டிஜிட்டல் பதிவுகள் ஒருங்கிணைப்பு",
            "Insurance & Billing": "காப்பீடு மற்றும் கட்டணம்", "Coverage and billing": "காப்பீட்டு பாதுகாப்பு மற்றும் கட்டண விவரங்கள்",
            "Contact Hospital": "மருத்துவமனையைத் தொடர்புகொள்ளவும்", "Hospital support": "மருத்துவமனை உதவி",
            "Health Library": "சுகாதார நூலகம்", "Trusted health information": "நம்பகமான சுகாதார தகவல்கள்",
            "General Medicine": "பொது மருத்துவம்", "Cardiology": "இதயவியல்", "Endocrinology": "நாளமில்லா சுரப்பியல்",
            "Gynecology": "மகப்பேறு மற்றும் மகளிர் மருத்துவம்", "Pediatrics": "குழந்தை மருத்துவம்", "Radiology": "கதிரியக்கவியல்", "Neurology": "நரம்பியல்",
        },
    }

    # Additional patient-facing translations used across the complete portal.
    PATIENT_TRANSLATIONS["हिन्दी"].update({
        "Appointments": "अपॉइंटमेंट", "Health Assessment": "स्वास्थ्य आकलन",
        "Available": "उपलब्ध", "AI Assistant": "AI सहायक", "Online": "ऑनलाइन",
        "Medical Knowledge": "चिकित्सा ज्ञान", "8 sources": "8 स्रोत",
        "Use HealthAI Assistant for evidence-grounded healthcare information.": "प्रमाण-आधारित स्वास्थ्य जानकारी के लिए HealthAI सहायक का उपयोग करें।",
        "Estimate diabetes risk using the trained machine-learning model.": "प्रशिक्षित मशीन लर्निंग मॉडल का उपयोग करके मधुमेह के जोखिम का अनुमान लगाएँ।",
        "Age": "आयु", "Gender": "लिंग", "Blood Glucose (mg/dL)": "रक्त ग्लूकोज़ (mg/dL)",
        "Total Cholesterol (mg/dL)": "कुल कोलेस्ट्रॉल (mg/dL)", "Region": "क्षेत्र",
        "Socioeconomic Status": "सामाजिक-आर्थिक स्थिति", "Low": "कम", "Middle": "मध्यम", "High": "उच्च",
        "Male": "पुरुष", "Female": "महिला", "Assess My Risk": "मेरा जोखिम आकलन करें",
        "Risk assessment completed.": "जोखिम आकलन पूरा हुआ।",
        "Estimated diabetes probability": "अनुमानित मधुमेह संभावना",
        "This is a machine-learning risk estimate, not a medical diagnosis.": "यह मशीन लर्निंग आधारित जोखिम अनुमान है, चिकित्सीय निदान नहीं।",
        "Ask a healthcare question in your own words.": "अपने शब्दों में स्वास्थ्य संबंधी प्रश्न पूछें।",
        "Choose a commonly asked question or type your own question below.": "कोई सामान्य प्रश्न चुनें या नीचे अपना प्रश्न लिखें।",
        "Commonly asked questions": "अक्सर पूछे जाने वाले प्रश्न",
        "Type your healthcare question...": "अपना स्वास्थ्य संबंधी प्रश्न लिखें...",
        "My requests": "मेरे अनुरोध", "Patient name": "मरीज़ का नाम", "Department": "विभाग",
        "Preferred date": "पसंदीदा तारीख", "Preferred time": "पसंदीदा समय",
        "Reason for appointment": "अपॉइंटमेंट का कारण", "Request Appointment": "अपॉइंटमेंट का अनुरोध करें",
        "Please enter patient name.": "कृपया मरीज़ का नाम दर्ज करें।",
        "Appointment request submitted.": "अपॉइंटमेंट अनुरोध जमा हो गया।", "Requested": "अनुरोधित",
        "Find a Doctor": "डॉक्टर खोजें", "Doctor and department discovery": "डॉक्टर और विभाग की जानकारी",
        "Appointment requests": "अपॉइंटमेंट अनुरोध", "Laboratory": "प्रयोगशाला", "Laboratory services": "प्रयोगशाला सेवाएँ",
        "Radiology": "रेडियोलॉजी", "Imaging services": "इमेजिंग सेवाएँ", "Pharmacy": "फार्मेसी",
        "Medication support": "दवा संबंधी सहायता", "Emergency": "आपातकालीन सेवा", "Emergency-care information": "आपातकालीन देखभाल की जानकारी",
        "Health Records": "स्वास्थ्य रिकॉर्ड", "Digital records integration": "डिजिटल रिकॉर्ड एकीकरण",
        "Insurance & Billing": "बीमा और बिलिंग", "Coverage and billing": "बीमा कवरेज और बिलिंग",
        "Contact Hospital": "अस्पताल से संपर्क करें", "Hospital support": "अस्पताल सहायता",
        "Health Library": "स्वास्थ्य पुस्तकालय", "Trusted health information": "विश्वसनीय स्वास्थ्य जानकारी",
        "General Medicine": "सामान्य चिकित्सा", "Cardiology": "हृदय रोग विज्ञान", "Endocrinology": "अंतःस्रावी विज्ञान",
        "Gynecology": "स्त्री रोग", "Pediatrics": "बाल चिकित्सा", "Neurology": "तंत्रिका विज्ञान",
        "Close previous answer": "पिछला उत्तर बंद करें", "Previous question": "पिछला प्रश्न",
        "Sources": "स्रोत", "Choose a question": "एक प्रश्न चुनें", "Ask HealthAI": "HealthAI से पूछें",
        "Finding the best answer...": "सबसे अच्छा उत्तर खोजा जा रहा है...",
    })
    PATIENT_TRANSLATIONS["தமிழ்"].update({
        "Requested": "கோரப்பட்டது", "Close previous answer": "முந்தைய பதிலை மூடவும்",
        "Previous question": "முந்தைய கேள்வி", "Choose a question": "ஒரு கேள்வியைத் தேர்ந்தெடுக்கவும்",
        "Sources": "ஆதாரங்கள்", "Ask HealthAI": "HealthAI-யிடம் கேளுங்கள்",
        "Finding the best answer...": "சிறந்த பதிலைத் தேடுகிறது...",
    })

    PATIENT_TRANSLATIONS["हिन्दी"].update({
        "💬 Current question": "💬 वर्तमान प्रश्न",
        "🕘 Previous questions": "🕘 पिछले प्रश्न",
        "Click a question to view its answer, or remove it from history.": "उत्तर देखने के लिए किसी प्रश्न पर क्लिक करें या उसे इतिहास से हटाएँ।",
        "📖 Previous question": "📖 पिछला प्रश्न",
        "✕ Close previous answer": "✕ पिछला उत्तर बंद करें",
        "💡 Commonly asked questions": "💡 अक्सर पूछे जाने वाले प्रश्न",
        "Choose a question": "एक प्रश्न चुनें",
        "Ask HealthAI": "HealthAI से पूछें",
        "Finding the best answer...": "सबसे उपयुक्त उत्तर खोजा जा रहा है...",
        "Sources": "स्रोत",
        "HealthAI Virtual Hospital — Portfolio Demonstration": "HealthAI वर्चुअल अस्पताल — पोर्टफोलियो प्रदर्शन",
        "AI outputs are decision-support demonstrations and do not replace professional medical diagnosis, treatment or emergency care.": "AI के परिणाम निर्णय-सहायता के प्रदर्शन हैं और पेशेवर चिकित्सीय निदान, उपचार या आपातकालीन देखभाल का विकल्प नहीं हैं।",
        "What documents and topics can I ask about?": "मैं किन दस्तावेज़ों और विषयों के बारे में पूछ सकता हूँ?",
    })

    PATIENT_TRANSLATIONS["தமிழ்"].update({
        "💬 Current question": "💬 தற்போதைய கேள்வி",
        "🕘 Previous questions": "🕘 முந்தைய கேள்விகள்",
        "Click a question to view its answer, or remove it from history.": "பதிலைப் பார்க்க ஒரு கேள்வியைத் தேர்ந்தெடுக்கவும் அல்லது அதை வரலாற்றிலிருந்து நீக்கவும்.",
        "📖 Previous question": "📖 முந்தைய கேள்வி",
        "✕ Close previous answer": "✕ முந்தைய பதிலை மூடுக",
        "💡 Commonly asked questions": "💡 பொதுவாக கேட்கப்படும் கேள்விகள்",
        "Choose a question": "ஒரு கேள்வியைத் தேர்ந்தெடுக்கவும்",
        "Ask HealthAI": "HealthAI-யிடம் கேட்கவும்",
        "Finding the best answer...": "சிறந்த பதிலைத் தேடுகிறது...",
        "Sources": "ஆதாரங்கள்",
        "HealthAI Virtual Hospital — Portfolio Demonstration": "HealthAI மெய்நிகர் மருத்துவமனை — போர்ட்ஃபோலியோ விளக்கம்",
        "AI outputs are decision-support demonstrations and do not replace professional medical diagnosis, treatment or emergency care.": "AI முடிவுகள் முடிவு ஆதரவு விளக்கங்களாகும்; அவை மருத்துவ நோயறிதல், சிகிச்சை அல்லது அவசர சிகிச்சைக்கு மாற்றாகாது.",
        "What documents and topics can I ask about?": "நான் எந்த ஆவணங்கள் மற்றும் தலைப்புகள் பற்றி கேட்கலாம்?",
    })

    PATIENT_QUESTION_LABELS_TAMIL = {
        "Select a question...": "ஒரு கேள்வியைத் தேர்ந்தெடுக்கவும்...",
        "What are the symptoms of diabetes?": "நீரிழிவின் அறிகுறிகள் என்ன?",
        "What are the risk factors for diabetes?": "நீரிழிவு ஏற்படுவதற்கான அபாய காரணிகள் என்ன?",
        "How can diabetes be managed through lifestyle changes?": "வாழ்க்கை முறை மாற்றங்கள் மூலம் நீரிழிவை எவ்வாறு நிர்வகிக்கலாம்?",
        "What is high blood pressure?": "உயர் இரத்த அழுத்தம் என்றால் என்ன?",
        "What are the symptoms of hypertension?": "உயர் இரத்த அழுத்தத்தின் அறிகுறிகள் என்ன?",
        "How can I reduce my blood pressure?": "எனது இரத்த அழுத்தத்தை எவ்வாறு குறைக்கலாம்?",
        "What is a chronic illness?": "நீண்டகால நோய் என்றால் என்ன?",
        "What are noncommunicable diseases?": "தொற்றாத நோய்கள் என்றால் என்ன?",
        "Why is regular physical activity important?": "வழக்கமான உடற்பயிற்சி ஏன் முக்கியம்?",
        "What are healthy eating habits?": "ஆரோக்கியமான உணவுப் பழக்கங்கள் என்ன?",
        "When should I seek medical attention for high blood pressure?": "உயர் இரத்த அழுத்தத்திற்கு எப்போது மருத்துவரை அணுக வேண்டும்?",
        "When should I see a doctor about diabetes symptoms?": "நீரிழிவு அறிகுறிகளுக்கு எப்போது மருத்துவரை அணுக வேண்டும்?",
    }
    PATIENT_QUESTION_LABELS_HINDI = {
        "Select a question...": "एक प्रश्न चुनें...",
        "What are the symptoms of diabetes?": "मधुमेह के लक्षण क्या हैं?",
        "What are the risk factors for diabetes?": "मधुमेह के जोखिम कारक क्या हैं?",
        "How can diabetes be managed through lifestyle changes?": "जीवनशैली में बदलाव से मधुमेह को कैसे नियंत्रित किया जा सकता है?",
        "What is high blood pressure?": "उच्च रक्तचाप क्या है?",
        "What are the symptoms of hypertension?": "उच्च रक्तचाप के लक्षण क्या हैं?",
        "How can I reduce my blood pressure?": "मैं अपना रक्तचाप कैसे कम कर सकता हूँ?",
        "What is a chronic illness?": "दीर्घकालिक बीमारी क्या है?",
        "What are noncommunicable diseases?": "गैर-संचारी रोग क्या हैं?",
        "Why is regular physical activity important?": "नियमित शारीरिक गतिविधि क्यों महत्वपूर्ण है?",
        "What are healthy eating habits?": "स्वस्थ खान-पान की आदतें क्या हैं?",
        "When should I seek medical attention for high blood pressure?": "उच्च रक्तचाप के लिए मुझे कब चिकित्सा सहायता लेनी चाहिए?",
        "When should I see a doctor about diabetes symptoms?": "मधुमेह के लक्षणों के लिए मुझे डॉक्टर को कब दिखाना चाहिए?",
    }

    def pt(text):
        return PATIENT_TRANSLATIONS.get(language, {}).get(text, text)

    def translate_patient_answer(text):
        """Translate AI/RAG answers into the patient's selected language.

        The backend translation service is used only after the medical answer
        has been produced, so the Agentic AI/RAG reasoning remains unchanged.
        """
        if not text or language == "English":
            return text

        target = "Hindi" if is_hindi else "Tamil"
        code, data = call_api(
            "/predict/translation",
            {"text": text, "target_language": target},
            timeout=180,
        )
        if code == 200 and isinstance(data, dict):
            translated = (
                data.get("translated_text")
                or data.get("translation")
                or data.get("translated")
            )
            if translated:
                return translated

        # Never replace a valid medical answer with an empty/error response.
        return text

    if st.sidebar.button("🌐 மொழியை மாற்றவும்" if is_tamil else ("🌐 भाषा बदलें" if is_hindi else "🌐 Change Language")):
        st.session_state["patient_language"] = None
        st.rerun()

    patient_services = [
        "Dashboard", "❤️ Health Risk", "🤖 HealthAI Assistant",
        "📅 Appointments", "🏥 Hospital Services",
    ]
    display_services = [pt(x) for x in patient_services]
    selected_display = st.sidebar.radio(pt("Patient Services"), display_services)
    page = patient_services[display_services.index(selected_display)]

    if page == "Dashboard":


        st.title("👤 நோயாளர் டாஷ்போர்டு" if is_tamil else ("👤 रोगी डैशबोर्ड" if is_hindi else "👤 Patient Dashboard"))

        a, b, c, d = st.columns(4)
        a.metric(
            pt("Appointments"),
            len(st.session_state.appointments),
        )
        b.metric(
            pt("Health Assessment"),
            pt("Available"),
        )
        c.metric(
            pt("AI Assistant"),
            pt("Online"),
        )
        d.metric(
            pt("Medical Knowledge"),
            pt("8 sources"),
        )

        st.info(pt("Use HealthAI Assistant for evidence-grounded healthcare information."))

    # -----------------------------------------------------------------
    # HEALTH RISK
    # -----------------------------------------------------------------

    elif page == "❤️ Health Risk":

        st.title(pt("❤️ Health Risk"))
        st.caption(pt("Estimate diabetes risk using the trained machine-learning model."))

        a, b = st.columns(2)

        with a:
            age = st.number_input(
                pt("Age"),
                1,
                120,
                30,
            )

            gender = st.selectbox(
                pt("Gender"),
                [pt("Male"), pt("Female")],
            )

            glucose = st.number_input(
                pt("Blood Glucose (mg/dL)"),
                50.0,
                500.0,
                100.0,
            )

            hba1c = st.number_input(
                pt("HbA1c (%)"),
                2.0,
                20.0,
                5.5,
            )

        with b:
            cholesterol = st.number_input(
                pt("Total Cholesterol (mg/dL)"),
                50.0,
                500.0,
                180.0,
            )

            bmi = st.number_input(
                pt("BMI"),
                10.0,
                70.0,
                24.0,
            )

            region = st.text_input(
                pt("Region"),
                "Tamil Nadu",
            )

            socio = st.selectbox(
                pt("Socioeconomic Status"),
                [pt("Low"), pt("Middle"), pt("High")],
            )

        if st.button(
            pt("Assess My Risk"),
            type="primary",
        ):

            payload = {
                "Age": age,
                "Gender": gender,
                "Region": region,
                "Socioeconomic_Status": socio,
                "Symptoms": "",
                "Blood_Glucose_mg_dL": glucose,
                "HbA1c_%": hba1c,
                "Total_Cholesterol_mg_dL": cholesterol,
                "BMI": bmi,
            }

            code, data = call_api(
                "/predict/diabetes",
                payload,
                timeout=180,
            )

            if code != 200:
                render_error(code, data)
            else:
                probability = data.get("diabetes_probability")

                if probability is not None:
                    p = safe_float(probability)

                    st.success(pt("Risk assessment completed."))

                    st.metric(
                        pt("Estimated diabetes probability"),
                        f"{p:.1%}",
                    )

                    st.progress(
                        min(max(p, 0.0), 1.0)
                    )

                    st.warning(
                        pt("This is a machine-learning risk estimate, not a medical diagnosis."),
                    )

                else:
                    render_error(
                        500,
                        {
                            "message": (
                                "The risk model returned no probability."
                            )
                        },
                    )

    # -----------------------------------------------------------------
    # HEALTHAI ASSISTANT
    # -----------------------------------------------------------------

    elif page == "🤖 HealthAI Assistant":
        st.title(pt("🤖 HealthAI Assistant"))
        st.caption(pt("Ask a healthcare question in your own words."))

        st.info(pt("Choose a commonly asked question or type your own question below."))

        # -----------------------------------------------------------------
        # Agentic AI / RAG knowledge-base information
        # -----------------------------------------------------------------

        with st.expander(pt("📚 What documents and topics can I ask about?"), expanded=False):
            if is_tamil:
                st.markdown(
                    """
                    **Agentic AI** என்பது கேள்வியை சரியான சுகாதார கருவிக்கு அனுப்பும் ஒருங்கிணைப்பு அடுக்கு.

                    **RAG ஆதாரங்கள்:** இந்த திட்டத்தில் பயன்படுத்தப்படும் 8 தேர்ந்தெடுக்கப்பட்ட ஆதாரங்கள் **WHO (World Health Organization)** மற்றும் **MedlinePlus** சுகாதார உள்ளடக்கங்களிலிருந்து பெறப்பட்டவை.

                    **8 தேர்ந்தெடுக்கப்பட்ட அறிவு ஆதாரங்கள்**
                    1. **MedlinePlus — Living With Chronic Illness**
                    2. **MedlinePlus — Diabetes Self-Care**
                    3. **MedlinePlus — Health Topics**
                    4. **WHO — Diabetes**
                    5. **WHO — Hypertension**
                    6. **WHO — India Diabetes**
                    7. **WHO — India Hypertension**
                    8. **WHO — Noncommunicable Diseases**

                    மேலே உள்ள பட்டியல் திட்டத்தின் முழுமையான அறிவுத் தளத்தைக் காட்டுகிறது. கேள்வி கேட்ட பிறகு, அந்த பதிலுக்கு உண்மையில் பயன்படுத்தப்பட்ட ஆதாரங்கள் மட்டும் **ஆதாரங்கள்** பகுதியில் காட்டப்படும்.

                    **பயன்படுத்தப்படும் அறிவு ஆதாரங்கள்**
                    - WHO சுகாதார தகவல்கள்
                    - MedlinePlus சுகாதார தகவல்கள்

                    **முக்கிய தலைப்புகள்**
                    - நீரிழிவு
                    - வகை 2 நீரிழிவு சுய பராமரிப்பு மற்றும் மேலாண்மை
                    - உயர் இரத்த அழுத்தம்
                    - தொற்றாத நோய்கள் (NCDs)
                    - நீண்டகால நோய்களுடன் வாழ்தல்
                    - ஆரோக்கியமான வாழ்க்கை முறை மற்றும் உடற்பயிற்சி

                    **உதாரணக் கேள்விகள்**
                    - நீரிழிவின் அறிகுறிகள் என்ன?
                    - நீரிழிவு ஏற்படுவதற்கான அபாய காரணிகள் என்ன?
                    - வாழ்க்கை முறையில் மாற்றங்கள் மூலம் நீரிழிவை எவ்வாறு நிர்வகிக்கலாம்?
                    - உயர் இரத்த அழுத்தம் என்றால் என்ன?
                    - உயர் இரத்த அழுத்தத்தின் அறிகுறிகள் என்ன?
                    - இரத்த அழுத்தத்தை எவ்வாறு குறைக்கலாம்?
                    - நீண்டகால நோய் என்றால் என்ன?
                    - தொற்றாத நோய்கள் என்றால் என்ன?
                    - வழக்கமான உடற்பயிற்சி ஏன் முக்கியம்?
                    - ஆரோக்கியமான உணவுப் பழக்கங்கள் என்ன?

                    நம்பகமான அறிவுத் தளத்தில் போதுமான ஆதாரம் இல்லாத கேள்விகளுக்கு HealthAI பதிலை உருவாக்காமல், நம்பகமாக பதிலளிக்க முடியாது என்பதைத் தெரிவிக்கும்.
                    """
                )
            elif is_hindi:
                st.markdown(
                    """
                    **Agentic AI** एक orchestration layer के रूप में काम करता है, जो प्रश्न को उपयुक्त स्वास्थ्य टूल तक पहुँचाता है।

                    **RAG स्रोत:** इस परियोजना में उपयोग किए गए 8 चुने हुए स्रोत **WHO (World Health Organization)** और **MedlinePlus** की स्वास्थ्य सामग्री से हैं.

                    **8 चुने हुए ज्ञान स्रोत**
                    1. **MedlinePlus — Living With Chronic Illness**
                    2. **MedlinePlus — Diabetes Self-Care**
                    3. **MedlinePlus — Health Topics**
                    4. **WHO — Diabetes**
                    5. **WHO — Hypertension**
                    6. **WHO — India Diabetes**
                    7. **WHO — India Hypertension**
                    8. **WHO — Noncommunicable Diseases**

                    ऊपर दी गई सूची परियोजना के पूरे knowledge base को दिखाती है। प्रश्न पूछने के बाद, उस उत्तर के लिए वास्तव में उपयोग किए गए स्रोत ही **Sources** भाग में दिखाए जाएंगे।

                    **उपयोग किए गए ज्ञान स्रोत**
                    - **WHO (World Health Organization)** स्वास्थ्य सामग्री
                    - **MedlinePlus** स्वास्थ्य जानकारी

                    **मुख्य विषय**
                    - मधुमेह
                    - टाइप 2 मधुमेह की स्व-देखभाल और प्रबंधन
                    - उच्च रक्तचाप
                    - गैर-संचारी रोग (NCDs)
                    - दीर्घकालिक बीमारी के साथ जीवन
                    - स्वस्थ जीवनशैली और शारीरिक गतिविधि
                    """
                )
            else:
                st.markdown(
                    """
                    **Agentic AI** acts as the orchestration layer. For healthcare
                    knowledge questions, it can route the request to the
                    **evidence-grounded RAG knowledge base**.

                    **RAG sources:** The project uses 8 curated sources from **WHO
                    (World Health Organization) and MedlinePlus** healthcare content.

                    **8 curated knowledge sources**
                    1. **MedlinePlus — Living With Chronic Illness**
                    2. **MedlinePlus — Diabetes Self-Care**
                    3. **MedlinePlus — Health Topics**
                    4. **WHO — Diabetes**
                    5. **WHO — Hypertension**
                    6. **WHO — India Diabetes**
                    7. **WHO — India Hypertension**
                    8. **WHO — Noncommunicable Diseases**

                    The list above describes the complete project knowledge base. After a question is asked, only the sources actually used for that answer are shown in the answer's **Sources** section.

                    **Knowledge sources used by the project**
                    - **WHO (World Health Organization)** curated healthcare content
                    - **MedlinePlus** curated health-information content

                    **Main topics covered**
                    - Diabetes
                    - Type 2 Diabetes self-care and management
                    - Hypertension / high blood pressure
                    - Noncommunicable diseases (NCDs)
                    - Living with chronic illness
                    - Healthy lifestyle and physical activity

                    **Examples of questions you can ask**
                    - What are the symptoms of diabetes?
                    - What are the risk factors for diabetes?
                    - How can diabetes be managed through lifestyle changes?
                    - What is high blood pressure?
                    - What are the symptoms of hypertension?
                    - How can I reduce my blood pressure?
                    - What is a chronic illness?
                    - What are noncommunicable diseases?
                    - Why is regular physical activity important?
                    - What are healthy eating habits?
                    """
                )

        RAG_SOURCE_CATALOG = [
            {"document": "MedlinePlus — Living With Chronic Illness", "topic": "Living with chronic illness", "source_id": "medlineplus_chronic_illness"},
            {"document": "MedlinePlus — Diabetes Self-Care", "topic": "Diabetes self-care", "source_id": "medlineplus_diabetes_selfcare"},
            {"document": "MedlinePlus — Health Topics", "topic": "General health topics", "source_id": "medlineplus_health_topics"},
            {"document": "WHO — Diabetes", "topic": "Diabetes", "source_id": "who_diabetes"},
            {"document": "WHO — Hypertension", "topic": "Hypertension", "source_id": "who_hypertension"},
            {"document": "WHO — India Diabetes", "topic": "Diabetes in India", "source_id": "who_india_diabetes"},
            {"document": "WHO — India Hypertension", "topic": "Hypertension in India", "source_id": "who_india_hypertension"},
            {"document": "WHO — Noncommunicable Diseases", "topic": "Noncommunicable diseases", "source_id": "who_ncd"},
        ]

        STANDARD_ANSWERS = {
            "What are the symptoms of diabetes?": (
                "Common symptoms of diabetes can include increased thirst, "
                "frequent urination, increased hunger, unexplained weight loss, "
                "tiredness, blurred vision, and slow-healing wounds. Some people "
                "may have few or no symptoms, especially early on. A blood glucose "
                "test is needed to diagnose diabetes."
            ),
            "What are the risk factors for diabetes?": (
                "Risk factors for type 2 diabetes include overweight or obesity, "
                "low physical activity, a family history of diabetes, increasing "
                "age, and a history of high blood glucose. Having a risk factor "
                "does not mean that you will develop diabetes."
            ),
            "How can diabetes be managed through lifestyle changes?": (
                "Helpful lifestyle measures include regular physical activity, a "
                "balanced eating pattern, maintaining a healthy weight, taking "
                "prescribed medicines as directed, and monitoring blood glucose "
                "when advised by a healthcare professional."
            ),
            "What is high blood pressure?": (
                "High blood pressure, or hypertension, means blood pressure stays "
                "higher than the recommended range over time. It often causes no "
                "obvious symptoms, which is why regular blood-pressure checks are "
                "important."
            ),
            "What are the symptoms of hypertension?": (
                "High blood pressure often has no symptoms. Very high blood pressure "
                "can sometimes cause severe headache, vision problems, chest pain, "
                "shortness of breath, confusion, or other serious symptoms and needs "
                "urgent medical assessment."
            ),
            "How can I reduce my blood pressure?": (
                "Blood pressure can often be improved through regular physical "
                "activity, reducing excess salt, eating a balanced diet, maintaining "
                "a healthy weight, avoiding tobacco, limiting alcohol, managing "
                "stress, and taking prescribed blood-pressure medicines correctly."
            ),
            "What is a chronic illness?": (
                "A chronic illness is a health condition that lasts for a long time "
                "and may need ongoing monitoring or treatment. Examples include "
                "diabetes, hypertension, asthma, and some heart diseases."
            ),
            "What are noncommunicable diseases?": (
                "Noncommunicable diseases, or NCDs, are conditions that are not "
                "spread from person to person. Major examples include cardiovascular "
                "disease, cancer, chronic respiratory disease, and diabetes."
            ),
            "Why is regular physical activity important?": (
                "Regular physical activity supports heart health, helps with weight "
                "management, improves blood glucose control, and can reduce the risk "
                "of several chronic diseases. Activity should be appropriate for your "
                "age and health status."
            ),
            "What are healthy eating habits?": (
                "Healthy eating generally means choosing a variety of vegetables, "
                "fruits, whole grains or other high-fibre foods, legumes, and suitable "
                "protein sources, while limiting excess salt, added sugars, and highly "
                "processed foods. Portion sizes also matter."
            ),
            "When should I seek medical attention for high blood pressure?": (
                "Seek urgent medical care for very high blood pressure accompanied by "
                "symptoms such as chest pain, severe headache, shortness of breath, "
                "weakness, confusion, or vision changes. For repeatedly high readings "
                "without emergency symptoms, arrange a medical review."
            ),
            "When should I see a doctor about diabetes symptoms?": (
                "See a healthcare professional if you have persistent symptoms such as "
                "unusual thirst, frequent urination, unexplained weight loss, marked "
                "tiredness, or blurred vision. Testing is needed to determine whether "
                "diabetes is present."
            ),
        }

        LOCALIZED_STANDARD_ANSWERS = {
            "தமிழ்": {
                "What are the symptoms of diabetes?": "நீரிழிவின் பொதுவான அறிகுறிகளில் அதிக தாகம், அடிக்கடி சிறுநீர் கழித்தல், அதிக பசி, காரணமின்றி உடல் எடை குறைதல், சோர்வு, மங்கலான பார்வை மற்றும் காயங்கள் மெதுவாக ஆறுதல் ஆகியவை அடங்கும். சிலருக்கு, குறிப்பாக ஆரம்ப நிலையில், மிகக் குறைவான அறிகுறிகளோ அல்லது எந்த அறிகுறிகளும் இல்லாமலோ இருக்கலாம். நீரிழிவை கண்டறிய இரத்த குளுக்கோஸ் பரிசோதனை தேவைப்படுகிறது.",
                "What are the risk factors for diabetes?": "வகை 2 நீரிழிவுக்கான அபாய காரணிகளில் அதிக உடல் எடை அல்லது உடல் பருமன், குறைந்த உடல் செயல்பாடு, குடும்பத்தில் நீரிழிவு வரலாறு, வயது அதிகரித்தல் மற்றும் அதிக இரத்த குளுக்கோஸ் வரலாறு ஆகியவை அடங்கும். ஒரு அபாய காரணி இருப்பது மட்டும் நீரிழிவு நிச்சயமாக ஏற்படும் என்று அர்த்தமல்ல.",
                "How can diabetes be managed through lifestyle changes?": "வழக்கமான உடற்பயிற்சி, சமநிலையான உணவுமுறை, ஆரோக்கியமான உடல் எடையை பராமரித்தல், பரிந்துரைக்கப்பட்ட மருந்துகளை மருத்துவர் கூறியபடி எடுத்துக்கொள்ளுதல் மற்றும் சுகாதார நிபுணர் அறிவுறுத்தினால் இரத்த குளுக்கோஸை கண்காணித்தல் ஆகியவை உதவக்கூடும்.",
                "What is high blood pressure?": "உயர் இரத்த அழுத்தம் அல்லது ஹைப்பர்டென்ஷன் என்பது காலப்போக்கில் இரத்த அழுத்தம் பரிந்துரைக்கப்பட்ட அளவை விட அதிகமாக இருப்பதாகும். இது பெரும்பாலும் வெளிப்படையான அறிகுறிகளை ஏற்படுத்தாது; அதனால் வழக்கமான இரத்த அழுத்த பரிசோதனை முக்கியம்.",
                "What are the symptoms of hypertension?": "உயர் இரத்த அழுத்தத்திற்கு பெரும்பாலும் அறிகுறிகள் இருக்காது. மிகவும் அதிகமான இரத்த அழுத்தம் சில நேரங்களில் கடுமையான தலைவலி, பார்வை பிரச்சினைகள், மார்பு வலி, மூச்சுத்திணறல் அல்லது குழப்பம் போன்ற தீவிர அறிகுறிகளை ஏற்படுத்தலாம்; அப்போது உடனடி மருத்துவ மதிப்பீடு தேவை.",
                "How can I reduce my blood pressure?": "வழக்கமான உடற்பயிற்சி, அதிகப்படியான உப்பை குறைத்தல், சமநிலையான உணவு, ஆரோக்கியமான உடல் எடையை பராமரித்தல், புகையிலை தவிர்த்தல், மதுபானத்தை கட்டுப்படுத்துதல், மன அழுத்தத்தை நிர்வகித்தல் மற்றும் பரிந்துரைக்கப்பட்ட இரத்த அழுத்த மருந்துகளை சரியாக எடுத்துக்கொள்ளுதல் ஆகியவை இரத்த அழுத்தத்தை மேம்படுத்த உதவும்.",
                "What is a chronic illness?": "நீண்டகால நோய் என்பது நீண்ட காலம் நீடிக்கும் மற்றும் தொடர்ந்து கண்காணிப்பு அல்லது சிகிச்சை தேவைப்படக்கூடிய உடல்நிலை. நீரிழிவு, உயர் இரத்த அழுத்தம், ஆஸ்துமா மற்றும் சில இதய நோய்கள் இதற்கான எடுத்துக்காட்டுகள்.",
                "What are noncommunicable diseases?": "தொற்றாத நோய்கள் அல்லது NCDs என்பது ஒருவரிடமிருந்து மற்றொருவருக்கு பரவாத நோய்களாகும். இதய மற்றும் இரத்த நாள நோய்கள், புற்றுநோய், நீண்டகால சுவாச நோய்கள் மற்றும் நீரிழிவு ஆகியவை முக்கிய எடுத்துக்காட்டுகள்.",
                "Why is regular physical activity important?": "வழக்கமான உடற்பயிற்சி இதய ஆரோக்கியத்தை ஆதரிக்கிறது, உடல் எடையை நிர்வகிக்க உதவுகிறது, இரத்த குளுக்கோஸ் கட்டுப்பாட்டை மேம்படுத்துகிறது மற்றும் பல நீண்டகால நோய்களின் அபாயத்தை குறைக்கலாம். உங்கள் வயது மற்றும் உடல்நிலைக்கு ஏற்ற செயல்பாட்டைத் தேர்ந்தெடுக்க வேண்டும்.",
                "What are healthy eating habits?": "ஆரோக்கியமான உணவுப் பழக்கம் என்பது பலவகையான காய்கறிகள், பழங்கள், முழுத்தானியங்கள் அல்லது நார்ச்சத்து அதிகமான உணவுகள், பருப்பு வகைகள் மற்றும் ஏற்ற புரத உணவுகளைத் தேர்ந்தெடுப்பதாகும். அதிக உப்பு, சேர்க்கப்பட்ட சர்க்கரை மற்றும் அதிகமாக பதப்படுத்தப்பட்ட உணவுகளை கட்டுப்படுத்துவது முக்கியம். உணவின் அளவும் முக்கியம்.",
                "When should I seek medical attention for high blood pressure?": "மிகவும் அதிகமான இரத்த அழுத்தத்துடன் மார்பு வலி, கடுமையான தலைவலி, மூச்சுத்திணறல், உடல் பலவீனம், குழப்பம் அல்லது பார்வை மாற்றங்கள் போன்ற அறிகுறிகள் இருந்தால் உடனடி மருத்துவ உதவியைப் பெறுங்கள். அவசர அறிகுறிகள் இல்லாமல் இரத்த அழுத்தம் தொடர்ந்து அதிகமாக இருந்தால் மருத்துவ பரிசோதனைக்கு ஏற்பாடு செய்யுங்கள்.",
                "When should I see a doctor about diabetes symptoms?": "அசாதாரணமான தாகம், அடிக்கடி சிறுநீர் கழித்தல், காரணமின்றி உடல் எடை குறைதல், அதிக சோர்வு அல்லது மங்கலான பார்வை போன்ற அறிகுறிகள் தொடர்ந்து இருந்தால் சுகாதார நிபுணரை அணுகுங்கள். நீரிழிவு உள்ளதா என்பதைத் தீர்மானிக்க பரிசோதனை தேவைப்படுகிறது.",
            },
            "हिन्दी": {
                "What are the symptoms of diabetes?": "मधुमेह के सामान्य लक्षणों में बहुत अधिक प्यास लगना, बार-बार पेशाब आना, अधिक भूख लगना, बिना कारण वजन कम होना, थकान, धुंधला दिखाई देना और घावों का धीरे भरना शामिल हो सकता है। कुछ लोगों में, खासकर शुरुआत में, बहुत कम या कोई लक्षण नहीं होते। मधुमेह का निदान करने के लिए रक्त ग्लूकोज़ की जाँच आवश्यक होती है।",
                "What are the risk factors for diabetes?": "टाइप 2 मधुमेह के जोखिम कारकों में अधिक वजन या मोटापा, कम शारीरिक गतिविधि, परिवार में मधुमेह का इतिहास, बढ़ती उम्र और पहले उच्च रक्त ग्लूकोज़ होना शामिल है। जोखिम कारक होने का मतलब यह नहीं है कि आपको मधुमेह निश्चित रूप से होगा।",
                "How can diabetes be managed through lifestyle changes?": "नियमित शारीरिक गतिविधि, संतुलित भोजन, स्वस्थ वजन बनाए रखना, डॉक्टर द्वारा बताई गई दवाएँ सही तरीके से लेना और स्वास्थ्यकर्मी की सलाह के अनुसार रक्त ग्लूकोज़ की निगरानी करना मधुमेह के प्रबंधन में मदद कर सकता है।",
                "What is high blood pressure?": "उच्च रक्तचाप या हाइपरटेंशन का अर्थ है कि समय के साथ रक्तचाप अनुशंसित सीमा से अधिक बना रहता है। इसमें अक्सर स्पष्ट लक्षण नहीं होते, इसलिए नियमित रक्तचाप की जाँच महत्वपूर्ण है।",
                "What are the symptoms of hypertension?": "उच्च रक्तचाप में अक्सर कोई लक्षण नहीं होते। बहुत अधिक रक्तचाप के कारण कभी-कभी तेज सिरदर्द, दृष्टि संबंधी समस्या, सीने में दर्द, सांस लेने में तकलीफ या भ्रम जैसे गंभीर लक्षण हो सकते हैं और तत्काल चिकित्सीय मूल्यांकन की आवश्यकता होती है।",
                "How can I reduce my blood pressure?": "नियमित शारीरिक गतिविधि, अधिक नमक कम करना, संतुलित आहार, स्वस्थ वजन बनाए रखना, तंबाकू से बचना, शराब को सीमित करना, तनाव का प्रबंधन करना और निर्धारित रक्तचाप की दवाएँ सही तरीके से लेना रक्तचाप को बेहतर करने में मदद कर सकता है।",
                "What is a chronic illness?": "दीर्घकालिक बीमारी ऐसी स्वास्थ्य स्थिति है जो लंबे समय तक रहती है और जिसके लिए लगातार निगरानी या उपचार की आवश्यकता हो सकती है। मधुमेह, उच्च रक्तचाप, अस्थमा और कुछ हृदय रोग इसके उदाहरण हैं।",
                "What are noncommunicable diseases?": "गैर-संचारी रोग या NCD ऐसी स्थितियाँ हैं जो एक व्यक्ति से दूसरे व्यक्ति में नहीं फैलतीं। हृदय और रक्त वाहिका रोग, कैंसर, दीर्घकालिक श्वसन रोग और मधुमेह इसके प्रमुख उदाहरण हैं।",
                "Why is regular physical activity important?": "नियमित शारीरिक गतिविधि हृदय स्वास्थ्य में मदद करती है, वजन प्रबंधन में सहायक होती है, रक्त ग्लूकोज़ नियंत्रण में सुधार कर सकती है और कई दीर्घकालिक रोगों के जोखिम को कम कर सकती है। गतिविधि आपकी उम्र और स्वास्थ्य स्थिति के अनुसार होनी चाहिए।",
                "What are healthy eating habits?": "स्वस्थ भोजन का अर्थ है विभिन्न प्रकार की सब्जियाँ, फल, साबुत अनाज या अन्य उच्च-फाइबर खाद्य पदार्थ, दालें और उपयुक्त प्रोटीन स्रोत चुनना। अधिक नमक, अतिरिक्त चीनी और अत्यधिक प्रसंस्कृत खाद्य पदार्थों को सीमित करना चाहिए। भोजन की मात्रा भी महत्वपूर्ण है।",
                "When should I seek medical attention for high blood pressure?": "यदि बहुत अधिक रक्तचाप के साथ सीने में दर्द, तेज सिरदर्द, सांस लेने में तकलीफ, कमजोरी, भ्रम या दृष्टि में बदलाव जैसे लक्षण हों, तो तुरंत चिकित्सा सहायता लें। यदि आपातकालीन लक्षण नहीं हैं लेकिन रक्तचाप बार-बार अधिक आता है, तो डॉक्टर से जाँच कराएँ।",
                "When should I see a doctor about diabetes symptoms?": "यदि असामान्य प्यास, बार-बार पेशाब आना, बिना कारण वजन कम होना, बहुत अधिक थकान या धुंधला दिखाई देना जैसे लक्षण लगातार बने रहें, तो स्वास्थ्यकर्मी से मिलें। मधुमेह है या नहीं यह निर्धारित करने के लिए जाँच आवश्यक है।",
            },
        }


        # -----------------------------------------------------------------
        # Conversation history
        #
        # Keep the existing session-state structure so previous functionality
        # remains compatible. Only the display is changed:
        #   - current question + answer are shown on the main page
        #   - older questions are shown as a clickable list
        #   - clicking an older question shows only that question + its answer
        # -----------------------------------------------------------------

        pairs = []
        current_pair = None

        i = 0
        while i < len(st.session_state.chat):
            message = st.session_state.chat[i]

            if message.get("role") == "user":
                question = message.get("text", "")
                answer_message = None
                answer_index = None

                if i + 1 < len(st.session_state.chat):
                    next_message = st.session_state.chat[i + 1]
                    if next_message.get("role") == "assistant":
                        answer_message = next_message
                        answer_index = i + 1

                if answer_message:
                    pairs.append(
                        {
                            "question": question,
                            "answer": answer_message.get("text", ""),
                            "sources": answer_message.get("sources", []) or [],
                            "user_index": i,
                            "assistant_index": answer_index,
                        }
                    )
                    i += 2
                    continue

                pairs.append(
                    {
                        "question": question,
                        "answer": "",
                        "sources": [],
                        "user_index": i,
                        "assistant_index": None,
                    }
                )

            i += 1

        # The newest Q&A is ALWAYS shown first on the page.
        # This keeps the current answer above the history and FAQ area.
        if pairs:
            current_pair = pairs[-1]

            st.subheader(pt("💬 Current question"))

            with st.chat_message("user"):
                current_question_display = (
                    PATIENT_QUESTION_LABELS_TAMIL.get(current_pair["question"], current_pair["question"])
                    if is_tamil
                    else PATIENT_QUESTION_LABELS_HINDI.get(current_pair["question"], current_pair["question"])
                    if is_hindi
                    else current_pair["question"]
                )
                st.write(current_question_display)

            if current_pair["answer"]:
                with st.chat_message("assistant"):
                    # Always localize the answer at display time as well.
                    # This also fixes answers that were already stored in the
                    # session before the patient selected Tamil/Hindi.
                    current_answer_display = (
                        LOCALIZED_STANDARD_ANSWERS.get(language, {}).get(
                            current_pair["question"]
                        )
                        or current_pair["answer"]
                    )
                    st.write(current_answer_display)

                    if current_pair.get("sources"):
                        with st.expander(pt("Sources")):
                            for source in current_pair["sources"]:
                                st.write(
                                    f"• {source.get('document')} — "
                                    f"{source.get('topic')}"
                                )

            previous_pairs = pairs[:-1]

            if previous_pairs:
                st.subheader(pt("🕘 Previous questions"))
                st.caption(pt("Click a question to view its answer, or remove it from history."))

                # Newest previous question appears first.
                for index in range(len(previous_pairs) - 1, -1, -1):
                    previous = previous_pairs[index]

                    question_col, remove_col = st.columns([8, 1])

                    with question_col:
                        previous_question_display = (
                            PATIENT_QUESTION_LABELS_TAMIL.get(previous["question"], previous["question"])
                            if is_tamil
                            else PATIENT_QUESTION_LABELS_HINDI.get(previous["question"], previous["question"])
                            if is_hindi
                            else previous["question"]
                        )
                        if st.button(
                            previous_question_display,
                            key=f"previous_question_{index}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_previous_question = index
                            st.rerun()

                    with remove_col:
                        if st.button(
                            "🗑️",
                            key=f"remove_previous_question_{index}",
                            help="Remove this question and answer from history",
                            use_container_width=True,
                        ):
                            user_index = previous["user_index"]
                            assistant_index = previous["assistant_index"]

                            # Remove the pair from session history.
                            indices_to_remove = [user_index]
                            if assistant_index is not None:
                                indices_to_remove.append(assistant_index)

                            st.session_state.chat = [
                                message
                                for position, message
                                in enumerate(st.session_state.chat)
                                if position not in indices_to_remove
                            ]

                            # Clear the selected previous-answer view if the
                            # deleted item was the one currently selected.
                            st.session_state.selected_previous_question = None
                            st.rerun()

        selected_index = st.session_state.get(
            "selected_previous_question",
            None,
        )

        # Show a selected previous answer below the history.
        if (
            selected_index is not None
            and 0 <= selected_index < len(pairs) - 1
        ):
            selected = pairs[selected_index]

            st.subheader(pt("📖 Previous question"))

            with st.chat_message("user"):
                st.write(selected["question"])

            with st.chat_message("assistant"):
                # Localize previously stored FAQ answers too.
                selected_answer_display = (
                    LOCALIZED_STANDARD_ANSWERS.get(language, {}).get(
                        selected["question"]
                    )
                    or selected["answer"]
                )
                st.write(selected_answer_display)

                if selected.get("sources"):
                    with st.expander(pt("Sources")):
                        for source in selected["sources"]:
                            st.write(
                                f"• {source.get('document')} — "
                                f"{source.get('topic')}"
                            )

            if st.button("✕ " + pt("Close previous answer")):
                st.session_state.selected_previous_question = None
                st.rerun()

        st.subheader("💡 " + pt("Commonly asked questions"))

        standard_questions = [
            "What are the symptoms of diabetes?",
            "What are the risk factors for diabetes?",
            "How can diabetes be managed through lifestyle changes?",
            "What is high blood pressure?",
            "What are the symptoms of hypertension?",
            "How can I reduce my blood pressure?",
            "What is a chronic illness?",
            "What are noncommunicable diseases?",
            "Why is regular physical activity important?",
            "What are healthy eating habits?",
            "When should I seek medical attention for high blood pressure?",
            "When should I see a doctor about diabetes symptoms?",
        ]

        label_map = PATIENT_QUESTION_LABELS_TAMIL if is_tamil else (PATIENT_QUESTION_LABELS_HINDI if is_hindi else {})
        display_questions = [label_map.get("Select a question...", "Select a question...")] + [label_map.get(q, q) for q in standard_questions]
        selected_display_question = st.selectbox(
            pt("Choose a question"),
            display_questions,
        )
        reverse_question = {v: k for k, v in label_map.items()}
        selected_question = reverse_question.get(selected_display_question, selected_display_question)

        ask_standard = st.button(
            pt("Ask HealthAI"),
            type="secondary",
            disabled=selected_question == "Select a question...",
        )

        question = st.chat_input(pt("Type your healthcare question..."))

        if ask_standard:
            question = selected_question

        if question:
            # A new question always returns the user to the current-question
            # view rather than leaving an older answer selected.
            st.session_state.selected_previous_question = None

            # FAQ answers are deliberately immediate and do not call the slow
            # Agentic/RAG pipeline.
            standard_answer = STANDARD_ANSWERS.get(question)

            if standard_answer:
                localized_answer = LOCALIZED_STANDARD_ANSWERS.get(language, {}).get(question)
                if not localized_answer:
                    localized_answer = translate_patient_answer(standard_answer)
                st.session_state.chat.append(
                    {"role": "user", "text": question}
                )
                standard_source_map = {
                    "What are the symptoms of diabetes?": ["medlineplus_health_topics", "who_diabetes"],
                    "What are the risk factors for diabetes?": ["who_diabetes"],
                    "How can diabetes be managed through lifestyle changes?": ["medlineplus_diabetes_selfcare", "who_diabetes"],
                    "What is high blood pressure?": ["who_hypertension"],
                    "What are the symptoms of hypertension?": ["medlineplus_health_topics", "who_hypertension"],
                    "How can I reduce my blood pressure?": ["who_hypertension", "who_ncd"],
                    "What is a chronic illness?": ["medlineplus_chronic_illness"],
                    "What are noncommunicable diseases?": ["who_ncd"],
                    "Why is regular physical activity important?": ["who_ncd", "medlineplus_diabetes_selfcare"],
                    "What are healthy eating habits?": ["medlineplus_diabetes_selfcare", "who_diabetes"],
                    "When should I seek medical attention for high blood pressure?": ["who_hypertension"],
                    "When should I see a doctor about diabetes symptoms?": ["medlineplus_health_topics", "who_diabetes"],
                }
                source_ids = standard_source_map.get(question, [])
                used_sources = [
                    source for source in RAG_SOURCE_CATALOG
                    if source["source_id"] in source_ids
                ]
                st.session_state.chat.append(
                    {
                        "role": "assistant",
                        "text": localized_answer,
                        "sources": used_sources,
                    }
                )
                st.rerun()

            st.session_state.chat.append(
                {"role": "user", "text": question}
            )

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner(pt("Finding the best answer...")):
                    # Primary path: Agentic AI.
                    code, data = call_api(
                        "/agent/chat",
                        {"query": question},
                        timeout=180,
                    )

                answer = None
                execution = {}
                status = None

                if code == 200 and isinstance(data, dict):
                    execution = data.get("execution_result", {}) or {}
                    answer = data.get("answer") or execution.get("answer")
                    status = data.get("status") or execution.get("status")

                # Patient-facing reliability fallback:
                # if the agent path cannot produce an answer, query the
                # evidence-grounded RAG service directly.
                if not answer:
                    rag_code, rag_data = call_api(
                        "/predict/rag",
                        {"query": question},
                        timeout=180,
                    )

                    if rag_code == 200 and isinstance(rag_data, dict):
                        answer = (
                            rag_data.get("answer")
                            or rag_data.get("execution_result", {}).get("answer")
                        )
                        execution = (
                            rag_data.get("execution_result")
                            or rag_data
                        )
                        status = rag_data.get("status", "success")

                if answer:
                    # Translate the final Agentic/RAG response for the selected
                    # patient language. Curated FAQ answers use the controlled
                    # medical translations above; free-form answers use FastAPI.
                    answer = translate_patient_answer(answer)
                    if status == "abstain":
                        st.warning(answer)
                    else:
                        st.write(answer)

                    sources = execution.get("sources", [])

                    if sources:
                        with st.expander("ஆதாரங்கள்" if is_tamil else "Sources"):
                            for source in sources:
                                st.write(
                                    f"• {source.get('document')} — "
                                    f"{source.get('topic')}"
                                )

                    st.session_state.chat.append(
                        {
                            "role": "assistant",
                            "text": answer,
                            "sources": sources,
                        }
                    )

                else:
                    message = (
                        "இந்தக் கேள்விக்கு நம்பகமாக பதிலளிக்க போதுமான நம்பகமான தகவல் என்னிடம் இல்லை." if is_tamil else
                        "I don't have enough trusted information to answer that question reliably."
                    )
                    st.warning(message)

                    st.session_state.chat.append(
                        {
                            "role": "assistant",
                            "text": message,
                            "sources": [],
                        }
                    )

            # The completed Q&A is already rendered above the history
            # in this run, so no rerun is needed here.

    # -----------------------------------------------------------------
    # APPOINTMENTS
    # -----------------------------------------------------------------

    elif page == "📅 Appointments":

        st.title("📅 சந்திப்புகள்" if is_tamil else ("📅 अपॉइंटमेंट" if is_hindi else "📅 Appointments"))

        a, b = st.columns(2)

        with a:
            name = st.text_input(
                pt("Patient name")
            )

            department = st.selectbox(
                pt("Department"),
                [
                    pt("General Medicine"), pt("Cardiology"), pt("Endocrinology"),
                    pt("Gynecology"), pt("Pediatrics"), pt("Radiology"), pt("Neurology"),
                ],
            )

        with b:
            preferred_date = st.date_input(
                pt("Preferred date"),
                min_value=date.today(),
                value=date.today() + timedelta(days=1),
            )

            preferred_time = st.selectbox(
                pt("Preferred time"),
                [
                    "09:00 AM",
                    "10:00 AM",
                    "11:00 AM",
                    "02:00 PM",
                    "03:00 PM",
                    "04:00 PM",
                ],
            )

        reason = st.text_area(
            pt("Reason for appointment")
        )

        if st.button(
            pt("Request Appointment"),
            type="primary",
        ):

            if not name.strip():
                st.error(
                    pt("Please enter patient name.")
                )
            else:

                st.session_state.appointments.append(
                    {
                        "patient": name,
                        "department": department,
                        "date": str(preferred_date),
                        "time": preferred_time,
                        "reason": reason,
                        "status": "Requested",
                    }
                )

                st.success(
                    pt("Appointment request submitted.")
                )

        if st.session_state.appointments:

            st.subheader(pt("My requests"))

            for appointment in st.session_state.appointments:

                st.write(
                    f"**{appointment['department']}** — "
                    f"{appointment['date']} "
                    f"{appointment['time']} — "
                    f"{pt(appointment['status'])} — "
                    f"{appointment['patient']}"
                )

    # -----------------------------------------------------------------
    # HOSPITAL SERVICES
    # -----------------------------------------------------------------

    elif page == "🏥 Hospital Services":

        st.title("🏥 மருத்துவமனை சேவைகள்" if is_tamil else ("🏥 अस्पताल सेवाएँ" if is_hindi else "🏥 Hospital Services"))

        services = [
            ("👨‍⚕️", pt("Find a Doctor"), pt("Doctor and department discovery")),
            ("📅", pt("Appointments"), pt("Appointment requests")),
            ("🧪", pt("Laboratory"), pt("Laboratory services")),
            ("🫁", pt("Radiology"), pt("Imaging services")),
            ("💊", pt("Pharmacy"), pt("Medication support")),
            ("🚑", pt("Emergency"), pt("Emergency-care information")),
            ("📋", pt("Health Records"), pt("Digital records integration")),
            ("💳", pt("Insurance & Billing"), pt("Coverage and billing")),
            ("📞", pt("Contact Hospital"), pt("Hospital support")),
            ("📚", pt("Health Library"), pt("Trusted health information")),
        ]

        cols = st.columns(3)

        for i, (icon, name, description) in enumerate(services):

            with cols[i % 3]:

                st.markdown(
                    f"""
                    <div class="card">
                        <h3>{icon} {name}</h3>
                        <p>{description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


st.divider()

# Keep the Doctor Portal and Home page in English.
# Patient-language localization applies only while the Patient Portal is active.
if portal == "👤 Patient Portal":
    _patient_language = st.session_state.get("patient_language", "English")
    _disclaimer_title = {
        "தமிழ்": "HealthAI மெய்நிகர் மருத்துவமனை — போர்ட்ஃபோலியோ விளக்கம்",
        "हिन्दी": "HealthAI वर्चुअल अस्पताल — पोर्टफोलियो प्रदर्शन",
    }.get(_patient_language, "HealthAI Virtual Hospital — Portfolio Demonstration")
    _disclaimer_text = {
        "தமிழ்": "AI முடிவுகள் முடிவு ஆதரவு விளக்கங்களாகும்; அவை மருத்துவ நோயறிதல், சிகிச்சை அல்லது அவசர சிகிச்சைக்கு மாற்றாகாது.",
        "हिन्दी": "AI के परिणाम निर्णय-सहायता के प्रदर्शन हैं और पेशेवर चिकित्सीय निदान, उपचार या आपातकालीन देखभाल का विकल्प नहीं हैं।",
    }.get(_patient_language, "AI outputs are decision-support demonstrations and do not replace professional medical diagnosis, treatment or emergency care.")
else:
    _disclaimer_title = "HealthAI Virtual Hospital — Portfolio Demonstration"
    _disclaimer_text = "AI outputs are decision-support demonstrations and do not replace professional medical diagnosis, treatment or emergency care."

st.markdown(
    f"<div class=\"disclaimer\"><b>{_disclaimer_title}</b><br>{_disclaimer_text}</div>",
    unsafe_allow_html=True,
)


