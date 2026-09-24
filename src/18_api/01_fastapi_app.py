"""
HealthAI Intelligent Platform
--------------------------------
Production-style FastAPI integration layer.

Capabilities:
1. Diabetes risk prediction
2. Hospital length-of-stay prediction
3. Primary diagnosis prediction
4. Medical NER
5. Sentiment analysis
6. Medical RAG
7. Agentic AI chat
8. Chest X-ray CNN analysis

Important:
- API does NOT train models.
- API reuses existing trained artifacts and adapters.
- X-ray endpoint passes a normalized NumPy image array
  to the existing X-ray adapter.
"""

from pathlib import Path
from functools import lru_cache
import importlib.util
import json
import subprocess
import sys
import time


# ---------------------------------------------------------------------
# FastAPI imports
# ---------------------------------------------------------------------

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AGENT_DIR = PROJECT_ROOT / "src" / "17_agentic_ai"

DEEP_LEARNING_DIR = PROJECT_ROOT / "src" / "16_deep_learning"

if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title="HealthAI Intelligent Platform",
    description=(
        "AI-powered healthcare platform integrating "
        "machine learning, deep learning, medical NLP, "
        "RAG and Agentic AI."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------
# Generic adapter loader
# ---------------------------------------------------------------------

def load_adapter(filename: str):
    """
    Dynamically load an adapter from src/17_agentic_ai.
    """

    path = AGENT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Adapter not found: {path}"
        )

    module_name = (
        "healthai_api_"
        + filename.replace(".py", "")
        .replace("-", "_")
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load adapter: {filename}"
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# ---------------------------------------------------------------------
# Cached adapters
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def diabetes_adapter():
    return load_adapter(
        "07_diabetes_prediction_adapter.py"
    )


@lru_cache(maxsize=1)
def los_adapter():
    return load_adapter(
        "08_hospital_los_adapter.py"
    )


@lru_cache(maxsize=1)
def diagnosis_adapter():
    return load_adapter(
        "05_primary_diagnosis_adapter.py"
    )


@lru_cache(maxsize=1)
def ner_adapter():
    return load_adapter(
        "04C_medical_ner_adapter.py"
    )


@lru_cache(maxsize=1)
def sentiment_adapter():
    return load_adapter(
        "16_sentiment_analysis_adapter.py"
    )


@lru_cache(maxsize=1)
def rag_adapter():
    return load_adapter(
        "17_medical_rag_adapter.py"
    )


@lru_cache(maxsize=1)
def clustering_adapter():
    return load_adapter(
        "12_patient_clustering_adapter.py"
    )


@lru_cache(maxsize=1)
def association_adapter():
    return load_adapter(
        "13_association_analysis_adapter.py"
    )


@lru_cache(maxsize=1)
def deterioration_adapter():
    return load_adapter(
        "15_deterioration_prediction_adapter.py"
    )


@lru_cache(maxsize=1)
def translation_adapter():
    adapter_path = PROJECT_ROOT / "src" / "17_agentic_ai" / "20_medical_translation_adapter.py"

    spec = importlib.util.spec_from_file_location(
        "healthai_api_translation_adapter",
        adapter_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Unable to load medical translation adapter."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules["healthai_api_translation_adapter"] = module
    spec.loader.exec_module(module)

    return module


@lru_cache(maxsize=1)
def xray_adapter():
    """
    Directly load the known-good X-ray model service.

    This bypasses the generic adapter loader because the CNN
    requires the Keras InputLayer compatibility loader contained
    in model_service.py.
    """

    model_service_path = (
        PROJECT_ROOT
        / "src"
        / "16_deep_learning"
        / "model_service.py"
    )

    module_name = "healthai_api_xray_model_service"

    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(
            module_name,
            model_service_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load model service: {model_service_path}"
            )

        module = importlib.util.module_from_spec(spec)

        sys.modules[module_name] = module

        spec.loader.exec_module(module)

    return module.HealthAIModelService()


@lru_cache(maxsize=1)
def agent_executor():
    """
    Load Agentic AI executor once.
    """

    return load_adapter(
        "03_agent_executor.py"
    )


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def add_latency(result, start_time):
    """
    Add API latency to dictionary responses.
    """

    if isinstance(result, dict):
        result["api_latency_ms"] = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

    return result


def bmi_category(bmi: float) -> str:

    if bmi < 18.5:
        return "Underweight"

    if bmi < 25:
        return "Normal"

    if bmi < 30:
        return "Overweight"

    return "Obese"


def age_group(age: float) -> str:

    if age < 18:
        return "Child"

    if age < 35:
        return "Young Adult"

    if age < 55:
        return "Middle Age"

    return "Senior"


def high_glucose(glucose: float) -> int:
    return 1 if glucose >= 126 else 0


def high_cholesterol(cholesterol: float) -> int:
    return 1 if cholesterol >= 200 else 0


def hba1c_category(hba1c: float) -> str:

    if hba1c < 5.7:
        return "Normal"

    if hba1c < 6.5:
        return "Prediabetes"

    return "High"


# ---------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------

class DiabetesRequest(BaseModel):

    Age: float = Field(
        ...,
        ge=0,
        le=120,
    )

    Gender: str

    Region: str

    Socioeconomic_Status: str

    Symptoms: str = ""

    Blood_Glucose_mg_dL: float = Field(
        ...,
        ge=0,
    )

    HbA1c_: float = Field(
        ...,
        alias="HbA1c_%",
        ge=0,
    )

    Total_Cholesterol_mg_dL: float = Field(
        ...,
        ge=0,
    )

    BMI: float = Field(
        ...,
        ge=0,
    )

    model_config = ConfigDict(
        populate_by_name=True
    )


class DiagnosisRequest(DiabetesRequest):
    pass


class LOSRequest(BaseModel):

    rcount: float = Field(
        ...,
        ge=0,
    )

    gender: str

    dialysisrenalendstage: int = 0
    asthma: int = 0
    irondef: int = 0
    pneum: int = 0
    substancedependence: int = 0
    psychologicaldisordermajor: int = 0
    depress: int = 0
    psychother: int = 0
    fibrosisandother: int = 0
    malnutrition: int = 0
    hemo: int = 0

    hematocrit: float
    neutrophils: float
    sodium: float
    glucose: float

    bloodureanitro: float = 15.0

    creatinine: float
    bmi: float
    pulse: float
    respiration: float

    secondarydiagnosisnonicd9: float = 0

    # Internal model category.
    # Not necessarily exposed in the patient UI.
    facid: str = "A"


class TextRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
    )


class TranslationRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
    )

    target_language: str = Field(
        default="Hindi",
        min_length=1,
    )


class RAGRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
    )


class AgentChatRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
    )


class DeteriorationRequest(BaseModel):

    patient_sequence: list[list[float]] = Field(
        ...,
        description=(
            "24-hour patient sequence with "
            "28 features per timestep."
        ),
    )


# ---------------------------------------------------------------------
# Feature engineering for structured patient requests
# ---------------------------------------------------------------------

def build_patient_features(request):

    age = float(request.Age)

    glucose = float(
        request.Blood_Glucose_mg_dL
    )

    hba1c = float(
        request.HbA1c_
    )

    cholesterol = float(
        request.Total_Cholesterol_mg_dL
    )

    bmi = float(
        request.BMI
    )

    return {

        "Age": age,

        "Gender": str(
            request.Gender
        ),

        "Region": str(
            request.Region
        ),

        "Socioeconomic_Status": str(
            request.Socioeconomic_Status
        ),

        "Symptoms": str(
            request.Symptoms
        ),

        "Blood_Glucose_mg_dL": glucose,

        "HbA1c_%": hba1c,

        "Total_Cholesterol_mg_dL": cholesterol,

        "BMI": bmi,

        "BMI_Category": bmi_category(
            bmi
        ),

        "Age_Group": age_group(
            age
        ),

        "High_Glucose": high_glucose(
            glucose
        ),

        "High_Cholesterol": high_cholesterol(
            cholesterol
        ),

        "HbA1c_Category": hba1c_category(
            hba1c
        ),
    }


# ---------------------------------------------------------------------
# Recursive response extraction
# ---------------------------------------------------------------------

def find_value(obj, keys):
    """
    Safely find a value inside nested dictionaries/lists.

    Useful because adapters can evolve their response structure
    without breaking the API/UI contract.
    """

    if isinstance(obj, dict):

        # First check direct keys.
        for key in keys:

            value = obj.get(key)

            if value not in (None, ""):
                return value

        # Then recursively search nested values.
        for value in obj.values():

            found = find_value(
                value,
                keys,
            )

            if found not in (
                None,
                "",
            ):
                return found

    elif isinstance(obj, (list, tuple)):

        for value in obj:

            found = find_value(
                value,
                keys,
            )

            if found not in (
                None,
                "",
            ):
                return found

    return None


# ---------------------------------------------------------------------
# X-ray response normalization
# ---------------------------------------------------------------------

def normalize_xray_result(result):
    """
    Convert the adapter result into one stable API contract.

    Expected final response:

    {
        "status": "success",
        "tool": "xray_analysis",
        "prediction": "PNEUMONIA",
        "pneumonia_probability": 0.65,
        "pneumonia_probability_pct": 65.0,
        "message": "Pneumonia detected."
    }
    """

    label = find_value(
        result,
        [
            "prediction",
            "predicted_class",
            "predicted_label",
            "class_name",
            "class",
            "label",
            "xray_prediction",
            "diagnosis",
        ],
    )

    probability = find_value(
        result,
        [
            "pneumonia_probability",
            "pneumonia_prob",
            "positive_probability",
            "probability",
            "confidence",
            "pneumonia_score",
            "score",
        ],
    )

    # ---------------------------------------------------------------
    # Normalize probability
    # ---------------------------------------------------------------

    if probability is not None:

        try:

            probability = float(
                probability
            )

            # Convert percentage to 0-1 if necessary.
            if probability > 1:
                probability /= 100.0

            probability = max(
                0.0,
                min(
                    1.0,
                    probability,
                ),
            )

        except (
            TypeError,
            ValueError,
        ):

            probability = None

    # ---------------------------------------------------------------
    # Derive label from probability if required
    # ---------------------------------------------------------------

    if (
        label is None
        and probability is not None
    ):

        label = (
            "PNEUMONIA"
            if probability >= 0.50
            else "NORMAL"
        )

    # ---------------------------------------------------------------
    # Normalize label
    # ---------------------------------------------------------------

    normalized = (
        str(label).strip().upper()
        if label is not None
        else None
    )

    if normalized in {
        "PNEUMONIA",
        "PNEUMONIA DETECTED",
        "POSITIVE",
        "1",
    }:

        normalized = "PNEUMONIA"

    elif normalized in {
        "NORMAL",
        "NO PNEUMONIA",
        "NEGATIVE",
        "0",
    }:

        normalized = "NORMAL"

    # ---------------------------------------------------------------
    # Final patient-facing message
    # ---------------------------------------------------------------

    if normalized == "PNEUMONIA":

        message = (
            "Pneumonia detected."
        )

    elif normalized == "NORMAL":

        message = (
            "No pneumonia detected."
        )

    else:

        message = (
            "CNN X-ray analysis completed."
        )

    return {

        "status": (
            result.get(
                "status",
                "success",
            )
            if isinstance(
                result,
                dict,
            )
            else "success"
        ),

        "tool": "xray_analysis",

        "prediction": normalized,

        "pneumonia_probability": (
            probability
        ),

        "pneumonia_probability_pct": (
            round(
                probability * 100,
                2,
            )
            if probability is not None
            else None
        ),

        "message": message,

        "raw_result": result,
    }


# =====================================================================
# BASIC ENDPOINTS
# =====================================================================

@app.get("/")
def root():

    return {

        "application":
            "HealthAI Intelligent Platform",

        "status":
            "running",

        "version":
            "1.0.0",

        "docs":
            "/docs",
    }


@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "service":
            "HealthAI Intelligent Platform",

        "version":
            "1.0.0",
    }


@app.get("/info")
def info():

    return {

        "project":
            "HealthAI-Intelligent-Platform",

        "version":
            "1.0.0",

        "capabilities": [

            "diabetes_prediction",

            "hospital_los_prediction",

            "primary_diagnosis",

            "medical_ner",

            "sentiment_analysis",

            "medical_rag",

            "xray_analysis",

            "agentic_ai",
        ],

        "documentation":
            "/docs",
    }


# =====================================================================
# DIABETES
# =====================================================================

@app.post("/predict/diabetes")
def predict_diabetes(
    request: DiabetesRequest,
):

    start = time.perf_counter()

    try:

        features = build_patient_features(
            request
        )

        result = (
            diabetes_adapter()
            .predict_diabetes_risk(
                features
            )
        )

        return add_latency(
            result,
            start,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "status":
                    "error",

                "tool":
                    "diabetes_prediction",

                "message":
                    str(exc),
            },
        )


# =====================================================================
# HOSPITAL LENGTH OF STAY
# =====================================================================

@app.post("/predict/los")
def predict_los(
    request: LOSRequest,
):

    start = time.perf_counter()

    try:

        result = (
            los_adapter()
            .execute_hospital_los_prediction(
                patient_data=request.model_dump()
            )
        )

        return add_latency(
            result,
            start,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "status":
                    "error",

                "tool":
                    "hospital_los_prediction",

                "message":
                    str(exc),
            },
        )


# =====================================================================
# PRIMARY DIAGNOSIS
# =====================================================================

@app.post("/predict/diagnosis")
def predict_diagnosis(
    request: DiagnosisRequest,
):

    start = time.perf_counter()

    try:

        features = build_patient_features(
            request
        )

        result = (
            diagnosis_adapter()
            .predict_primary_diagnosis(
                features
            )
        )

        if isinstance(
            result,
            dict,
        ):

            result["tool"] = (
                "primary_diagnosis"
            )

        return add_latency(
            result,
            start,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "status":
                    "error",

                "tool":
                    "primary_diagnosis",

                "message":
                    str(exc),
            },
        )


# =====================================================================
# MEDICAL NER
# =====================================================================

@app.post("/predict/ner")
def predict_ner(
    request: TextRequest,
):

    start = time.perf_counter()

    try:

        result = (
            ner_adapter()
            .execute_medical_ner(
                request.text
            )
        )

        return add_latency(
            result,
            start,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "status":
                    "error",

                "tool":
                    "medical_ner",

                "message":
                    str(exc),
            },
        )


# =====================================================================
# SENTIMENT
# =====================================================================

@app.post("/predict/sentiment")
def predict_sentiment(
    request: TextRequest,
):

    start = time.perf_counter()

    try:

        module = sentiment_adapter()

        # Current adapter interface.
        if hasattr(
            module,
            "execute_sentiment_analysis",
        ):

            result = (
                module
                .execute_sentiment_analysis(
                    query=request.text
                )
            )

        elif hasattr(
            module,
            "predict_sentiment",
        ):

            result = (
                module
                .predict_sentiment(
                    request.text
                )
            )

        else:

            raise AttributeError(
                "No supported sentiment "
                "prediction function found."
            )

        if not isinstance(
            result,
            dict,
        ):

            result = {
                "prediction": result
            }

        prediction = result.get(
            "prediction"
        )

        sentiment = result.get(
            "sentiment"
        )

        # Derive sentiment from binary prediction.
        if (
            sentiment is None
            and prediction is not None
        ):

            try:

                sentiment = (
                    "Positive"
                    if int(prediction) == 1
                    else "Negative"
                )

            except (
                TypeError,
                ValueError,
            ):

                sentiment = str(
                    prediction
                )

        result["status"] = (
            "success"
        )

        result["tool"] = (
            "sentiment_analysis"
        )

        result["sentiment"] = (
            str(
                sentiment
            ).capitalize()
            if sentiment is not None
            else "Unknown"
        )

        if prediction is not None:

            try:

                result["prediction"] = (
                    int(prediction)
                )

            except (
                TypeError,
                ValueError,
            ):

                pass

        return add_latency(
            result,
            start,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "status":
                    "error",

                "tool":
                    "sentiment_analysis",

                "message":
                    str(exc),
            },
        )


# =====================================================================
# MEDICAL RAG
# =====================================================================

@app.post("/predict/rag")
def predict_rag(
    request: RAGRequest,
):

    start = time.perf_counter()

    try:

        result = (
            rag_adapter()
            .execute_medical_rag(
                query=request.query
            )
        )

        return add_latency(
            result,
            start,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "status":
                    "error",

                "tool":
                    "medical_rag",

                "message":
                    str(exc),
            },
        )


# =====================================================================
# PATIENT CLUSTERING
# =====================================================================

@app.post("/predict/translation")
def predict_translation(request: TranslationRequest):
    start = time.perf_counter()

    try:
        worker_path = (
            PROJECT_ROOT
            / "src"
            / "17_agentic_ai"
            / "20_translation_worker.py"
        )

        command = [
            r"C:\Users\radha\anaconda3\python.exe",
            str(worker_path),
            request.text,
            request.target_language,
        ]

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )

        if process.returncode != 0:
            raise RuntimeError(
                process.stderr.strip()
                or "Translation worker failed."
            )

        lines = [
            line.strip()
            for line in process.stdout.splitlines()
            if line.strip().startswith("{")
            and line.strip().endswith("}")
        ]

        if not lines:
            raise RuntimeError(
                "Translation worker returned no JSON result."
            )

        result = json.loads(lines[-1])

        return add_latency(result, start)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "tool": "medical_translation",
                "message": str(exc),
            },
        )


@app.post("/predict/clustering")
def predict_clustering():

    start = time.perf_counter()

    try:

        adapter = clustering_adapter()

        result = adapter.execute_patient_clustering(
            query="Show me the patient clusters."
        )

        return add_latency(
            result,
            start,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "tool": "patient_clustering",
                "message": str(exc),
            },
        )


# =====================================================================
# ASSOCIATION ANALYSIS
# =====================================================================

@app.post("/predict/association")
def predict_association():

    start = time.perf_counter()

    try:

        adapter = association_adapter()

        result = adapter.execute_association_analysis(
            query=(
                "Show me the strongest "
                "clinical association rules."
            )
        )

        return add_latency(
            result,
            start,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "tool": "association_analysis",
                "message": str(exc),
            },
        )


# =====================================================================
# CLINICAL DETERIORATION - RNN
# =====================================================================

@app.post("/predict/deterioration")
def predict_deterioration(
    request: DeteriorationRequest,
):

    start = time.perf_counter()

    try:

        adapter = deterioration_adapter()

        result = adapter.execute_deterioration_prediction(
            query=(
                "Predict deterioration risk "
                "for this patient."
            ),
            patient_sequence=request.patient_sequence,
        )

        return add_latency(
            result,
            start,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "tool": "deterioration_prediction",
                "message": str(exc),
            },
        )


# =====================================================================
# AGENTIC AI CHAT
# =====================================================================

@app.post("/agent/chat")
def agent_chat(
    request: AgentChatRequest,
):

    start = time.perf_counter()

    try:

        executor = agent_executor()

        # Preferred current interface.
        if hasattr(
            executor,
            "run_agent",
        ):

            result = (
                executor.run_agent(
                    query=request.query
                )
            )

        # Backward-compatible interface.
        elif hasattr(
            executor,
            "execute_agent_query",
        ):

            result = (
                executor.execute_agent_query(
                    query=request.query
                )
            )

        else:

            raise AttributeError(
                "Agent executor does not "
                "contain a supported execution function."
            )

        return add_latency(
            result,
            start,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "status":
                    "error",

                "tool":
                    "agentic_ai",

                "message":
                    str(exc),
            },
        )


# =====================================================================
# CHEST X-RAY CNN
# =====================================================================

@app.post("/predict/xray")
async def predict_xray(
    file: UploadFile = File(...),
):
    """
    Analyze an uploaded chest X-ray.

    Pipeline:

        uploaded image
              â†“
        PIL validation
              â†“
        RGB conversion
              â†“
        resize 128 x 128
              â†“
        normalize 0-1
              â†“
        add batch dimension
              â†“
        X-ray adapter
              â†“
        CNN model
              â†“
        normalized API response

    IMPORTANT:
    The existing HealthAIModelService.predict_xray()
    expects an image array, so this endpoint passes
    image_array to the adapter.
    """

    start = time.perf_counter()

    try:

        # -------------------------------------------------------------
        # Read uploaded file
        # -------------------------------------------------------------

        data = await file.read()

        if not data:

            raise HTTPException(
                status_code=400,
                detail="Empty X-ray file.",
            )

        # -------------------------------------------------------------
        # Validate and preprocess image
        # -------------------------------------------------------------

        from io import BytesIO

        import numpy as np

        from PIL import Image

        try:

            image = Image.open(
                BytesIO(data)
            )

            # CNN expects 3-channel RGB.
            image = image.convert(
                "RGB"
            )

        except Exception:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid image file. "
                    "Please upload a valid "
                    "chest X-ray image."
                ),
            )

        # Existing CNN input size.
        image = image.resize(
            (128, 128)
        )

        # Convert to float32 and normalize.
        image_array = (
            np.asarray(
                image,
                dtype=np.float32,
            )
            / 255.0
        )

        # Add batch dimension:
        #
        # (128, 128, 3)
        #
        # becomes:
        #
        # (1, 128, 128, 3)

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        # -------------------------------------------------------------
        # Execute existing X-ray adapter
        # -------------------------------------------------------------

        model_service = xray_adapter()

        result = model_service.predict_xray(
            image_array
        )

        # -------------------------------------------------------------
        # Normalize result
        # -------------------------------------------------------------

        response = normalize_xray_result(
            result
        )

        return add_latency(
            response,
            start,
        )

    except HTTPException:
        raise

    except Exception as exc:

        # Print traceback in terminal so
        # debugging is easy during development.
        print(
            "\n========== X-RAY API ERROR =========="
        )

        print(
            str(exc)
        )

        import traceback

        traceback.print_exc()

        print(
            "=====================================\n"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "status":
                    "error",

                "tool":
                    "xray_analysis",

                "message":
                    str(exc),
            },
        )


# =====================================================================
# APPLICATION START
# =====================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.18_api.01_fastapi_app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )

