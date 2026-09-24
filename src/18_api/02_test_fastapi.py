import requests
import json

BASE_URL = "http://127.0.0.1:8000"


def test_endpoint(name, method, endpoint, payload=None):
    print("\n" + "=" * 70)
    print(f"TEST: {name}")
    print("=" * 70)

    try:
        if method == "GET":
            response = requests.get(
                f"{BASE_URL}{endpoint}",
                timeout=60
            )
        else:
            response = requests.post(
                f"{BASE_URL}{endpoint}",
                json=payload,
                timeout=120
            )

        print("Status code:", response.status_code)

        try:
            result = response.json()
            print(json.dumps(result, indent=2))
        except Exception:
            print(response.text)

        if response.status_code == 200:
            print("RESULT: PASS")
            return True
        else:
            print("RESULT: FAIL")
            return False

    except Exception as e:
        print("ERROR:", str(e))
        print("RESULT: FAIL")
        return False


def main():

    print("\n" + "#" * 70)
    print("HEALTHAI - FASTAPI END-TO-END API TEST")
    print("#" * 70)

    results = []

    # ---------------------------------------------------------
    # 1. HEALTH
    # ---------------------------------------------------------
    results.append(
        test_endpoint(
            "Health Check",
            "GET",
            "/health"
        )
    )

    # ---------------------------------------------------------
    # 2. INFO
    # ---------------------------------------------------------
    results.append(
        test_endpoint(
            "API Information",
            "GET",
            "/info"
        )
    )

    # ---------------------------------------------------------
    # 3. DIABETES PREDICTION
    # ---------------------------------------------------------
    diabetes_payload = {
        "Age": 45,
        "Gender": "Female",
        "Region": "South",
        "Socioeconomic_Status": "Middle",
        "Symptoms": "increased thirst, frequent urination",
        "Blood_Glucose_mg_dL": 180,
        "HbA1c_%": 7.2,
        "Total_Cholesterol_mg_dL": 210,
        "BMI": 28.5
    }

    results.append(
        test_endpoint(
            "Diabetes Prediction",
            "POST",
            "/predict/diabetes",
            diabetes_payload
        )
    )

    # ---------------------------------------------------------
    # 4. HOSPITAL LOS
    # ---------------------------------------------------------
    los_payload = {
        "rcount": 1,
        "gender": "F",
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
        "hematocrit": 38.0,
        "neutrophils": 65.0,
        "sodium": 138.0,
        "glucose": 110.0,
        "bloodureanitro": 15.0,
        "creatinine": 1.0,
        "bmi": 24.5,
        "pulse": 78.0,
        "respiration": 18.0,
        "secondarydiagnosisnonicd9": 2,
        "facid": "F01"
    }

    results.append(
        test_endpoint(
            "Hospital Length of Stay",
            "POST",
            "/predict/los",
            los_payload
        )
    )

    # ---------------------------------------------------------
    # 5. PRIMARY DIAGNOSIS
    # ---------------------------------------------------------
    diagnosis_payload = {
        "Age": 45,
        "Gender": "Female",
        "Region": "South",
        "Socioeconomic_Status": "Middle",
        "Occupation": "Office",
        "Symptoms": "increased thirst, frequent urination",
        "Blood_Glucose_mg_dL": 180,
        "HbA1c_%": 7.2,
        "Total_Cholesterol_mg_dL": 210,
        "BMI": 28.5
    }

    results.append(
        test_endpoint(
            "Primary Diagnosis",
            "POST",
            "/predict/diagnosis",
            diagnosis_payload
        )
    )

    # ---------------------------------------------------------
    # 6. MEDICAL NER
    # ---------------------------------------------------------
    ner_payload = {
        "text": "Patient has blood glucose 180 mg/dL and takes metformin 500mg bd."
    }

    results.append(
        test_endpoint(
            "Medical NER",
            "POST",
            "/predict/ner",
            ner_payload
        )
    )

    # ---------------------------------------------------------
    # 7. SENTIMENT
    # ---------------------------------------------------------
    sentiment_payload = {
        "text": "The staff were very helpful and explained everything clearly."
    }

    results.append(
        test_endpoint(
            "Medical Sentiment",
            "POST",
            "/predict/sentiment",
            sentiment_payload
        )
    )

    # ---------------------------------------------------------
    # 8. RAG
    # ---------------------------------------------------------
    rag_payload = {
        "query": "What are the symptoms of diabetes?"
    }

    results.append(
        test_endpoint(
            "Medical RAG",
            "POST",
            "/predict/rag",
            rag_payload
        )
    )

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------
    passed = sum(results)
    total = len(results)

    print("\n" + "#" * 70)
    print("FINAL API TEST SUMMARY")
    print("#" * 70)

    print(f"Tests passed : {passed}/{total}")
    print(f"Tests failed : {total - passed}/{total}")
    print(f"Accuracy     : {(passed / total) * 100:.2f}%")

    if passed == total:
        print("\nSTEP 17.17 API TESTS PASSED")
    else:
        print("\nSTEP 17.17 API TESTS NEED ATTENTION")


if __name__ == "__main__":
    main()