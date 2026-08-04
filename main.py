import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# Geheugen-opslag voor de nieuwste binnenkomende stroomdata van de hardware-groep
latest_data = {}

# 1. Probeer het AI-model bij het opstarten in te laden
def load_model():
    try:
        model = joblib.load("appliance_classifier.pkl")
        print("✅ AI Model succesvol geladen!")
        return model
    except Exception as e:
        print(f"⚠️ Kon model niet laden: {e}")
        return None

model = load_model()

# Datastructuren voor Pydantic
class PredictRequest(BaseModel):
    fft_features: list[float]

class LearnRequest(BaseModel):
    label: str
    fft_features: list[float]


# --- BASIC & HARDWARE ENDPOINTS ---

@app.get("/")
def home():
    return {"message": "PowerSense AI Server draait succesvol!"}

@app.post("/data")
async def receive_data(data: dict):
    """Ontvangt live transient- metingen van de stroomdetectie groep/ESP32."""
    global latest_data
    latest_data = data
    return {"status": "success", "message": "Data ontvangen"}

@app.get("/latest")
async def get_latest():
    """Geeft de meest recente stroomdata terug aan het dashboard."""
    if not latest_data:
        return {"status": "empty", "raw_transient": []}
    return latest_data


# --- AI PREDICTIE ENDPOINT (VOOR OLED EN DASHBOARD) ---

@app.post("/predict")
async def predict_appliance(data: PredictRequest):
    """
    Herken een apparaat op basis van de FFT kenmerken.
    Geeft bijvoorbeeld "VENTILATOR" of "STANDBY" terug voor het OLED-scherm.
    """
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model is nog niet getraind of geladen op de server")
    
    try:
        prediction = model.predict([data.fft_features])[0]
        return {
            "status": "success",
            "appliance": str(prediction).upper()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fout bij voorspelling: {str(e)}")


# --- AI LEARN / TRAINING ENDPOINT ---

@app.post("/learn")
async def learn_pattern(data: LearnRequest):
    """
    Wanneer op de Learn-knop op het dashboard of socket wordt gedrukt:
    1. Slaat het nieuwe FFT-patroon op in dataset.json met het meegegeven label.
    2. Her-traint direct de RandomForestClassifier.
    3. Werkt het actieve AI-model in het geheugen bij.
    """
    global model

    # 1. Voeg het nieuwe patroon toe aan dataset.json
    new_entry = {
        "label": data.label,
        "fft_features": data.fft_features
    }
    
    try:
        with open("dataset.json", "a") as f:
            f.write(json.dumps(new_entry) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kon data niet opslaan in dataset: {e}")

    # 2. Her-train het model direct met alle verzamelde data
    X, y = [], []
    try:
        with open("dataset.json", "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    X.append(item["fft_features"])
                    y.append(item["label"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fout bij uitlezen dataset: {e}")

    unique_labels = set(y)
    if len(unique_labels) < 2:
        return {
            "status": "saved",
            "message": f"Patroon opgeslagen voor '{data.label}'. Er zijn minstens 2 verschillende labels nodig om te trainen."
        }

    # 3. Train het model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(np.array(X), np.array(y))
    
    # Sla op naar bestand en update in geheugen
    joblib.dump(clf, "appliance_classifier.pkl")
    model = clf

    return {
        "status": "success",
        "message": f"Patroon voor '{data.label}' succesvol geleerd en AI-model opnieuw getraind!",
        "active_labels": list(unique_labels)
    }