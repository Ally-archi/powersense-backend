import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_ai():
    X = [] # De FFT frequentie features
    y = [] # De labels ("ventilator", "standby")

    # 1. Lees de verzamelde data uit dataset.json
    try:
        with open("dataset.json", "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    X.append(item["fft_features"])
                    y.append(item["label"])
    except FileNotFoundError:
        print("❌ Fout: dataset.json is niet gevonden!")
        return

    unique_labels = set(y)
    print(f"📊 Totaal aantal voorbeelden: {len(y)}")
    print(f"🏷️ Gevonden apparaten/status: {list(unique_labels)}")

    if len(unique_labels) < 2:
        print("\n⚠️ Je hebt minstens 2 verschillende labels nodig om te trainen!")
        return

    # 2. Zet om naar Numpy arrays en train het Machine Learning model
    X = np.array(X)
    y = np.array(y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # 3. Sla het getrainde model op
    joblib.dump(clf, "appliance_classifier.pkl")
    print("\n🎉 MODEL SUCCESVOL GETRAIND EN OPGESLAGEN ALS 'appliance_classifier.pkl'!")

if __name__ == "__main__":
    train_ai()