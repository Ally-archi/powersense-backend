import json
import numpy as np
import requests

# Set the appliance name to fan
APPARAAT_NAAM = "standby"

SERVER_URL = "https://powersense-api-1x24u.onrender.com/latest"

print(f"Fetching data from server for '{APPARAAT_NAAM}'...")

try:
    response = requests.get(SERVER_URL, timeout=15)

    if response.status_code != 200:
        print(
            f"Server returned status code {response.status_code}: {response.text}"
        )
        exit()

    data = response.json()
    raw_samples = data.get("raw_transient")

    if not raw_samples:
        print("No signal data found on the server yet.")
        exit()

    # Calculate FFT spectrum
    signal = np.array(raw_samples)
    fft_spectrum = np.abs(np.fft.rfft(signal))

    # Normalize features between 0 and 1
    if np.max(fft_spectrum) > 0:
        fft_normalized = (fft_spectrum / np.max(fft_spectrum)).tolist()
    else:
        fft_normalized = fft_spectrum.tolist()

    # Save entry to dataset.json
    meting = {"label": APPARAAT_NAAM, "fft_features": fft_normalized}

    with open("dataset.json", "a") as f:
        f.write(json.dumps(meting) + "\n")

    print(
        f"Success! FFT pattern for '{APPARAAT_NAAM}' saved to dataset.json."
    )

except Exception as e:
    print(f"Error connecting to server: {e}")