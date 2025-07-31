import requests
import base64

ROBOFLOW_API_KEY = "rf_pO8HKBnumvT4TqSlaibulaIEuit2"
ROBOFLOW_MODEL_URL = f"https://infer.roboflow.com/spacenet/buildings-2pbzy/5?api_key={ROBOFLOW_API_KEY}"

with open("some_small_satellite_image.jpg", "rb") as img_file:
    b64img = base64.b64encode(img_file.read()).decode("utf-8")

headers = {"Content-Type": "application/json"}
payload = {"image": b64img}

response = requests.post(ROBOFLOW_MODEL_URL, headers=headers, json=payload)
print(response.status_code)
print(response.text)


