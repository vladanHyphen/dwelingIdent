import streamlit as st
import requests
import base64
from PIL import Image
import io

st.title("Roboflow Universe 405 Test")

# Build a test image in memory (100x100 white)
img = Image.new("RGB", (100, 100), (255, 255, 255))
buf = io.BytesIO()
img.save(buf, format="JPEG")
b64img = base64.b64encode(buf.getvalue()).decode("utf-8")

api_key = "rf_pO8HKBnumvT4TqSlaibulaIEuit2"
model_url = f"https://infer.roboflow.com/spacenet/buildings-2pbzy/5?api_key={api_key}"

headers = {"Content-Type": "application/json"}
payload = {"image": b64img}

if st.button("Test Roboflow API"):
    resp = requests.post(model_url, headers=headers, json=payload)
    st.write("Status code:", resp.status_code)
    st.write("Response:", resp.text)
