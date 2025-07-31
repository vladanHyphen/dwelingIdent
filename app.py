import streamlit as st
import mercantile
import requests
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import io
from pyproj import Transformer
from inference_sdk import InferenceHTTPClient

# --- ROBOFLOW SETTINGS: PUT YOUR KEY HERE ---
ROBOFLOW_API_KEY = "6bkg1HjCQc6QPtEDWj1p"  # <--- PUT YOUR API KEY HERE
MODEL_ID = "vladan-dltgb/roof-mws17/1"

st.set_page_config(page_title="Automated Roof Detection from Satellite Map", layout="wide")

st.title("Satellite Roof Detector — Roboflow Inference SDK")
st.markdown("""
1. Enter the bounding box for your area of interest (in WGS84 lat/lon).
2. Download a high-resolution satellite map patch (Esri World Imagery).
3. Detect buildings using your Roboflow model.
4. Download the resulting Excel with real-world coordinates.
""")

# ---- BOUNDING BOX INPUTS ----
col1, col2 = st.columns(2)
min_lon = col1.number_input("Min Longitude (left)", value=28.253, format="%.6f")
max_lon = col2.number_input("Max Longitude (right)", value=28.257, format="%.6f")
min_lat = col1.number_input("Min Latitude (bottom)", value=-25.718, format="%.6f")
max_lat = col2.number_input("Max Latitude (top)", value=-25.715, format="%.6f")
zoom = st.slider("Zoom Level (higher = sharper, smaller area, default 18)", 15, 20, 18)

if st.button("Download Map and Detect Buildings"):
    if min_lat >= max_lat or min_lon >= max_lon:
        st.error("Error: Min Latitude/Longitude must be less than Max Latitude/Longitude.")
        st.stop()

    try:
        st.info("Downloading map tiles from Esri World Imagery...")
        tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zoom))
        if len(tiles) == 0:
            st.error("No tiles to download for this bounding box and zoom level. Try a larger area or different coordinates.")
            st.stop()

        tile_url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        tile_size = 256
        cols = max([t.x for t in tiles]) - min([t.x for t in tiles]) + 1
        rows = max([t.y for t in tiles]) - min([t.y for t in tiles]) + 1
        MAX_PIXELS = 60_000_000
        mosaic_width = cols * tile_size
        mosaic_height = rows * tile_size
        total_pixels = mosaic_width * mosaic_height
        if total_pixels > MAX_PIXELS:
            st.error(f"The selected area at this zoom is too large ({total_pixels:,} pixels). Please choose a smaller area or lower zoom.")
            st.stop()

        mosaic = Image.new('RGB', (mosaic_width, mosaic_height))
        min_x, min_y = min([t.x for t in tiles]), min([t.y for t in tiles])
        failed_tiles = 0
        for t in tiles:
            url = tile_url.format(z=zoom, x=t.x, y=t.y)
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content))
            except Exception as e:
                failed_tiles += 1
                st.warning(f"Could not download tile x={t.x}, y={t.y} (zoom {zoom}): {e}")
                img = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
            x_offset = (t.x - min_x) * tile_size
            y_offset = (t.y - min_y) * tile_size
            mosaic.paste(img, (x_offset, y_offset))
        if failed_tiles > 0:
            st.warning(f"{failed_tiles} tiles could not be loaded and were filled with black.")

        # Optional: Crop mosaic to precise AOI
        try:
            bbox_ul = mercantile.xy_bounds(mercantile.Tile(min_x, min_y, zoom))
            bbox_lr = mercantile.xy_bounds(mercantile.Tile(min_x+cols-1, min_y+rows-1, zoom))
            def lonlat_to_pixel(lon, lat, mosaic_w, mosaic_h):
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
                x, y = transformer.transform(lon, lat)
                x0, y0 = bbox_ul.left, bbox_lr.top
                x1, y1 = bbox_lr.right, bbox_ul.bottom
                px = int((x - x0) / (x1 - x0) * mosaic_w)
                py = int((y - y0) / (y1 - y0) * mosaic_h)
                return px, py
            px0, py0 = lonlat_to_pixel(min_lon, max_lat, mosaic.width, mosaic.height)
            px1, py1 = lonlat_to_pixel(max_lon, min_lat, mosaic.width, mosaic.height)
            px_left, px_right = min(px0, px1), max(px0, px1)
            py_top, py_bottom = min(py0, py1), max(py0, py1)
            px_left = max(0, px_left)
            py_top = max(0, py_top)
            px_right = min(mosaic.width, px_right)
            py_bottom = min(mosaic.height, py_bottom)
            if px_right - px_left > 0 and py_bottom - py_top > 0:
                mosaic = mosaic.crop((px_left, py_top, px_right, py_bottom))
            else:
                st.warning("Invalid crop area calculated. Using full mosaic.")
        except Exception as e:
            st.warning(f"Optional cropping skipped (pyproj error): {e}. Map covers the bounding box.")

        st.image(mosaic, caption="Downloaded Map", use_container_width=True)
        st.success("Map downloaded. Sending to Roboflow for building detection...")

        # ---- ROBOFLOW DETECTION (Inference SDK, in-memory PIL image) ----
        CLIENT = InferenceHTTPClient(
            api_url="https://infer.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )

        result = CLIENT.infer(mosaic, model_id=MODEL_ID)
        predictions = result.get("predictions", [])

        # Draw detections on the PIL image
        draw = ImageDraw.Draw(mosaic)
        for pred in predictions:
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            box = [x - w // 2, y - h // 2, x + w // 2, y + h // 2]
            draw.rectangle(box, outline="green", width=3)
            draw.text((box[0], box[1] - 10), pred["class"], fill="green")

        st.image(mosaic, caption="Buildings detected by Roboflow", use_container_width=True)

        # ---- Convert detections to coordinates ----
        transformer_back = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        x0, y0 = bbox_ul.left, bbox_lr.top
        x1, y1 = bbox_lr.right, bbox_ul.bottom
        w, h = mosaic.width, mosaic.height

        def pixel_to_lonlat(px, py):
            x = x0 + (x1 - x0) * (px / w)
            y = y0 + (y1 - y0) * (py / h)
            lon, lat = transformer_back.transform(x, y)
            return lon, lat

        df = pd.DataFrame([
            {
                'Building_ID': idx+1,
                'X_pixel': int(pred["x"]),
                'Y_pixel': int(pred["y"]),
                'Longitude': pixel_to_lonlat(pred["x"], pred["y"])[0],
                'Latitude': pixel_to_lonlat(pred["x"], pred["y"])[1],
                'Width': pred["width"],
                'Height': pred["height"],
                'Class': pred["class"],
                'Confidence': pred["confidence"]
            }
            for idx, pred in enumerate(predictions)
        ])
        if len(df) > 0:
            st.dataframe(df.head(10))
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)
            st.download_button(
                label="Download Excel with Building Coordinates (Lon/Lat)",
                data=output,
                file_name="building_centroids_lonlat.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("Done! Download your Excel above.")
        else:
            st.warning("No buildings detected by Roboflow in this area.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        import traceback
        st.text(traceback.format_exc())
