import streamlit as st
import mercantile
import requests
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import io

st.set_page_config(page_title="Automated Roof Detection from Satellite Map", layout="wide")

st.title("Satellite Roof Detector — Enter Coordinates, Download Map, Detect, Export")

st.markdown("""
1. Enter the bounding box for your area of interest (in WGS84 lat/lon).
2. The app will download a high-resolution satellite map patch (Esri World Imagery).
3. Roof detection is performed.
4. Download the resulting Excel with real-world coordinates.
""")

# ---- BOUNDING BOX INPUTS ----
col1, col2 = st.columns(2)
min_lon = col1.number_input("Min Longitude (left)", value=28.253, format="%.6f")
max_lon = col2.number_input("Max Longitude (right)", value=28.257, format="%.6f")
min_lat = col1.number_input("Min Latitude (bottom)", value=-25.718, format="%.6f")
max_lat = col2.number_input("Max Latitude (top)", value=-25.715, format="%.6f")
zoom = st.slider("Zoom Level (higher = sharper, smaller area, default 18)", 15, 20, 18)

if st.button("Download Map and Detect Roofs"):
    st.info("Downloading map tiles from Esri World Imagery...")

    # Calculate tile list
    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zoom))
    tile_url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    tile_size = 256
    cols = max([t.x for t in tiles]) - min([t.x for t in tiles]) + 1
    rows = max([t.y for t in tiles]) - min([t.y for t in tiles]) + 1

    mosaic = Image.new('RGB', (cols * tile_size, rows * tile_size))
    min_x, min_y = min([t.x for t in tiles]), min([t.y for t in tiles])

    for t in tiles:
        url = tile_url.format(z=zoom, x=t.x, y=t.y)
        resp = requests.get(url)
        img = Image.open(io.BytesIO(resp.content))
        x_offset = (t.x - min_x) * tile_size
        y_offset = (t.y - min_y) * tile_size
        mosaic.paste(img, (x_offset, y_offset))

    # Optionally, crop to precise AOI
    bbox_ul = mercantile.xy_bounds(mercantile.Tile(min_x, min_y, zoom))
    bbox_lr = mercantile.xy_bounds(mercantile.Tile(min_x+cols-1, min_y+rows-1, zoom))

    def lonlat_to_pixel(lon, lat, mosaic_w, mosaic_h):
        # Convert lat/lon to pixel in the stitched image (WebMercator projection)
        from pyproj import Proj, transform
        in_proj = Proj(init='epsg:4326')
        out_proj = Proj(init='epsg:3857')
        x, y = transform(in_proj, out_proj, lon, lat)
        # x/y bounds for entire mosaic
        x0, y0 = bbox_ul.left, bbox_lr.top
        x1, y1 = bbox_lr.right, bbox_ul.bottom
        px = int((x - x0) / (x1 - x0) * mosaic_w)
        py = int((y - y0) / (y1 - y0) * mosaic_h)
        return px, py

    try:
        from pyproj import Proj, transform
        px0, py0 = lonlat_to_pixel(min_lon, max_lat, mosaic.width, mosaic.height)
        px1, py1 = lonlat_to_pixel(max_lon, min_lat, mosaic.width, mosaic.height)
        px_left, px_right = min(px0, px1), max(px0, px1)
        py_top, py_bottom = min(py0, py1), max(py0, py1)
        mosaic = mosaic.crop((px_left, py_top, px_right, py_bottom))
    except Exception as e:
        st.warning("Optional cropping skipped (pyproj missing or cropping error). Map covers the bounding box.")

    st.image(mosaic, caption="Downloaded Map", use_column_width=True)
    st.success("Map downloaded. Detecting roofs...")

    # ---- Roof detection (as before) ----
    img_rgb = np.array(mosaic)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    roof_mask = ((s < 65) & (v > 100)).astype(np.uint8) * 255
    kernel = np.ones((2, 2), np.uint8)
    roof_mask_clean = cv2.morphologyEx(roof_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(roof_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_result = img_rgb.copy()
    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))
                cv2.circle(overlay_result, (cx, cy), 8, (0,255,0), -1)  # green dot

    num_dwellings = len(centroids)
    st.success(f"Detected {num_dwellings} dwellings (green dots)")

    st.image(overlay_result, caption="Detected dwellings (green dots)", use_column_width=True)

    # ---- Calculate coordinates for each centroid ----
    # Inverse pixel to lon/lat using mercantile.xy_bounds
    from pyproj import Proj, transform
    in_proj = Proj(init='epsg:3857')
    out_proj = Proj(init='epsg:4326')
    x0, y0 = bbox_ul.left, bbox_lr.top
    x1, y1 = bbox_lr.right, bbox_ul.bottom
    w, h = mosaic.width, mosaic.height

    def pixel_to_lonlat(px, py):
        x = x0 + (x1 - x0) * (px / w)
        y = y0 + (y1 - y0) * (py / h)
        lon, lat = transform(in_proj, out_proj, x, y)
        return lon, lat

    df = pd.DataFrame([
        {
            'Dwelling_ID': idx+1,
            'X_pixel': cx,
            'Y_pixel': cy,
            'Longitude': pixel_to_lonlat(cx, cy)[0],
            'Latitude': pixel_to_lonlat(cx, cy)[1],
            'Address': f'Dwelling_{idx+1}'
        }
        for idx, (cx, cy) in enumerate(centroids)
    ])

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)

    st.download_button(
        label="Download Excel with Dwelling Coordinates (Lon/Lat)",
        data=output,
        file_name="roof_centroids_lonlat.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ) 

    st.success("Done! Download your Excel above.")

