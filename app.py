"""
PyGIS Explorer — search and download Sentinel and Landsat imagery.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Keep app.py and image_downloader.py in the same folder.
"""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import zipfile
from datetime import date, timedelta
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import Draw
from shapely.affinity import translate
from shapely.geometry import mapping, shape
from streamlit_folium import st_folium

import image_downloader as sd

LOGO = Path(__file__).with_name("pygis_logo.png")
UPATTA_LOGO = Path(__file__).with_name("upatta_logo.png")
# Half of the size previously shown in the sidebar.
LOGO_WIDTH = 130
st.set_page_config(page_title="PyGIS Explorer", page_icon=str(LOGO), layout="wide")

OUT_ROOT = Path("pygis_runs")
S2_BANDS = ["blue", "green", "red", "nir", "rededge1", "rededge2", "rededge3",
            "nir08", "swir16", "swir22", "scl", "coastal", "visual"]
# Sentinel-1 acquisition choices. Polarization also selects the bands to download.
S1_MODES = {
    "SM": "SM — Stripmap, 3.5 m × 3.5 m",
    "IW": "IW — Interferometric Wide Swath, 10 m × 10 m",
    "EW": "EW — Extra-Wide Swath, 40 m × 40 m",
}
S1_POLARIZATIONS = {
    "VV": ("VV", ["vv"]),
    "VV+VH": ("VV+VH", ["vv", "vh"]),
    "HH": ("HH", ["hh"]),
    "HH+HV": ("HH+HV", ["hh", "hv"]),
}
# Copernicus stores dual-pol as VV&VH / HH&HV.
S1_CDSE_POL = {"VV": "VV", "VV+VH": "VV&VH", "HH": "HH", "HH+HV": "HH&HV"}
LS_BANDS = ["blue", "green", "red", "nir", "swir16", "swir22", "coastal",
            "lwir11", "qa_pixel"]
# Planetary Computer Sentinel-3 variables worth offering (not tie-point files).
S3_VARIABLES = {
    "sentinel-3-olci-lfr-l2-netcdf": ["otci", "gifapar", "iwv", "rc-gifapar", "lqsf"],
    "sentinel-3-olci-wfr-l2-netcdf": [
        "chl-oc4me", "chl-nn", "tsm-nn", "iop-nn", "iwv", "par", "wqsf",
        "oa04-reflectance", "oa06-reflectance", "oa08-reflectance",
    ],
    "sentinel-3-slstr-lst-l2-netcdf": ["lst-in", "slstr-flags-in"],
    "sentinel-3-slstr-wst-l2-netcdf": ["l2p"],
    "sentinel-3-slstr-frp-l2-netcdf": ["frp-in"],
    "sentinel-3-synergy-aod-l2-netcdf": ["ntc-aod"],
    "sentinel-3-synergy-syn-l2-netcdf": [
        "syn-oa04-reflectance", "syn-oa06-reflectance", "syn-oa08-reflectance",
        "syn-oa17-reflectance", "syn-s5n-reflectance", "syn-s6n-reflectance",
        "syn-ato550",
    ],
    "sentinel-3-synergy-vg1-l2-netcdf": ["ndvi", "b0", "b2", "b3", "mir"],
}
# Sentinel-5P product type -> (label, asset key). Sentinel-5 itself is not catalogued.
S5P_PRODUCTS = {
    "L2__NO2___": ("Nitrogen dioxide", "no2"),
    "L2__O3____": ("Ozone", "o3"),
    "L2__SO2___": ("Sulfur dioxide", "so2"),
    "L2__CO____": ("Carbon monoxide", "co"),
    "L2__CH4___": ("Methane", "ch4"),
    "L2__HCHO__": ("Formaldehyde", "hcho"),
    "L2__AER_AI": ("Aerosol index", "aer-ai"),
    "L2__AER_LH": ("Aerosol layer height", "aer-lh"),
    "L2__CLOUD_": ("Cloud", "cloud"),
}
COLLECTIONS = {
    "earth-search": ["sentinel-2-l2a", "sentinel-2-l1c", "landsat-c2-l2", "naip"],
    "planetary": [
        "sentinel-2-l2a", "sentinel-1-rtc", "sentinel-1-grd", "landsat-c2-l2",
        "modis-17A2H-061", "modis-15A2H-061", "modis-09Q1-061", "modis-13Q1-061",
        "hls2-s30", "hls2-l30",
        "sentinel-3-synergy-vg1-l2-netcdf", "sentinel-3-olci-lfr-l2-netcdf",
        "sentinel-3-slstr-lst-l2-netcdf", "sentinel-3-olci-wfr-l2-netcdf",
        "sentinel-3-slstr-wst-l2-netcdf", "sentinel-3-slstr-frp-l2-netcdf",
        "sentinel-3-synergy-syn-l2-netcdf", "sentinel-3-synergy-aod-l2-netcdf",
        "sentinel-5p-l2-netcdf", "jrc-gsw",
    ],
    "cdse": ["SENTINEL-2", "SENTINEL-1", "SENTINEL-3", "SENTINEL-5P",
             "LANDSAT-8", "LANDSAT-9"],
    "noaa": ["noaa-rgb8", "noaa-rgb16", "noaa-rgbn8", "noaa-cir8", "noaa-naip"],
    "usgs": ["lidar-lpc", "lidar-projects"],
}
COLLECTION_LABELS = {
    "sentinel-2-l2a": "Sentinel-2 L2A",
    "sentinel-2-l1c": "Sentinel-2 L1C",
    "sentinel-1-rtc": "Sentinel-1 RTC",
    "sentinel-1-grd": "Sentinel-1 GRD",
    "landsat-c2-l2": "Landsat 8–9 Collection 2 L2",
    "naip": "USDA NAIP (latest)",
    "modis-17A2H-061": "MODIS Gross Primary Productivity 8-Day",
    "modis-15A2H-061": "MODIS Leaf Area Index/FPAR 8-Day",
    "modis-09Q1-061": "MODIS Surface Reflectance 8-Day (250m)",
    "modis-13Q1-061": "MODIS Vegetation Indices 16-Day (250m)",
    "hls2-s30": "HLS v2.0, Sentinel-2",
    "hls2-l30": "HLS v2.0, Landsat",
    "sentinel-3-synergy-vg1-l2-netcdf": "Sentinel-3 1-Day Surface Reflectance and NDVI (SPOT VEGETATION)",
    "sentinel-3-olci-lfr-l2-netcdf": "Sentinel-3 Land (Full Resolution)",
    "sentinel-3-slstr-lst-l2-netcdf": "Sentinel-3 Land Surface Temperature",
    "sentinel-3-olci-wfr-l2-netcdf": "Sentinel-3 Water (Full Resolution)",
    "sentinel-3-slstr-wst-l2-netcdf": "Sentinel-3 sea surface temperature",
    "sentinel-3-slstr-frp-l2-netcdf": "Sentinel-3 fire radiative power",
    "sentinel-3-synergy-syn-l2-netcdf": "Sentinel-3 synergy reflectance",
    "sentinel-3-synergy-aod-l2-netcdf": "Sentinel-3 synergy aerosol",
    "sentinel-5p-l2-netcdf": "Sentinel-5P Level-2",
    "jrc-gsw": "JRC Global Surface Water",
    "SENTINEL-2": "Sentinel-2",
    "SENTINEL-1": "Sentinel-1",
    "SENTINEL-3": "Sentinel-3",
    "SENTINEL-5P": "Sentinel-5P",
    "LANDSAT-8": "Landsat 8",
    "LANDSAT-9": "Landsat 9",
    "noaa-rgb8": "Natural color, 8-bit",
    "noaa-rgb16": "Natural color, 16-bit",
    "noaa-rgbn8": "Color and near infrared, 8-bit",
    "noaa-cir8": "Color infrared, 8-bit",
    "noaa-naip": "USDA NAIP (latest)",
    "lidar-lpc": "Lidar point cloud (LAZ)",
    "lidar-projects": "Lidar projects",
}
CDSE_TYPES = {
    "SENTINEL-2": ["S2MSI2A", "S2MSI1C"],
    "SENTINEL-1": ["GRD", "SLC"],
    "SENTINEL-3": [
        "OL_2_LFR___", "OL_2_WFR___", "OL_1_EFR___",
        "SL_2_LST___", "SL_2_WST___", "SL_2_FRP___", "SL_1_RBT___",
        "SY_2_SYN___", "SY_2_AOD___", "SY_2_VGP___", "SY_2_VG1___", "SY_2_V10___",
    ],
    "SENTINEL-5P": [
        "L2__NO2___", "L2__O3____", "L2__SO2___", "L2__CO____", "L2__CH4___",
        "L2__HCHO__", "L2__AER_AI", "L2__AER_LH", "L2__CLOUD_",
        "L2B_SIF___", "L2__BRO___",
    ],
    "LANDSAT-8": ["L1TP", "L1GT"],
    "LANDSAT-9": ["L1TP", "L1GT"],
}
CDSE_TYPE_LABELS = {
    "S2MSI2A": "L2A surface reflectance",
    "S2MSI1C": "L1C top of atmosphere",
    "GRD": "Ground range detected",
    "SLC": "Single look complex",
    "OL_2_LFR___": "OLCI land colour (full resolution)",
    "OL_2_WFR___": "OLCI water colour (full resolution)",
    "OL_1_EFR___": "OLCI L1 radiance (full resolution)",
    "SL_2_LST___": "SLSTR land surface temperature",
    "SL_2_WST___": "SLSTR sea surface temperature",
    "SL_2_FRP___": "SLSTR fire radiative power",
    "SL_1_RBT___": "SLSTR L1 radiances",
    "SY_2_SYN___": "Synergy surface reflectance",
    "SY_2_AOD___": "Synergy aerosol",
    "SY_2_VGP___": "Synergy top-of-atmosphere reflectance",
    "SY_2_VG1___": "Synergy 1-day NDVI",
    "SY_2_V10___": "Synergy 10-day NDVI",
    "L2__NO2___": "Nitrogen dioxide",
    "L2__O3____": "Ozone",
    "L2__SO2___": "Sulfur dioxide",
    "L2__CO____": "Carbon monoxide",
    "L2__CH4___": "Methane",
    "L2__HCHO__": "Formaldehyde",
    "L2__AER_AI": "Aerosol index",
    "L2__AER_LH": "Aerosol layer height",
    "L2__CLOUD_": "Cloud",
    "L2B_SIF___": "Solar-induced fluorescence",
    "L2__BRO___": "Bromine monoxide",
    "L1TP": "L1TP precision terrain",
    "L1GT": "L1GT systematic terrain",
}
LS_PLATFORM_CHOICES = {
    "Landsat 8 and 9": "landsat-8,landsat-9",
    "Landsat 8": "landsat-8",
    "Landsat 9": "landsat-9",
}


def mission(collection: str) -> str:
    c = collection.lower()
    if "sentinel-1" in c:
        return "s1"
    if c.startswith("landsat"):
        return "landsat"
    if "sentinel-3" in c:
        return "s3"
    if "sentinel-5" in c:
        return "s5p"
    if c.startswith("noaa"):
        return "noaa"
    if c.startswith("lidar"):
        return "lidar"
    if c == "naip":
        return "naip"
    if c.startswith("modis-"):
        return "modis"
    if c.startswith("hls2-"):
        return "hls"
    if c == "jrc-gsw":
        return "jrc"
    return "s2"

ss = st.session_state
ss.setdefault("aoi_geom", None)
ss.setdefault("results", None)        # list of STAC item dicts or CDSE products
ss.setdefault("results_source", None)
ss.setdefault("last_run", None)
ss.setdefault("applied_bbox", "")
ss.setdefault("applied_upload", None)
ss.setdefault("browse_id", None)
ss.setdefault("thumb_cache", {})


# --------------------------------------------------------------------------- #
# Cached searches (return plain dicts so Streamlit can cache them)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=1800, show_spinner=False)
def search_stac(source, collection, geom_json, start, end, cloud, limit,
                platforms, s5p_type, s1_mode, s1_pol, s1_orbit):
    geom = json.loads(geom_json)
    aoi = sd.AOI(tuple(shape(geom).bounds), geom)
    extra = {}
    if s5p_type:
        extra["s5p:product_type"] = {"eq": s5p_type}
    if s1_mode:
        extra["sar:instrument_mode"] = {"eq": s1_mode}
        extra["sar:polarizations"] = {"eq": [b.upper() for b in S1_POLARIZATIONS[s1_pol][1]]}
        extra["sat:orbit_state"] = {"eq": s1_orbit}
    plats = tuple(platforms.split(",")) if platforms else None
    items = sd.stac_search(source, collection, aoi, start, end, cloud, limit,
                           plats, extra or None)
    return [i.to_dict() for i in items]


NAIP_STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "PR": "Puerto Rico", "VI": "U.S. Virgin Islands",
}


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_naip_coverage():
    # Cache key includes this body, so a catalogue-parser change refetches.
    return sd.naip_coverage()


def show_naip_coverage(coastal_note: bool):
    """Nationwide USDA NAIP years and where each year was flown."""
    st.markdown("### USDA NAIP availability")
    if coastal_note:
        st.caption(
            "NOAA’s download uses the coastal mosaic. The map and grid are the "
            "nationwide USDA catalogue: which years were flown in each state."
        )
    else:
        st.caption(
            "Years on file for each state. A download uses the newest of these "
            "years that covers your area."
        )
    try:
        with st.spinner("Loading NAIP coverage…"):
            rows = load_naip_coverage()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"NAIP coverage could not be loaded: {exc}")
        return
    if not rows:
        st.info("The NAIP catalogue did not return any state folders.")
        return

    frame = pd.DataFrame(rows)
    frame["state"] = frame["code"].map(lambda c: NAIP_STATE_NAMES.get(c, c))
    frame["years label"] = frame["years"].map(lambda ys: ", ".join(str(y) for y in ys))
    ymin, ymax = int(frame["earliest"].min()), int(frame["latest"].max())
    states = frame[frame["code"].str.len().eq(2) & ~frame["code"].isin(["PR", "VI"])]
    islands = [NAIP_STATE_NAMES[c] for c in ("HI", "PR", "VI") if c in set(frame["code"])]
    island_text = ", ".join(islands) if islands else "none on file"

    st.markdown(
        f"**{int(states['code'].nunique())} states** have imagery from **{ymin}** to "
        f"**{ymax}**. The newest year in the catalogue is **{int(frame['latest'].max())}**. "
        f"Island areas on file: {island_text}."
    )

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Latest year by state")
        fig = px.choropleth(
            states, locations="code", locationmode="USA-states",
            color="latest", scope="usa", hover_name="state",
            hover_data={"years label": True, "code": False, "latest": True},
            color_continuous_scale="YlGn",
            labels={"latest": "Latest year", "years label": "Years"},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=420,
                          coloraxis_colorbar=dict(title="Year", tickformat="d"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Alaska is outside this NAIP catalogue. Puerto Rico and the "
                   "U.S. Virgin Islands are listed in the table.")
    with right:
        st.markdown("#### Years on file")
        table = frame.sort_values("state")[
            ["state", "latest", "years label"]
        ].rename(columns={"state": "State", "latest": "Latest",
                          "years label": "Years available"})
        st.dataframe(table, hide_index=True, height=420, width="stretch")

    st.markdown("#### Coverage by year")
    year_cols = list(range(ymin, ymax + 1))
    grid = pd.DataFrame(
        [[1 if year in years else 0 for year in year_cols] for years in frame["years"]],
        index=frame["code"], columns=year_cols,
    ).sort_index()
    heat = px.imshow(
        grid, color_continuous_scale=["#eef2f6", "#1f6feb"], aspect="auto",
        labels=dict(x="Year", y="State", color="Flown"),
        height=max(420, 18 * len(grid)),
    )
    heat.update_coloraxes(showscale=False)
    heat.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    heat.update_xaxes(side="top", dtick=1)
    heat.update_traces(
        hovertemplate="State %{y}<br>Year %{x}<br>%{customdata}<extra></extra>",
        customdata=[["Flown" if v else "Not flown" for v in row] for row in grid.to_numpy()],
    )
    st.plotly_chart(heat, width="stretch")


@st.cache_data(ttl=1800, show_spinner=False)
def search_naip(geom_json, limit):
    geom = json.loads(geom_json)
    aoi = sd.AOI(tuple(shape(geom).bounds), geom)
    return [i.to_dict() for i in sd.naip_search(aoi, limit)]


@st.cache_data(ttl=1800, show_spinner=False)
def search_noaa(collection, geom_json, start, end, limit):
    geom = json.loads(geom_json)
    aoi = sd.AOI(tuple(shape(geom).bounds), geom)
    return sd.NOAA().search(collection, aoi, start, end, limit)


@st.cache_data(ttl=1800, show_spinner=False)
def search_usgs(collection, geom_json, start, end, limit):
    geom = json.loads(geom_json)
    aoi = sd.AOI(tuple(shape(geom).bounds), geom)
    return sd.USGS().search(collection, aoi, start, end, limit)


@st.cache_data(ttl=1800, show_spinner=False)
def search_cdse(collection, geom_json, start, end, cloud, ptype, limit,
                s1_mode, s1_pol, s1_orbit):
    geom = json.loads(geom_json)
    aoi = sd.AOI(tuple(shape(geom).bounds), geom)
    return sd.CDSE().search(
        collection, aoi, start, end, cloud, ptype, limit,
        s1_mode=s1_mode or None,
        s1_polarisation=S1_CDSE_POL.get(s1_pol),
        s1_orbit=s1_orbit.upper() if s1_orbit else None)


def to_items(dicts):
    import pystac
    return [pystac.Item.from_dict(d) for d in dicts]


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def reflectance(arr, item_dict, asset_key):
    """Convert Sentinel-2 L2A DN to reflectance using STAC scale/offset."""
    arr = arr.astype("float32")
    asset = item_dict.get("assets", {}).get(asset_key, {})
    rb = (asset.get("raster:bands") or [{}])[0]
    if "scale" in rb or "offset" in rb:
        return arr * rb.get("scale", 1.0) + rb.get("offset", 0.0)
    baseline = item_dict.get("properties", {}).get("s2:processing_baseline")
    try:
        offset = -1000 if baseline and float(baseline) >= 4.0 else 0
    except ValueError:
        offset = 0
    return (arr + offset) / 10000.0


def read_band(path):
    import rasterio
    with rasterio.open(path) as r:
        a = r.read(1).astype("float32")
        if r.nodata is not None:
            a[a == r.nodata] = np.nan
        return a


def stretch(a, lo=2, hi=98):
    v = a[np.isfinite(a)]
    if v.size == 0:
        return np.zeros_like(a)
    p1, p2 = np.percentile(v, (lo, hi))
    return np.clip((a - p1) / (p2 - p1 + 1e-9), 0, 1)


def quicklook_rgb(idir, item_dict, band_map):
    arrs = []
    for b in ("red", "green", "blue"):
        a = read_band(idir / f"{b}.tif")
        a[a == 0] = np.nan
        arrs.append(stretch(reflectance(a, item_dict, band_map[b])))
    rgb = np.dstack(arrs)
    return (np.nan_to_num(rgb) * 255).astype("uint8")


def scene_cards(results, source, aoi):
    """One card per search result: id, caption, and a browser thumbnail."""
    cards = []
    naip_hrefs = []
    for item in results:
        if source == "cdse":
            when = (item.get("ContentDate") or {}).get("Start") or ""
            cards.append({
                "id": item["Name"],
                "title": item["Name"],
                "caption": when[:10],
                "image": sd.cdse_quicklook(item),
            })
            continue
        if source == "noaa":
            props = item["properties"]
            cards.append({
                "id": item["id"],
                "title": props.get("title") or item["id"],
                "caption": " · ".join(
                    part for part in (
                        str(props.get("year") or ""),
                        f"{props.get('resolution')} m" if props.get("resolution") else "",
                        props.get("sensor") or "",
                    ) if part
                ),
                "image": sd.NOAA().preview_url(item, aoi),
            })
            continue
        if source == "usgs":
            props = item["properties"]
            info = item.get("lidar") or {}
            size = props.get("size_mb")
            if info.get("kind") == "project":
                caption = " · ".join(
                    part for part in (
                        props.get("quality") or "",
                        (props.get("datetime") or "")[:10],
                        props.get("method") or "",
                    ) if part
                )
            else:
                caption = " · ".join(
                    part for part in (
                        (props.get("datetime") or "")[:10],
                        f"{size:.0f} MB" if isinstance(size, (int, float)) else "",
                        "LAZ",
                    ) if part
                )
            cards.append({
                "id": item["id"],
                "title": props.get("title") or item["id"],
                "caption": caption,
                "image": info.get("preview"),
                "link": info.get("download") or info.get("metadata") or info.get("folder"),
                "link_label": ("Download this LAZ from USGS" if info.get("download")
                               else "Open the project folder"),
            })
            continue
        props = item.get("properties") or {}
        when = (props.get("datetime") or "")[:10]
        cloud = props.get("eo:cloud_cover")
        bits = [when] if when else []
        if isinstance(cloud, (int, float)):
            bits.append(f"{cloud:.0f}% cloud")
        if props.get("naip:year") is not None:
            bits = [f"NAIP {props.get('naip:year')}",
                    (props.get("naip:state") or "").upper()]
        href = sd.naip_image_href(item)
        if href:
            naip_hrefs.append(href)
            image = ("naip", href)
        else:
            image = sd.browse_href(item)
        cards.append({
            "id": item["id"],
            "title": item["id"],
            "caption": " · ".join(bit for bit in bits if bit),
            "image": image,
        })
    missing = [h for h in naip_hrefs if h not in ss.thumb_cache]
    if missing:
        with st.spinner("Loading scene thumbnails…"):
            ss.thumb_cache.update(sd.naip_previews(missing))
    for card in cards:
        image = card["image"]
        if isinstance(image, tuple) and image and image[0] == "naip":
            card["image"] = ss.thumb_cache.get(image[1])
    return cards


def show_scene_browser(results, source, aoi):
    """Thumbnail grid plus a larger view of the scene you pick."""
    cards = scene_cards(results, source, aoi)
    if not cards:
        return
    st.markdown("#### Scene browser")
    if source == "noaa":
        st.caption("Each thumbnail is this dataset over the area you drew. "
                   "Choose one to view it larger.")
    elif source == "usgs":
        st.caption("USGS browse images for the lidar in this search. "
                   "Choose one to view it larger.")
    else:
        st.caption("Thumbnails of the scenes in this search. Choose one to view it.")
    ids = [card["id"] for card in cards]
    if ss.browse_id not in ids:
        ss.browse_id = ids[0]

    per_row = 4
    for start in range(0, len(cards), per_row):
        cols = st.columns(per_row)
        for col, card in zip(cols, cards[start:start + per_row]):
            with col:
                if card["image"] is not None:
                    st.image(card["image"], width="stretch")
                else:
                    st.caption("No thumbnail for this scene.")
                st.caption(card["caption"] or card["title"])
                viewing = card["id"] == ss.browse_id
                if st.button("Viewing" if viewing else "View",
                             key=f"browse-{card['id']}",
                             disabled=viewing,
                             width="stretch"):
                    ss.browse_id = card["id"]
                    st.rerun()

    chosen = next(card for card in cards if card["id"] == ss.browse_id)
    st.markdown(f"##### {chosen['title']}")
    if chosen["caption"]:
        st.caption(chosen["caption"])
    if chosen.get("link"):
        st.markdown(f"[{chosen.get('link_label') or 'Open the National Map record'}]({chosen['link']})")
    if chosen["image"] is not None:
        left, _right = st.columns([3, 2])
        with left:
            st.image(chosen["image"], width="stretch")
    else:
        st.caption("This scene has no browser thumbnail.")


def ndvi(idir, item_dict, band_map):
    red = read_band(idir / "red.tif")
    nir = read_band(idir / "nir.tif")
    if red.shape != nir.shape:
        return None
    red[red == 0] = np.nan
    nir[nir == 0] = np.nan
    r = reflectance(red, item_dict, band_map["red"])
    n = reflectance(nir, item_dict, band_map["nir"])
    with np.errstate(divide="ignore", invalid="ignore"):
        return (n - r) / (n + r)


def s1_db(idir, band):
    a = read_band(idir / f"{band}.tif")
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(np.where(a > 0, a, np.nan))
    return db


def zip_folder(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in folder.rglob("*"):
            if f.is_file() and not f.name.endswith(".part"):
                z.write(f, f.relative_to(folder.parent))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.image(LOGO, width=LOGO_WIDTH)
    st.title("PyGIS Explorer")
    st.caption("Search and download Sentinel, Landsat, NOAA imagery, and USGS lidar.")
    source = st.selectbox("Source", list(COLLECTIONS),
                          format_func={"earth-search": "Earth Search (AWS)",
                                       "planetary": "Planetary Computer",
                                       "cdse": "Copernicus (CDSE)",
                                       "noaa": "NOAA Digital Coast",
                                       "usgs": "USGS 3DEP Lidar"}.get)
    collection = st.selectbox("Collection", COLLECTIONS[source],
                              format_func=lambda c: COLLECTION_LABELS.get(c, c))
    kind = mission(collection)
    is_s1 = kind == "s1"
    is_netcdf = "netcdf" in collection

    c1, c2 = st.columns(2)
    if source == "usgs":
        start = c1.date_input("Start", date(2000, 1, 1), key="lidar-start")
        end = c2.date_input("End", date.today(), key="lidar-end")
    else:
        start = c1.date_input("Start", date.today() - timedelta(days=60))
        end = c2.date_input("End", date.today())
    cloud = None
    if kind in ("s2", "landsat", "hls"):
        cloud = st.slider("Max cloud cover (%)", 0, 100, 20)
    if collection == "naip" or collection == "lidar-lpc":
        limit_label = "Max tiles"
    elif collection == "lidar-projects":
        limit_label = "Max projects"
    elif source == "noaa":
        limit_label = "Max datasets"
    else:
        limit_label = "Max scenes"
    if source == "usgs":
        limit = st.number_input(limit_label, 1, 40, 8)
    else:
        limit = st.number_input(limit_label, 1, 500, 30)
    if collection in ("naip", "noaa-naip"):
        st.caption("USDA NAIP uses the newest year available for this area.")

    ptype = None
    s5p_type = ""
    platforms = ""
    s1_mode = s1_pol = s1_orbit = ""
    bands, mode, stack = [], "clip", False
    if kind == "s1":
        s1_mode = st.selectbox("Acquisition mode", list(S1_MODES), index=1,
                               format_func=S1_MODES.get)
        s1_pol = st.selectbox("Polarization", list(S1_POLARIZATIONS), index=1,
                              format_func=lambda k: S1_POLARIZATIONS[k][0])
        s1_orbit = st.selectbox("Orbit direction", ["ascending", "descending"],
                                format_func=str.capitalize)
        if collection == "sentinel-1-rtc" and s1_mode != "IW":
            st.caption("RTC on Planetary Computer is Interferometric Wide. "
                       "Use Sentinel-1 GRD or Copernicus for Stripmap and Extra-Wide.")
    if source == "cdse":
        ptype = st.selectbox(
            "Product type", CDSE_TYPES[collection],
            format_func=lambda k: CDSE_TYPE_LABELS.get(k, k))
        with st.expander("Copernicus login", expanded=not os.environ.get("CDSE_USER")):
            u = st.text_input("Username (email)", os.environ.get("CDSE_USER", ""))
            p = st.text_input("Password", type="password")
            if u:
                os.environ["CDSE_USER"] = u
            if p:
                os.environ["CDSE_PASSWORD"] = p
        if kind == "s5p":
            st.caption("Sentinel-5 Precursor (TROPOMI). Sentinel-5 is not in "
                       "the Copernicus catalogue.")
        elif kind == "landsat":
            st.caption("USGS Collection 2 Level-1 scenes, downloaded whole.")
        elif kind == "s3":
            st.caption("Sentinel-3 products are downloaded whole.")
    elif source == "noaa":
        mode = "full"
        if collection == "noaa-naip":
            st.caption(
                "Latest USDA NAIP aerial imagery in the NOAA "
                "[Data Access Viewer](https://coast.noaa.gov/dataviewer/#/imagery/search) "
                "mosaic. No account is required. Coverage is coastal, and the year is "
                "the newest NAIP dataset NOAA has for this shoreline. Download saves "
                "a GeoTIFF of the area. Large areas are resampled to fit NOAA's "
                "export size limit."
            )
        else:
            st.caption(
                "NOAA Digital Coast imagery from the "
                "[Data Access Viewer](https://coast.noaa.gov/dataviewer/#/imagery/search). "
                "No account is required. The start and end dates select a year range, "
                "and each result is one coastal dataset that intersects the area. "
                "Download saves a GeoTIFF of that area. Large areas are resampled "
                "so the image stays within NOAA's export size limit. Coverage is coastal."
            )
    elif source == "usgs":
        mode = "full"
        if collection == "lidar-projects":
            st.caption(
                "USGS 3DEP lidar projects from "
                "[Lidar Explorer](https://apps.nationalmap.gov/lidar-explorer/#/). "
                "No account is required. Each row is a project whose collection "
                "dates meet the range above and whose footprint meets the area. "
                "Download saves the LAZ tiles from that project inside the area, "
                "up to the max you set, plus a small file of the project links."
            )
        else:
            st.caption(
                "USGS 3DEP lidar point clouds from "
                "[Lidar Explorer](https://apps.nationalmap.gov/lidar-explorer/#/). "
                "No account is required. Each row is one LAZ tile published in "
                "the date range. Tiles are often 20–80 MB."
            )
    elif kind in ("modis", "hls", "jrc"):
        table = sd.PLANETARY_BANDS[collection]
        bands = st.multiselect(
            "Layers" if kind == "jrc" else "Bands",
            list(table), default=sd.PLANETARY_DEFAULT_BANDS[collection],
            key=f"pc-bands-{collection}")
        if kind == "jrc":
            mode = "clip"
            st.caption(
                "JRC Global Surface Water from Planetary Computer. "
                "Each tile is the 1984–2020 mosaic, so the start and end dates "
                "are not used. Download clips the selected layers to the area you drew."
            )
        else:
            mode = st.radio("Download mode", ["clip", "full"], horizontal=True,
                            help="clip = only your area (small, fast); full = whole tile")
            stack = st.checkbox("Also write a multi-band stack.tif", value=kind == "hls",
                                disabled=mode != "clip")
            if kind == "modis":
                st.caption(
                    "MODIS Collection 6.1 from Planetary Computer. No account is "
                    "required. Terra and Aqua tiles are both included."
                )
            else:
                st.caption(
                    "Harmonized Landsat and Sentinel-2 (HLS) version 2.0 from "
                    "Planetary Computer."
                )
    elif kind == "naip":
        bands = ["red", "green", "blue", "nir"]
        mode = "clip"
        st.caption(
            "Latest USDA NAIP aerial imagery from Earth Search on AWS. "
            "Each row is a tile from the newest year covering this area. "
            "Download saves a clip with red, green, blue, and near infrared."
        )
    elif is_netcdf:
        if kind == "s5p":
            s5p_type = st.selectbox(
                "Product", list(S5P_PRODUCTS),
                format_func=lambda k: S5P_PRODUCTS[k][0])
            bands = [S5P_PRODUCTS[s5p_type][1]]
            st.caption("Sentinel-5 Precursor (TROPOMI) NetCDF. Sentinel-5 is not "
                       "in this catalogue. The whole product file is downloaded.")
        else:
            opts = S3_VARIABLES[collection]
            bands = st.multiselect("Variables", opts, default=opts[:min(3, len(opts))],
                                   key=f"vars-{collection}")
            st.caption("Sentinel-3 NetCDF. Each variable file is downloaded whole.")
        mode = "full"
    else:
        st.divider()
        if kind == "landsat":
            sat = st.selectbox("Satellite", list(LS_PLATFORM_CHOICES))
            platforms = LS_PLATFORM_CHOICES[sat]
            options, default = LS_BANDS, ["red", "green", "blue", "nir"]
            st.caption("Collection 2 Level-2 surface reflectance.")
        elif kind == "s1":
            bands = list(S1_POLARIZATIONS[s1_pol][1])
            st.caption("Polarization selects the bands that are downloaded.")
            options, default = None, None
        else:
            options, default = S2_BANDS, ["red", "green", "blue", "nir", "scl"]
        if options is not None:
            bands = st.multiselect("Bands", options, default=default,
                                   key=f"bands-{collection}")
        mode = st.radio("Download mode", ["clip", "full"], horizontal=True,
                        help="clip = only your area (small, fast); full = whole tile")
        stack = st.checkbox("Also write a multi-band stack.tif", value=kind != "s1",
                            disabled=mode != "clip" or kind == "s1")

    st.divider()
    st.subheader("Area of interest")
    st.caption("Draw a rectangle or polygon on the map, or enter one below.")
    # Widget values can only be reset before those widgets are created.
    if ss.pop("_clear_aoi", False):
        ss.bbox_text = ""
        ss.pop("aoi_upload", None)
    bbox_txt = st.text_input("BBox  minlon,minlat,maxlon,maxlat", key="bbox_text")
    up = st.file_uploader("…or upload GeoJSON", type=["geojson", "json"], key="aoi_upload")
    if bbox_txt.strip() and bbox_txt != ss.applied_bbox:
        try:
            ss.aoi_geom = sd.AOI.from_bbox(
                [float(x) for x in bbox_txt.split(",")]).geometry
            ss.applied_bbox = bbox_txt
        except (ValueError, SystemExit):
            st.error("BBox must be four numbers: minlon,minlat,maxlon,maxlat")
    if up is not None:
        token = f"{up.name}:{up.size}"
        if token != ss.applied_upload:
            tmp = OUT_ROOT / "_upload.geojson"
            tmp.parent.mkdir(exist_ok=True)
            tmp.write_bytes(up.getvalue())
            try:
                ss.aoi_geom = sd.AOI.from_geojson(tmp).geometry
                ss.applied_upload = token
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read that GeoJSON: {exc}")
    if ss.aoi_geom and st.button("Clear AOI"):
        ss.aoi_geom = None
        ss.results = None
        ss.applied_bbox = ""
        ss.applied_upload = None
        ss._clear_aoi = True
        st.rerun()

    with st.container(key="upatta_credit", horizontal=True,
                      vertical_alignment="center", gap="small"):
        st.image(UPATTA_LOGO, width=64, alt="Upatta Data Analytics")
        st.markdown("**Zia Ahmed, PhD**")


# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    .pygis-hero h1 { font-size: 1.7rem; font-weight: 750; margin: 0; letter-spacing: -0.03em; }
    .pygis-hero p { margin: 0.15rem 0 0.8rem 0; color: #4d5d6e; }
    [data-testid="stSidebarUserContent"] {
        overflow-y: auto;
        max-height: calc(100vh - 156px);
        box-sizing: border-box;
    }
    .st-key-upatta_credit {
        position: absolute;
        left: 0;
        right: 0;
        bottom: 0;
        width: auto;
        box-sizing: border-box;
        margin: 0;
        padding: 4px 12px 6px;
        background: #ffffff;
        border-top: 1px solid rgba(20, 32, 51, 0.12);
        z-index: 1000001;
    }
    .st-key-upatta_credit p { margin: 0; }
    [data-testid="stSidebar"][data-collapsed="true"] .st-key-upatta_credit { display: none; }
    </style>
    <div class="pygis-hero">
      <h1><span style="color:#d4920b">Py</span>GIS Explorer</h1>
      <p>Draw an area, search the catalogue, and download the scenes you select.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
def wrap_geometry(geom: dict) -> dict:
    """Move a drawing that landed on the next copy of the world back to -180..180."""
    bounds = shape(geom).bounds
    minx, maxx = bounds[0], bounds[2]
    shift = 0.0
    if minx > 180:
        shift = -360.0 * math.ceil((minx - 180) / 360)
    elif maxx < -180:
        shift = 360.0 * math.ceil((-180 - maxx) / 360)
    if not shift:
        return geom
    return mapping(translate(shape(geom), xoff=shift))


st.markdown("### Map")
if ss.aoi_geom:
    fixed = wrap_geometry(ss.aoi_geom)
    if fixed != ss.aoi_geom:
        ss.aoi_geom = fixed
center = [42.44, -76.50]
if ss.aoi_geom:
    c = shape(ss.aoi_geom).centroid
    center = [c.y, c.x]
m = folium.Map(location=center, zoom_start=11 if ss.aoi_geom else 7,
               tiles=None, control_scale=True, max_bounds=True)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Satellite", overlay=False, show=True).add_to(m)
folium.TileLayer("OpenStreetMap", name="Streets", overlay=False, show=False).add_to(m)
Draw(draw_options={"polyline": False, "circle": False, "marker": False,
                   "circlemarker": False, "polygon": True, "rectangle": True},
     edit_options={"edit": False}).add_to(m)

if ss.results and ss.results_source != "cdse":
    fg = folium.FeatureGroup("Scene footprints")
    for d in ss.results:
        if not d.get("geometry"):
            continue
        folium.GeoJson(d["geometry"], tooltip=d["id"],
                       style_function=lambda _: {"color": "#2b7bb9", "weight": 1,
                                                 "fillOpacity": 0.04}).add_to(fg)
    fg.add_to(m)
if ss.aoi_geom:
    folium.GeoJson(ss.aoi_geom, name="AOI",
                   style_function=lambda _: {"color": "#e4572e", "weight": 2,
                                             "fillOpacity": 0.1}).add_to(m)
folium.LayerControl(collapsed=True).add_to(m)

map_out = st_folium(m, height=430, use_container_width=True,
                    returned_objects=["last_active_drawing"])
drawn = (map_out or {}).get("last_active_drawing")
if drawn and drawn.get("geometry"):
    geom = wrap_geometry(drawn["geometry"])
    if geom != ss.aoi_geom:
        ss.aoi_geom = geom
        st.rerun()

if ss.aoi_geom:
    b = shape(ss.aoi_geom).bounds
    st.caption(f"AOI bounds: {b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}, {b[3]:.4f}")

if collection in ("naip", "noaa-naip"):
    show_naip_coverage(collection == "noaa-naip")


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
if source == "usgs":
    search_label = "🔎 Search lidar"
else:
    search_label = "🔎 Search scenes"
if st.button(search_label, type="primary", disabled=not ss.aoi_geom):
    gj = json.dumps(ss.aoi_geom)
    with st.spinner("Searching…"):
        try:
            if source == "cdse":
                ss.results = search_cdse(collection, gj, str(start), str(end),
                                         cloud, ptype, int(limit),
                                         s1_mode, s1_pol, s1_orbit)
            elif source == "noaa":
                ss.results = search_noaa(collection, gj, str(start), str(end),
                                         int(limit))
            elif source == "usgs":
                ss.results = search_usgs(collection, gj, str(start), str(end),
                                         int(limit))
            elif collection == "naip":
                ss.results = search_naip(gj, int(limit))
            else:
                ss.results = search_stac(source, collection, gj, str(start), str(end),
                                         cloud, int(limit), platforms, s5p_type,
                                         s1_mode, s1_pol, s1_orbit)
            ss.results_source = source
        except SystemExit as e:  # CDSE credential errors
            st.error(str(e))
            ss.results = None
        except Exception as e:  # noqa: BLE001
            st.error(f"Search failed: {e}")
            ss.results = None
    st.rerun()

if not ss.aoi_geom:
    st.info("Draw an area on the map (or enter a bbox in the sidebar) to begin.")
    st.stop()
if ss.results is None:
    st.stop()
if not ss.results:
    if is_s1:
        st.warning("No scenes found for this mode, polarization, and orbit. "
                   "Try the other orbit, a wider date range, or Sentinel-1 GRD "
                   "for Stripmap and Extra-Wide.")
    elif collection == "naip":
        st.warning("No USDA NAIP imagery covers this area. NAIP covers the "
                   "conterminous United States and Hawaii.")
    elif collection == "noaa-naip":
        st.warning("No USDA NAIP in the NOAA coastal catalogue covers this area. "
                   "Earth Search (AWS) has nationwide NAIP. NOAA coverage is coastal.")
    elif source == "noaa":
        st.warning("No NOAA coastal imagery intersects this area and year range. "
                   "Coverage is coastal — try a shoreline area or a wider range of years.")
    elif source == "usgs":
        st.warning("No 3DEP lidar meets this area and date range. "
                   "Widen the dates or try another area. Many projects were "
                   "collected years ago.")
    elif collection == "jrc-gsw":
        st.warning("No JRC Global Surface Water tile covers this area.")
    else:
        st.warning("No scenes found — widen the dates or raise the cloud limit.")
    st.stop()


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
if ss.results_source == "noaa":
    rows = []
    for d in ss.results:
        props = d["properties"]
        rows.append({
            "download": True,
            "year": props.get("year"),
            "dataset": props.get("title"),
            "resolution m": props.get("resolution"),
            "tiles": props.get("tiles"),
            "sensor": props.get("sensor"),
            "id": d["id"],
        })
    df = pd.DataFrame(rows)
elif ss.results_source == "usgs":
    rows = []
    lidar_kind = (ss.results[0].get("lidar") or {}).get("kind")
    for d in ss.results:
        props = d["properties"]
        if lidar_kind == "project":
            rows.append({
                "download": True,
                "collected": (props.get("datetime") or "")[:10],
                "quality": props.get("quality"),
                "project": props.get("project"),
                "work unit": props.get("title"),
                "method": props.get("method"),
                "id": d["id"],
            })
        else:
            rows.append({
                "download": True,
                "published": (props.get("datetime") or "")[:10],
                "size MB": props.get("size_mb"),
                "tile": props.get("title"),
                "id": d["id"],
            })
    df = pd.DataFrame(rows)
elif ss.results_source == "cdse":
    rows = []
    for p in ss.results:
        row = {
            "download": True,
            "date": p["ContentDate"]["Start"][:16].replace("T", " "),
            "cloud %": sd.cdse_attr(p, "cloudCover"),
            "size MB": round((p.get("ContentLength") or 0) / 1e6),
            "name": p["Name"],
        }
        if sd.cdse_attr(p, "operationalMode"):
            row["mode"] = sd.cdse_attr(p, "operationalMode")
            row["polarization"] = (sd.cdse_attr(p, "polarisationChannels") or "").replace("&", "+")
            row["orbit"] = (sd.cdse_attr(p, "orbitDirection") or "").capitalize()
        rows.append(row)
    df = pd.DataFrame(rows)
else:
    rows = []
    for d in ss.results:
        props = d["properties"]
        row = {
            "download": True,
            "date": (props.get("datetime") or "")[:16].replace("T", " "),
            "cloud %": props.get("eo:cloud_cover"),
            "platform": props.get("platform"),
            "tile": props.get("s2:mgrs_tile") or props.get("sat:relative_orbit"),
            "id": d["id"],
        }
        if props.get("naip:year") is not None:
            row["NAIP year"] = props.get("naip:year")
            row["gsd m"] = props.get("gsd")
            row["state"] = (props.get("naip:state") or "").upper()
        if props.get("sar:instrument_mode"):
            row["mode"] = props.get("sar:instrument_mode")
            row["polarization"] = "+".join(props.get("sar:polarizations") or [])
            row["orbit"] = (props.get("sat:orbit_state") or "").capitalize()
        rows.append(row)
    df = pd.DataFrame(rows)

k1, k2, k3, k4 = st.columns(4)
if ss.results_source == "usgs":
    k1.metric("Projects" if "work unit" in df.columns else "Tiles", len(df))
else:
    k1.metric("Datasets" if ss.results_source == "noaa" else "Scenes", len(df))
if ss.results_source == "usgs":
    if "size MB" in df.columns and df["size MB"].notna().any():
        k2.metric("Listed size", f"{df['size MB'].sum():.0f} MB")
    if "collected" in df.columns and df["collected"].astype(str).str.strip().any():
        k4.metric("Collected", f"{df['collected'].min()} → {df['collected'].max()}")
    elif "published" in df.columns and df["published"].astype(str).str.strip().any():
        k4.metric("Published", f"{df['published'].min()} → {df['published'].max()}")
    if ss.results and (ss.results[0].get("properties") or {}).get("truncated"):
        st.caption("More lidar meets this area than the list shows. Raise the max to see it.")
elif ss.results_source == "noaa":
    finest = df["resolution m"].dropna()
    if not finest.empty:
        k2.metric("Finest resolution", f"{finest.min():g} m")
    years = df["year"].dropna()
    if not years.empty:
        k4.metric("Years", f"{int(years.min())} → {int(years.max())}")
elif "cloud %" in df.columns and df["cloud %"].notna().any():
    k2.metric("Median cloud", f"{df['cloud %'].median():.1f}%")
    k3.metric("Clearest", f"{df['cloud %'].min():.1f}%")
    k4.metric("Date span", f"{df['date'].min()[:10]} → {df['date'].max()[:10]}")
else:
    k4.metric("Date span", f"{df['date'].min()[:10]} → {df['date'].max()[:10]}")

left, right = st.columns([3, 2])
with left:
    if ss.results_source == "usgs" and "work unit" in df.columns:
        st.markdown("#### Projects")
    elif ss.results_source == "usgs":
        st.markdown("#### Tiles")
    else:
        st.markdown("#### Datasets" if ss.results_source == "noaa" else "#### Scenes")
    cloud_config = {}
    if "cloud %" in df.columns:
        cloud_config["cloud %"] = st.column_config.ProgressColumn(
            "cloud %", min_value=0, max_value=100, format="%.1f")
    edited = st.data_editor(df, hide_index=True, width="stretch", height=330,
                            disabled=[c for c in df.columns if c != "download"],
                            column_config=cloud_config)
with right:
    if ss.results_source == "usgs" and "size MB" in df.columns:
        st.markdown("#### Tile size")
        fig = px.bar(df, x="published", y="size MB", hover_name="tile", height=330)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    elif ss.results_source == "usgs":
        st.markdown("#### Quality levels")
        counts = df["quality"].fillna("Unknown").value_counts().reset_index()
        counts.columns = ["quality", "projects"]
        fig = px.bar(counts, x="quality", y="projects", height=330)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    elif ss.results_source == "noaa":
        st.markdown("#### Tiles in the area")
        chart = df.assign(year=df["year"].map(lambda y: str(int(y)) if pd.notna(y) else ""))
        fig = px.bar(chart, x="year", y="tiles", hover_name="dataset", height=330)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    elif "cloud %" in df.columns and df["cloud %"].notna().any():
        st.markdown("#### Cloud cover over time")
        fig = px.bar(df, x="date", y="cloud %", height=330,
                     color="cloud %", color_continuous_scale="Blues_r",
                     range_color=(0, 100))
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False,
                          xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    else:
        st.markdown("#### Acquisitions")
        fig = px.histogram(df, x="date", height=330)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None)
        st.plotly_chart(fig, width="stretch")

show_scene_browser(ss.results, ss.results_source, sd.AOI(
    tuple(shape(ss.aoi_geom).bounds), ss.aoi_geom))

selected = edited[edited["download"]]
if ss.results_source == "noaa":
    noun = "dataset(s)"
elif ss.results_source == "usgs" and "work unit" in df.columns:
    noun = "project(s)"
elif ss.results_source == "usgs":
    noun = "tile(s)"
else:
    noun = "scene(s)"
st.markdown(f"**{len(selected)} {noun} selected**")


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
if st.button("⬇️ Download selected", type="primary", disabled=selected.empty):
    run_dir = OUT_ROOT / f"{collection}_{start}_{end}"
    run_dir.mkdir(parents=True, exist_ok=True)
    geom = ss.aoi_geom
    aoi = sd.AOI(tuple(shape(geom).bounds), geom)
    (run_dir / "aoi.geojson").write_text(json.dumps(
        {"type": "Feature", "geometry": geom, "properties": {}}))

    if ss.results_source == "cdse":
        cdse = sd.CDSE()
        chosen = [p for p in ss.results if p["Name"] in set(selected["name"])]
        bar = st.progress(0.0, text="Downloading SAFE products…")
        for i, p in enumerate(chosen, 1):
            try:
                sd.retry(lambda p=p: cdse.download(p, run_dir), what=p["Name"])
            except Exception as e:  # noqa: BLE001
                st.error(f"{p['Name']}: {e}")
            bar.progress(i / len(chosen), text=f"{i}/{len(chosen)} products")
        ss.last_run = {"dir": str(run_dir), "cdse": True}
    elif ss.results_source == "noaa":
        chosen = [d for d in ss.results if d["id"] in set(selected["id"])]
        noaa = sd.NOAA()
        bar = st.progress(0.0, text="Exporting NOAA GeoTIFF clips…")
        notes = []
        for i, d in enumerate(chosen, 1):
            title = d["properties"].get("title") or d["id"]
            try:
                _path, note = sd.retry(lambda d=d: noaa.download(d, aoi, run_dir),
                                       what=title)
                if note:
                    notes.append(note)
            except Exception as e:  # noqa: BLE001
                st.error(f"{title}: {e}")
            bar.progress(i / len(chosen), text=f"{i}/{len(chosen)} datasets")
        ss.last_run = {"dir": str(run_dir), "cdse": False, "mode": "full",
                       "noaa": True, "notes": notes}
    elif ss.results_source == "usgs":
        chosen = [d for d in ss.results if d["id"] in set(selected["id"])]
        usgs = sd.USGS()
        bar = st.progress(0.0, text="Downloading USGS lidar…")
        notes = []
        for i, d in enumerate(chosen, 1):
            title = d["properties"].get("title") or d["id"]
            try:
                _paths, note = sd.retry(
                    lambda d=d: usgs.download(d, aoi, run_dir, int(limit)),
                    what=title)
                if note:
                    notes.append(note)
            except Exception as e:  # noqa: BLE001
                st.error(f"{title}: {e}")
            bar.progress(i / len(chosen), text=f"{i}/{len(chosen)} lidar downloads")
        ss.last_run = {"dir": str(run_dir), "cdse": False, "mode": "full",
                       "usgs": True, "notes": notes}
    elif ss.results and ss.results[0].get("collection") == "naip":
        chosen = [d for d in ss.results if d["id"] in set(selected["id"])]
        with st.spinner(f"Clipping {len(chosen)} NAIP tile(s)…"):
            failures = sd.naip_download(to_items(chosen), aoi, run_dir)
        if failures:
            st.warning(f"{len(failures)} tile(s) failed")
            st.dataframe(pd.DataFrame(failures, columns=["scene", "error"]))
        ss.last_run = {"dir": str(run_dir), "cdse": False,
                       "ids": [d["id"] for d in chosen], "mode": "clip",
                       "band_map": {b: b for b in ("red", "green", "blue", "nir")},
                       "is_s1": False}
    else:
        chosen = [d for d in ss.results if d["id"] in set(selected["id"])]
        stack_bands = ([b for b in ("red", "green", "blue", "nir") if b in bands]
                       if stack and mode == "clip" and not is_s1 else None)
        with st.spinner(f"Fetching {len(chosen) * len(bands)} band file(s)…"):
            failures = sd.stac_download(to_items(chosen), source, collection, aoi,
                                        bands, mode, run_dir, stack_bands or None, 6)
        if failures:
            st.warning(f"{len(failures)} band file(s) failed")
            st.dataframe(pd.DataFrame(failures, columns=["scene", "band", "error"]))
        ss.last_run = {"dir": str(run_dir), "cdse": False,
                       "ids": [d["id"] for d in chosen], "mode": mode,
                       "band_map": sd.resolve_bands(source, collection, bands),
                       "is_s1": is_s1}
    st.rerun()


# --------------------------------------------------------------------------- #
# Results of last download
# --------------------------------------------------------------------------- #
run = ss.last_run
if run:
    run_dir = Path(run["dir"])
    st.divider()
    st.markdown(f"### Downloaded → `{run_dir.resolve()}`")
    for note in run.get("notes") or []:
        st.info(note)

    if not run["cdse"] and run["mode"] == "clip":
        band_map = run["band_map"]
        ids = [i for i in run["ids"] if (run_dir / sd.safe_name(i)).exists()]
        if ids:
            pick = st.selectbox("Inspect scene", ids, key="inspect")
            idir = run_dir / sd.safe_name(pick)
            item_dict = json.loads((idir / "item.json").read_text())
            if run["is_s1"]:
                pols = [b for b in band_map if (idir / f"{b}.tif").exists()]
                cols = st.columns(max(len(pols), 1))
                for col, b in zip(cols, pols):
                    db = s1_db(idir, b)
                    f = px.imshow(db, color_continuous_scale="gray",
                                  zmin=np.nanpercentile(db, 2),
                                  zmax=np.nanpercentile(db, 98),
                                  title=f"{b.upper()} backscatter (dB)")
                    f.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                    col.plotly_chart(f, width="stretch")
            else:
                c1, c2 = st.columns(2)
                if all((idir / f"{b}.tif").exists() for b in ("red", "green", "blue")):
                    c1.image(quicklook_rgb(idir, item_dict, band_map),
                             caption="True colour (2–98% stretch)",
                             width="stretch")
                if all((idir / f"{b}.tif").exists() for b in ("red", "nir")):
                    nd = ndvi(idir, item_dict, band_map)
                    if nd is not None:
                        f = px.imshow(nd, color_continuous_scale="RdYlGn",
                                      zmin=-0.2, zmax=0.9, title="NDVI")
                        f.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                        f.update_xaxes(visible=False)
                        f.update_yaxes(visible=False)
                        c2.plotly_chart(f, width="stretch")
                        v = nd[np.isfinite(nd)]
                        if v.size:
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Mean NDVI", f"{v.mean():.3f}")
                            m2.metric("Median NDVI", f"{np.median(v):.3f}")
                            m3.metric("NDVI > 0.5", f"{(v > 0.5).mean() * 100:.0f}%")

            # NDVI time series across downloaded scenes
            if not run["is_s1"] and len(ids) > 1:
                rows = []
                for i in ids:
                    d_ = run_dir / sd.safe_name(i)
                    if (d_ / "red.tif").exists() and (d_ / "nir.tif").exists():
                        it = json.loads((d_ / "item.json").read_text())
                        nd = ndvi(d_, it, band_map)
                        props = it.get("properties") or {}
                        when = props.get("datetime") or props.get("start_datetime") or ""
                        if nd is not None and np.isfinite(nd).any() and when:
                            rows.append({"date": str(when)[:10],
                                         "mean NDVI": float(np.nanmean(nd))})
                if rows:
                    f = px.line(pd.DataFrame(rows).sort_values("date"), x="date",
                                y="mean NDVI", markers=True, height=280,
                                title="Mean NDVI over AOI")
                    f.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(f, width="stretch")
                    st.caption("Uses all pixels; cloud masking with SCL is not applied.")

    files = [f for f in run_dir.rglob("*") if f.is_file()]
    size_mb = sum(f.stat().st_size for f in files) / 1e6
    st.caption(f"{len(files)} files, {size_mb:.1f} MB")
    if size_mb < 500:
        st.download_button("📦 Download results as .zip", zip_folder(run_dir),
                           file_name=f"{run_dir.name}.zip", mime="application/zip")
    else:
        st.caption("Too large to zip in the browser; open the folder above instead.")
    if st.button("🗑️ Delete this run"):
        shutil.rmtree(run_dir, ignore_errors=True)
        ss.last_run = None
        st.rerun()
