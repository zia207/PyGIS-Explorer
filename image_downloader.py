#!/usr/bin/env python3
"""
image_downloader.py — search and download Sentinel, Landsat, and NOAA coastal imagery.

Sources
-------
  earth-search   Element 84 Earth Search on AWS (STAC, COGs, no account)
                 collections: sentinel-2-l2a, sentinel-2-l1c, landsat-c2-l2, naip
                 naip is the newest USDA NAIP year covering the area
                 (4-band red, green, blue, near infrared)
  planetary      Microsoft Planetary Computer (STAC, no account)
                 collections: sentinel-2-l2a, sentinel-1-rtc, sentinel-1-grd,
                 landsat-c2-l2, MODIS Collection 6.1 (GPP, LAI/FPAR,
                 250 m reflectance, 250 m vegetation indices), HLS v2.0,
                 sentinel-3-* netcdf, sentinel-5p-l2-netcdf,
                 JRC Global Surface Water
  cdse           Copernicus Data Space Ecosystem (OData, full products,
                 free account; set CDSE_USER and CDSE_PASSWORD env variables)
                 collections: SENTINEL-1, SENTINEL-2, SENTINEL-3, SENTINEL-5P,
                 LANDSAT-8, LANDSAT-9  (+ --product-type)
  noaa           NOAA Digital Coast imagery (ImageServer, no account)
                 https://coast.noaa.gov/dataviewer/#/imagery/search
                 collections: noaa-rgb8, noaa-rgb16, noaa-rgbn8, noaa-cir8, noaa-naip
                 noaa-naip is the newest USDA NAIP dataset in the coastal mosaic
                 The start and end dates are used as a year range. Download
                 writes a GeoTIFF of the area of interest.
  usgs           USGS 3DEP lidar (no account), the catalogue behind
                 https://apps.nationalmap.gov/lidar-explorer/#/
                 collections: lidar-lpc (LAZ tiles), lidar-projects (work units)
                 Dates filter tile publication or project collection dates.

Landsat Collection 2 Level-2 searches are limited to Landsat 8 and Landsat 9.
Sentinel-5 (UVNS on MetOp-SG) is not in these catalogues; use Sentinel-5P.

Modes (STAC sources)
--------------------
  clip   read only the AOI window from each COG band (fast, small files)  [default]
  full   download the whole band file for each scene

Examples
--------
  # Search Sentinel-2 L2A over Ithaca, summer 2026, <10% cloud
  python image_downloader.py search --bbox -76.6 42.4 -76.4 42.5 \
      --start 2026-06-01 --end 2026-09-30 --cloud 10

  # Clip RGB+NIR+SCL to the AOI and also write a 4-band stack at 10 m
  python image_downloader.py download --bbox -76.6 42.4 -76.4 42.5 \
      --start 2026-06-01 --end 2026-09-30 --cloud 10 \
      --bands red green blue nir scl --stack red green blue nir --out s2_ithaca

  # Sentinel-1 RTC (VV/VH) from Planetary Computer, AOI from a GeoJSON
  python image_downloader.py download --source planetary \
      --collection sentinel-1-rtc --aoi field.geojson \
      --start 2026-07-01 --end 2026-07-31 --bands vv vh --out s1_rtc

  # Landsat 8/9 Collection 2 Level-2, clipped to the AOI
  python image_downloader.py download --collection landsat-c2-l2 \
      --bbox -76.6 42.4 -76.4 42.5 --start 2026-06-01 --end 2026-09-30 \
      --cloud 20 --bands red green blue nir --out ls89_ithaca

  # USGS 3DEP lidar point cloud tiles (LAZ) around Charleston
  python image_downloader.py search --source usgs --collection lidar-lpc \
      --bbox -79.95 32.75 -79.85 32.82 --start 2010-01-01 --end 2026-10-15 --limit 5

  # Full Sentinel-2 L2A SAFE products from Copernicus
  export CDSE_USER=you@example.com CDSE_PASSWORD=...
  python image_downloader.py download --source cdse --collection SENTINEL-2 \
      --product-type S2MSI2A --bbox -76.6 42.4 -76.4 42.5 \
      --start 2026-07-01 --end 2026-07-15 --cloud 20 --limit 2 --out safe

Install
-------
  pip install pystac-client rasterio requests shapely tqdm
  pip install planetary-computer      # only for --source planetary
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date, datetime, timezone
from xml.etree import ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import requests

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it=None, **kw):
        return it if it is not None else _NullBar()

    class _NullBar:
        def update(self, *_): ...
        def close(self): ...


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
STAC_URLS = {
    "earth-search": "https://earth-search.aws.element84.com/v1",
    "planetary": "https://planetarycomputer.microsoft.com/api/stac/v1",
}

DEFAULT_COLLECTION = {
    "earth-search": "sentinel-2-l2a",
    "planetary": "sentinel-2-l2a",
    "cdse": "SENTINEL-2",
    "noaa": "noaa-rgb8",
    "usgs": "lidar-lpc",
}

# USGS 3DEP lidar. The index is the same service Lidar Explorer draws, and the
# tiles are the LAZ files the National Map download API publishes.
# https://apps.nationalmap.gov/lidar-explorer/#/
LIDAR_EXPLORER = "https://apps.nationalmap.gov/lidar-explorer/#/"
LIDAR_INDEX = ("https://index.nationalmap.gov/arcgis/rest/services/"
               "3DEPElevationIndex/MapServer/24/query")
TNM_PRODUCTS = "https://tnmaccess.nationalmap.gov/api/v1/products"
LIDAR_COLLECTIONS = ("lidar-lpc", "lidar-projects")

# NOAA Digital Coast mosaics served by the Data Access Viewer imagery services.
# https://coast.noaa.gov/dataviewer/#/imagery/search
NOAA_IMAGE = "https://coast.noaa.gov/arcgisimg/rest/services/Imagery"
NOAA_PRODUCTS = {
    "noaa-rgb8": "3Band_RGB_8Bit_Imagery",
    "noaa-rgb16": "3Band_RGB_16Bit_Imagery",
    "noaa-rgbn8": "4Band_RGBN_8Bit_Imagery",
    "noaa-cir8": "3Band_CIR_8Bit_Imagery",
    "noaa-naip": "4Band_RGBN_8Bit_Imagery",
}
# Public copy of the same USDA NAIP COGs catalogued by Earth Search.
# The naip-analytic S3 bucket is requester-pays, so clips are read from this host.
NAIP_BLOB = "https://naipeuwest.blob.core.windows.net/naip/v002"
NAIP_BANDS = ("red", "green", "blue", "nir")
# Postal codes checked for a year folder under the public NAIP catalogue.
NAIP_STATE_CODES = (
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "dc", "fl", "ga", "hi",
    "id", "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn",
    "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh",
    "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa",
    "wv", "wi", "wy", "pr", "vi",
)
# exportImage rejects requests above these pixel dimensions.
NOAA_MAX_WIDTH = 4000
NOAA_MAX_HEIGHT = 4000

# Common band names -> asset keys, per source. Unknown names are passed through,
# so any native asset key (e.g. "B8A", "rededge2", "visual") also works.
S2_ALIASES = {
    "earth-search": {
        "coastal": "coastal", "blue": "blue", "green": "green", "red": "red",
        "rededge1": "rededge1", "rededge2": "rededge2", "rededge3": "rededge3",
        "nir": "nir", "nir08": "nir08", "nir09": "nir09",
        "swir16": "swir16", "swir22": "swir22", "scl": "scl", "visual": "visual",
    },
    "planetary": {
        "coastal": "B01", "blue": "B02", "green": "B03", "red": "B04",
        "rededge1": "B05", "rededge2": "B06", "rededge3": "B07",
        "nir": "B08", "nir08": "B8A", "nir09": "B09",
        "swir16": "B11", "swir22": "B12", "scl": "SCL", "visual": "visual",
    },
}

# Landsat Collection 2 Level-2 asset keys are the same on Earth Search and
# Planetary Computer. "nir" is the common alias for the 30 m NIR band (B5).
LS_ALIASES = {
    "coastal": "coastal", "blue": "blue", "green": "green", "red": "red",
    "nir": "nir08", "nir08": "nir08", "swir16": "swir16", "swir22": "swir22",
    "lwir": "lwir11", "lwir11": "lwir11", "qa": "qa_pixel", "qa_pixel": "qa_pixel",
    "qa_aerosol": "qa_aerosol", "qa_radsat": "qa_radsat",
}

# Planetary Computer collections whose asset keys are not the band names
# shown in the app. Values are the STAC asset keys.
PLANETARY_BANDS = {
    "modis-17A2H-061": {
        "gpp": "Gpp_500m",
        "net_photosynthesis": "PsnNet_500m",
        "qc": "Psn_QC_500m",
    },
    "modis-15A2H-061": {
        "lai": "Lai_500m",
        "fpar": "Fpar_500m",
        "lai_sd": "LaiStdDev_500m",
        "fpar_sd": "FparStdDev_500m",
        "qc": "FparLai_QC",
    },
    "modis-09Q1-061": {
        "red": "sur_refl_b01",
        "nir": "sur_refl_b02",
        "qc": "sur_refl_qc_250m",
        "state": "sur_refl_state_250m",
    },
    "modis-13Q1-061": {
        "ndvi": "250m_16_days_NDVI",
        "evi": "250m_16_days_EVI",
        "red": "250m_16_days_red_reflectance",
        "nir": "250m_16_days_NIR_reflectance",
        "blue": "250m_16_days_blue_reflectance",
        "mir": "250m_16_days_MIR_reflectance",
        "quality": "250m_16_days_VI_Quality",
        "reliability": "250m_16_days_pixel_reliability",
    },
    "hls2-s30": {
        "coastal": "B01", "blue": "B02", "green": "B03", "red": "B04",
        "rededge1": "B05", "rededge2": "B06", "rededge3": "B07",
        "nir": "B08", "nir08": "B8A", "swir16": "B11", "swir22": "B12",
        "fmask": "Fmask",
    },
    "hls2-l30": {
        "coastal": "B01", "blue": "B02", "green": "B03", "red": "B04",
        "nir": "B05", "swir16": "B06", "swir22": "B07",
        "cirrus": "B09", "lwir": "B10", "lwir2": "B11", "fmask": "Fmask",
    },
    "jrc-gsw": {
        "occurrence": "occurrence",
        "change": "change",
        "seasonality": "seasonality",
        "recurrence": "recurrence",
        "transitions": "transitions",
        "extent": "extent",
    },
}
PLANETARY_DEFAULT_BANDS = {
    "modis-17A2H-061": ["gpp", "net_photosynthesis"],
    "modis-15A2H-061": ["lai", "fpar"],
    "modis-09Q1-061": ["red", "nir"],
    "modis-13Q1-061": ["ndvi", "evi", "red", "nir"],
    "hls2-s30": ["red", "green", "blue", "nir"],
    "hls2-l30": ["red", "green", "blue", "nir"],
    "jrc-gsw": ["occurrence", "extent"],
}

CDSE_TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
                  "protocol/openid-connect/token")
CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products({id})/$value"


# --------------------------------------------------------------------------- #
# AOI handling
# --------------------------------------------------------------------------- #
@dataclass
class AOI:
    bbox: tuple[float, float, float, float]          # lon/lat, EPSG:4326
    geometry: dict = field(default_factory=dict)     # GeoJSON geometry

    @classmethod
    def from_bbox(cls, bbox):
        minx, miny, maxx, maxy = map(float, bbox)
        if minx >= maxx or miny >= maxy:
            sys.exit("Invalid --bbox: expected MINLON MINLAT MAXLON MAXLAT")
        geom = {"type": "Polygon", "coordinates": [[
            [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]}
        return cls((minx, miny, maxx, maxy), geom)

    @classmethod
    def from_geojson(cls, path):
        from shapely.geometry import mapping, shape
        from shapely.ops import unary_union

        data = json.loads(Path(path).read_text())
        if data.get("type") == "FeatureCollection":
            geoms = [shape(f["geometry"]) for f in data["features"]]
        elif data.get("type") == "Feature":
            geoms = [shape(data["geometry"])]
        else:
            geoms = [shape(data)]
        union = unary_union(geoms)
        return cls(tuple(union.bounds), mapping(union))

    def wkt(self) -> str:
        from shapely.geometry import shape
        return shape(self.geometry).wkt


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def retry(fn, tries=4, delay=3, what="request"):
    """Call fn() with simple exponential back-off."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt == tries:
                raise
            wait = delay * 2 ** (attempt - 1)
            print(f"  ! {what} failed ({exc}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


# --------------------------------------------------------------------------- #
# STAC sources (Earth Search / Planetary Computer)
# --------------------------------------------------------------------------- #
def open_stac(source: str):
    from pystac_client import Client

    modifier = None
    if source == "planetary":
        try:
            import planetary_computer
        except ImportError:
            sys.exit("--source planetary needs:  pip install planetary-computer")
        modifier = planetary_computer.sign_inplace
    return Client.open(STAC_URLS[source], modifier=modifier)


def _rfc3339_range(start, end) -> str:
    """Earth Search rejects date-only intervals; other APIs accept full stamps."""
    def one(value, end_of_day):
        text = str(value)
        if "T" in text:
            return text
        return f"{text}T23:59:59Z" if end_of_day else f"{text}T00:00:00Z"
    return f"{one(start, False)}/{one(end, True)}"


def _supports_cloud(collection: str) -> bool:
    return (collection.startswith("sentinel-2") or collection == "landsat-c2-l2"
            or collection.startswith("hls2-"))


def stac_search(source, collection, aoi: AOI, start, end, cloud, limit,
                platforms=None, extra_query=None):
    client = open_stac(source)
    params = dict(
        collections=[collection],
        intersects=aoi.geometry,
        datetime=_rfc3339_range(start, end),
        max_items=limit,
        sortby=[{"field": "properties.datetime", "direction": "asc"}],
    )
    # JRC Global Surface Water is one 1984–2020 mosaic per tile. A recent
    # date window does not overlap it.
    if collection == "jrc-gsw":
        params.pop("datetime")
    query = {}
    if cloud is not None and _supports_cloud(collection):
        query["eo:cloud_cover"] = {"lte": cloud}
    if platforms:
        query["platform"] = {"in": list(platforms)}
    elif collection == "landsat-c2-l2":
        query["platform"] = {"in": ["landsat-8", "landsat-9"]}
    if extra_query:
        query.update(extra_query)
    if query:
        params["query"] = query
    try:
        items = list(client.search(**params).items())
    except Exception:  # some APIs reject sortby; retry without it
        params.pop("sortby")
        items = list(client.search(**params).items())
    for item in items:
        # Composites publish a range. pystac writes properties.datetime from
        # item.datetime, so copy the interval start onto the item itself.
        if item.datetime is None and (item.properties or {}).get("start_datetime"):
            item.datetime = datetime.fromisoformat(
                item.properties["start_datetime"].replace("Z", "+00:00"))
    return sorted(items, key=lambda i: i.datetime.isoformat() if i.datetime else "")


def naip_search(aoi: AOI, limit):
    """Newest USDA NAIP year for each state that meets the AOI."""
    client = open_stac("earth-search")
    params = dict(
        collections=["naip"],
        intersects=aoi.geometry,
        datetime=_rfc3339_range("2010-01-01", date.today().isoformat()),
        max_items=max(int(limit) * 8, 80),
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
    )
    try:
        items = list(client.search(**params).items())
    except Exception:
        params.pop("sortby")
        items = list(client.search(**params).items())

    def year(item):
        raw = item.properties.get("naip:year")
        if raw is None and item.datetime:
            return item.datetime.year
        return int(raw or 0)

    by_state = {}
    for item in items:
        state = (item.properties.get("naip:state") or "").lower()
        by_state.setdefault(state, []).append(item)
    kept = []
    for group in by_state.values():
        newest = max(year(item) for item in group)
        kept.extend(item for item in group if year(item) == newest)
    kept.sort(key=year, reverse=True)
    return kept[: int(limit)]


def _naip_year_prefixes(code: str) -> list[int]:
    """Years that have a folder in the public NAIP catalogue for one state."""
    prefix = f"v002/{code}/"
    years, marker, pages = [], None, 0
    while pages < 20:
        params = {
            "restype": "container",
            "comp": "list",
            "prefix": prefix,
            "delimiter": "/",
            "maxresults": "1000",
        }
        if marker:
            params["marker"] = marker
        r = requests.get("https://naipeuwest.blob.core.windows.net/naip",
                         params=params, timeout=40)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for el in root.iter("Name"):
            part = (el.text or "").strip("/").split("/")[-1]
            if part.isdigit() and len(part) == 4:
                years.append(int(part))
        marker = next((el.text for el in root.iter("NextMarker") if el.text), None)
        pages += 1
        if not marker:
            break
    return sorted(set(years))


def naip_coverage() -> list[dict]:
    """USDA NAIP years available for each state and territory in the catalogue."""
    def one(code: str):
        try:
            return code, _naip_year_prefixes(code), None
        except Exception as exc:  # noqa: BLE001
            return code, [], str(exc)

    found, errors = [], []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for code, years, err in pool.map(one, NAIP_STATE_CODES):
            if err:
                errors.append(err)
            if years:
                found.append({
                    "code": code.upper(),
                    "years": years,
                    "latest": years[-1],
                    "earliest": years[0],
                })
    if not found and errors:
        raise RuntimeError(errors[0])
    found.sort(key=lambda row: row["code"])
    return found


def clip_naip(href: str, aoi: AOI, idir: Path):
    """Clip one NAIP COG into red, green, blue, and near-infrared GeoTIFFs."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    url = naip_cog_url(href)
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
               GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")
    with rasterio.Env(**env), rasterio.open(url) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, *aoi.bbox, densify_pts=21)
        win = from_bounds(*bounds, transform=src.transform)
        c0, r0 = math.floor(win.col_off), math.floor(win.row_off)
        c1 = math.ceil(win.col_off + win.width)
        r1 = math.ceil(win.row_off + win.height)
        try:
            win = Window(c0, r0, c1 - c0, r1 - r0).intersection(
                Window(0, 0, src.width, src.height))
        except Exception:
            raise RuntimeError("AOI does not overlap this NAIP tile") from None
        if win.width <= 0 or win.height <= 0:
            raise RuntimeError("AOI does not overlap this NAIP tile")
        data = src.read(window=win)
        transform = src.window_transform(win)
        crs = src.crs
    names = list(NAIP_BANDS[: data.shape[0]])
    profile = dict(driver="GTiff", height=data.shape[1], width=data.shape[2],
                   count=1, dtype=data.dtype, crs=crs, transform=transform,
                   nodata=0, compress="deflate")
    if data.shape[1] >= 256 and data.shape[2] >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    for index, name in enumerate(names):
        dst = idir / f"{name}.tif"
        with rasterio.open(dst, "w", **profile) as out:
            out.write(data[index], 1)
    stack_profile = dict(profile)
    stack_profile["count"] = len(names)
    with rasterio.open(idir / "stack.tif", "w", **stack_profile) as out:
        out.write(data[: len(names)])
        out.descriptions = tuple(names)
    return names


def naip_download(items, aoi: AOI, out: Path):
    """Clip the latest NAIP tiles. Returns a list of (id, error) failures."""
    out.mkdir(parents=True, exist_ok=True)
    failures = []
    for item in items:
        idir = out / safe_name(item.id)
        idir.mkdir(exist_ok=True)
        (idir / "item.json").write_text(json.dumps(item.to_dict(), indent=1))
        asset = item.assets.get("image")
        if asset is None:
            failures.append((item.id, "no image asset"))
            continue
        try:
            retry(lambda asset=asset: clip_naip(asset.href, aoi, idir),
                  what=item.id)
        except Exception as exc:  # noqa: BLE001
            failures.append((item.id, str(exc)))
    return failures


def resolve_bands(source, collection, bands):
    if collection.startswith("sentinel-2"):
        table = S2_ALIASES.get(source, {})
        return {b: table.get(b.lower(), b) for b in bands}
    if collection == "landsat-c2-l2":
        return {b: LS_ALIASES.get(b.lower(), b) for b in bands}
    if collection.startswith("sentinel-1"):
        return {b: b.lower() for b in bands}  # vv, vh, hh, hv
    if collection in PLANETARY_BANDS:
        table = PLANETARY_BANDS[collection]
        return {b: table.get(b, b) for b in bands}
    return {b: b for b in bands}  # native asset keys (Sentinel-3, Sentinel-5P)


def clip_cog(href: str, aoi: AOI, dst: Path):
    """Read only the AOI window of a (cloud-optimized) GeoTIFF and save it."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
               CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.TIF,.jp2",
               GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
               AWS_NO_SIGN_REQUEST="YES")
    with rasterio.Env(**env), rasterio.open(href) as src:
        if src.crs is None:
            raise RuntimeError("asset has no CRS (e.g. raw S1 GRD); use sentinel-1-rtc "
                               "or --mode full")
        bounds = transform_bounds("EPSG:4326", src.crs, *aoi.bbox, densify_pts=21)
        win = from_bounds(*bounds, transform=src.transform)
        # snap outward to whole pixels (works across rasterio versions)
        import math
        c0, r0 = math.floor(win.col_off), math.floor(win.row_off)
        c1 = math.ceil(win.col_off + win.width)
        r1 = math.ceil(win.row_off + win.height)
        try:
            win = Window(c0, r0, c1 - c0, r1 - r0).intersection(
                Window(0, 0, src.width, src.height))
        except Exception:  # rasterio raises WindowError when there is no overlap
            raise RuntimeError("AOI does not overlap this asset") from None
        if win.width <= 0 or win.height <= 0:
            raise RuntimeError("AOI does not overlap this asset")
        data = src.read(window=win)
        profile = src.profile.copy()
        profile.update(driver="GTiff", height=data.shape[1], width=data.shape[2],
                       transform=src.window_transform(win), compress="deflate",
                       tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
        for key in ("photometric",):
            profile.pop(key, None)
        tmp = dst.with_suffix(".part.tif")
        with rasterio.open(tmp, "w", **profile) as out:
            out.write(data)
            out.update_tags(**src.tags())
        tmp.replace(dst)


def asset_suffix(href: str) -> str:
    path = href.split("?", 1)[0].lower()
    for ext in (".tiff", ".tif", ".jp2", ".nc4", ".nc", ".xml", ".json", ".txt"):
        if path.endswith(ext):
            return ".tif" if ext == ".tiff" else ext
    return ".bin"


_BROWSER_MEDIA = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


def _browser_href(asset: dict) -> str | None:
    """HTTP JPEG or PNG a browser can show. COGs and requester-pays keys are skipped."""
    if not isinstance(asset, dict):
        return None
    href = asset.get("href") or ""
    media = (asset.get("type") or "").split(";")[0].strip().lower()
    if media.startswith("image/") and media not in _BROWSER_MEDIA:
        return None
    path = href.split("?", 1)[0].lower()
    if path.endswith((".tif", ".tiff", ".jp2", ".nc", ".xml")):
        return None
    if href.startswith("s3://"):
        https = to_https(href)
        if https.startswith("https://sentinel-cogs."):
            return https
        return None
    if href.startswith("http"):
        return href
    return None


def browse_href(item: dict) -> str | None:
    """Public thumbnail URL for a STAC item, when the catalogue publishes one."""
    assets = item.get("assets") or {}
    for key in ("thumbnail", "browse", "rendered_preview", "reduced_resolution_browse", "preview"):
        href = _browser_href(assets.get(key) or {})
        if href:
            return href
    # Some catalogues keep the JPEG on a requester-pays bucket and proxy it
    # from the item thumbnail link. That link 404s when no thumbnail asset exists.
    if any(key in assets for key in ("thumbnail", "rendered_preview", "reduced_resolution_browse")):
        for link in item.get("links") or []:
            if link.get("rel") == "thumbnail":
                href = _browser_href({"href": link.get("href"), "type": "image/jpeg"})
                if href:
                    return href
    return None


def naip_image_href(item: dict) -> str | None:
    href = ((item.get("assets") or {}).get("image") or {}).get("href") or ""
    if href.startswith("s3://naip-analytic/") or "/naip/" in href:
        return href
    return None


def naip_preview(href: str, max_px: int = 512):
    """Small true-colour view of one NAIP tile, read from the public COG."""
    import numpy as np
    import rasterio

    url = naip_cog_url(href)
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
               GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="1")
    with rasterio.Env(**env), rasterio.open(url) as src:
        scale = max(src.width, src.height) / float(max_px)
        height = max(1, int(round(src.height / scale)))
        width = max(1, int(round(src.width / scale)))
        data = src.read(indexes=[1, 2, 3], out_shape=(3, height, width))
    rgb = np.moveaxis(data, 0, -1).astype("float32")
    out = np.zeros_like(rgb)
    for band in range(3):
        values = rgb[:, :, band]
        valid = values[values > 0]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, (2, 98))
        out[:, :, band] = np.clip((values - lo) / (hi - lo + 1e-6), 0, 1)
    return (out * 255).astype("uint8")


def naip_previews(hrefs) -> dict:
    """Thumbnails for several NAIP tiles. Failed tiles map to None."""
    unique = list(dict.fromkeys(h for h in hrefs if h))

    def one(href):
        try:
            return href, naip_preview(href)
        except Exception:
            return href, None

    found = {}
    if unique:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for href, image in pool.map(one, unique):
                found[href] = image
    return found


def naip_cog_url(href: str) -> str:
    """Map an Earth Search NAIP S3 key to the public COG of that same tile."""
    prefix = "s3://naip-analytic/"
    if href.startswith("http"):
        return href
    if not href.startswith(prefix):
        raise ValueError(f"Not an Earth Search NAIP asset: {href}")
    state, year, res, _kind, quad, name = href[len(prefix):].split("/", 5)
    padded = f"{int(''.join(ch for ch in res if ch.isdigit())):03d}cm"
    return f"{NAIP_BLOB}/{state}/{year}/{state}_{padded}_{year}/{quad}/{name}"


def to_https(href: str) -> str:
    if href.startswith("s3://naip-analytic/"):
        return naip_cog_url(href)
    rewrites = (
        ("s3://sentinel-cogs/", "https://sentinel-cogs.s3.us-west-2.amazonaws.com/"),
        ("s3://usgs-landsat/", "https://usgs-landsat.s3.us-west-2.amazonaws.com/"),
    )
    for prefix, base in rewrites:
        if href.startswith(prefix):
            return base + href[len(prefix):]
    return href


def http_download(url: str, dst: Path, session=None, headers=None, desc=None,
                  timeout=120):
    """Stream a URL to disk with resume support."""
    session = session or requests.Session()
    tmp = dst.with_name(dst.name + ".part")
    pos = tmp.stat().st_size if tmp.exists() else 0
    hdrs = dict(headers or {})
    if pos:
        hdrs["Range"] = f"bytes={pos}-"
    with session.get(url, headers=hdrs, stream=True, timeout=timeout) as r:
        if r.status_code == 416:  # already complete
            tmp.replace(dst)
            return
        r.raise_for_status()
        if pos and r.status_code != 206:  # server ignored Range; start over
            pos = 0
        total = int(r.headers.get("Content-Length", 0)) + pos
        bar = tqdm(total=total or None, initial=pos, unit="B", unit_scale=True,
                   desc=desc or dst.name, leave=False)
        with open(tmp, "ab" if pos else "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        bar.close()
    tmp.replace(dst)


def stack_bands(files: list[Path], dst: Path, names: list[str]):
    """Resample bands to the grid of the first (finest-listed) band and stack."""
    import numpy as np
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(files[0]) as ref:
        profile = ref.profile.copy()
        h, w, transform, crs = ref.height, ref.width, ref.transform, ref.crs
    dtype = np.result_type(*[rasterio.open(f).dtypes[0] for f in files])
    profile.update(count=len(files), dtype=dtype, compress="deflate")
    with rasterio.open(dst, "w", **profile) as out:
        for i, (f, name) in enumerate(zip(files, names), start=1):
            with rasterio.open(f) as src:
                arr = np.zeros((h, w), dtype=dtype)
                is_class = name.lower() in ("scl", "qa", "mask") or name.lower().startswith("qa")
                reproject(rasterio.band(src, 1), arr, dst_transform=transform,
                          dst_crs=crs, resampling=Resampling.nearest if is_class
                          else Resampling.bilinear)
                out.write(arr, i)
                out.set_band_description(i, name)


def stac_download(items, source, collection, aoi, bands, mode, out, stack, workers):
    band_map = resolve_bands(source, collection, bands)
    out.mkdir(parents=True, exist_ok=True)
    jobs, skipped = [], 0

    for item in items:
        idir = out / safe_name(item.id)
        idir.mkdir(exist_ok=True)
        (idir / "item.json").write_text(json.dumps(item.to_dict(), indent=1))
        for name, key in band_map.items():
            asset = item.assets.get(key)
            if asset is None:
                print(f"  - {item.id}: no asset '{key}' "
                      f"(available: {', '.join(sorted(item.assets))})", file=sys.stderr)
                continue
            dst = idir / f"{name}{asset_suffix(asset.href)}"
            if dst.exists():
                skipped += 1
                continue
            jobs.append((item.id, name, asset.href, dst))

    print(f"{len(items)} scene(s), {len(jobs)} band file(s) to fetch"
          + (f", {skipped} already present" if skipped else ""))
    if mode == "clip" and any(asset_suffix(href) not in {".tif", ".jp2"}
                              for _, _, href, _ in jobs):
        print("  NetCDF and other non-COG assets are downloaded whole.",
              file=sys.stderr)

    def run(job):
        item_id, name, href, dst = job
        if mode == "clip" and asset_suffix(href) in {".tif", ".jp2"}:
            retry(lambda: clip_cog(href, aoi, dst), what=f"{item_id}/{name}")
        else:
            url = to_https(href)
            retry(lambda: http_download(url, dst, desc=f"{item_id[:24]}/{name}"),
                  what=f"{item_id}/{name}")
        return job

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, j): j for j in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="bands"):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                job = futures[fut]
                failures.append((job[0], job[1], str(exc)))

    if stack and mode == "clip":
        for item in items:
            idir = out / safe_name(item.id)
            files = [idir / f"{b}.tif" for b in stack]
            if all(f.exists() for f in files):
                stack_bands(files, idir / "stack.tif", stack)
    elif stack:
        print("  (--stack is only applied in clip mode)", file=sys.stderr)

    write_summary(out, items, failures)
    return failures


# --------------------------------------------------------------------------- #
# CDSE (full SAFE products via OData)
# --------------------------------------------------------------------------- #
class CDSE:
    def __init__(self):
        self.user = os.environ.get("CDSE_USER")
        self.password = os.environ.get("CDSE_PASSWORD")
        self.session = requests.Session()
        self._token, self._expires = None, 0.0

    def token(self):
        if not (self.user and self.password):
            sys.exit("Set CDSE_USER and CDSE_PASSWORD (register free at "
                     "https://dataspace.copernicus.eu)")
        if self._token and time.time() < self._expires - 60:
            return self._token
        r = requests.post(CDSE_TOKEN_URL, timeout=60, data={
            "client_id": "cdse-public", "grant_type": "password",
            "username": self.user, "password": self.password})
        if r.status_code != 200:
            sys.exit(f"CDSE login failed: {r.status_code} {r.text[:200]}")
        js = r.json()
        self._token = js["access_token"]
        self._expires = time.time() + js.get("expires_in", 600)
        return self._token

    def search(self, collection, aoi: AOI, start, end, cloud, product_type, limit,
               *, s1_mode=None, s1_polarisation=None, s1_orbit=None):
        filt = [
            f"Collection/Name eq '{collection.upper()}'",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi.wkt()}')",
            f"ContentDate/Start ge {start}T00:00:00.000Z",
            f"ContentDate/Start le {end}T23:59:59.999Z",
        ]

        def string_attr(name, value):
            return ("Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
                    f"'{name}' and att/OData.CSC.StringAttribute/Value eq '{value}')")

        if product_type:
            filt.append(string_attr("productType", product_type))
        if s1_mode:
            filt.append(string_attr("operationalMode", s1_mode))
        if s1_polarisation:
            filt.append(string_attr("polarisationChannels", s1_polarisation))
        if s1_orbit:
            filt.append(string_attr("orbitDirection", s1_orbit))
        if cloud is not None and (collection.upper() == "SENTINEL-2"
                                  or collection.upper().startswith("LANDSAT")):
            filt.append("Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq "
                        f"'cloudCover' and att/OData.CSC.DoubleAttribute/Value le "
                        f"{float(cloud):.2f})")
        # Repeated $expand keys: this catalogue rejects a comma-joined expand.
        params = [
            ("$filter", " and ".join(filt)),
            ("$orderby", "ContentDate/Start asc"),
            ("$top", min(limit or 1000, 1000)),
            ("$expand", "Attributes"),
            ("$expand", "Assets"),
        ]
        r = retry(lambda: self.session.get(CDSE_ODATA_URL, params=params, timeout=120),
                  what="CDSE search")
        r.raise_for_status()
        products = [p for p in r.json().get("value", []) if p.get("Online", True)]
        return products[:limit] if limit else products

    def download(self, product, out: Path):
        dst = out / f"{product['Name'].removesuffix('.SAFE')}.zip"
        if dst.exists():
            return dst
        url = CDSE_DOWNLOAD_URL.format(id=product["Id"])
        # Follow redirects manually so the bearer token is kept
        for _ in range(5):
            headers = {"Authorization": f"Bearer {self.token()}"}
            r = self.session.head(url, headers=headers, allow_redirects=False, timeout=60)
            if r.status_code in (301, 302, 303, 307, 308):
                url = r.headers["Location"]
                continue
            break
        http_download(url, dst, session=self.session,
                      headers={"Authorization": f"Bearer {self.token()}"},
                      desc=product["Name"][:40])
        return dst


def cdse_quicklook(product) -> str | None:
    """Public JPEG of a Copernicus product, when the catalogue lists one."""
    for asset in product.get("Assets") or []:
        if (asset.get("Type") or "").upper() == "QUICKLOOK" and asset.get("DownloadLink"):
            return asset["DownloadLink"]
    return None


def cdse_attr(product, name):
    for a in product.get("Attributes", []):
        if a.get("Name") == name:
            return a.get("Value")
    return None


# --------------------------------------------------------------------------- #
# NOAA Digital Coast
# --------------------------------------------------------------------------- #
class NOAA:
    """Search and clip NOAA coastal imagery. No account is required."""

    def _service(self, product: str) -> str:
        name = NOAA_PRODUCTS.get(product)
        if not name:
            raise ValueError(
                f"Unknown NOAA product {product!r}. "
                f"Choose one of: {', '.join(NOAA_PRODUCTS)}")
        return f"{NOAA_IMAGE}/{name}/ImageServer"

    def _get(self, url, params):
        r = requests.get(url, params=params, timeout=90)
        r.raise_for_status()
        if "json" in r.headers.get("content-type", ""):
            data = r.json()
            if isinstance(data, dict) and data.get("error"):
                message = data["error"].get("message") or str(data["error"])
                raise RuntimeError(message)
            return data
        return r

    def search(self, product, aoi: AOI, start, end, limit):
        """Return dataset rows (one per NOAA mission) that meet the AOI."""
        latest = product == "noaa-naip"
        y0, y1 = int(str(start)[:4]), int(str(end)[:4])
        if y1 < y0:
            y0, y1 = y1, y0
        if latest:
            where = "Category=1 AND (Sensor='NAIP' OR Name_1 LIKE '%NAIP%')"
            services = ["4Band_RGBN_8Bit_Imagery", "3Band_RGB_8Bit_Imagery"]
        else:
            where = f"Category=1 AND Year>={y0} AND Year<={y1}"
            services = [NOAA_PRODUCTS[product]]
        minx, miny, maxx, maxy = aoi.bbox
        features, service = [], services[0]
        for service in services:
            data = self._get(f"{NOAA_IMAGE}/{service}/ImageServer/query", {
                "where": where,
                "geometry": f"{minx},{miny},{maxx},{maxy}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "groupByFieldsForStatistics": "Mission,Name_1,Year,Resolution,Sensor",
                "outStatistics": json.dumps([{
                    "statisticType": "count",
                    "onStatisticField": "OBJECTID",
                    "outStatisticFieldName": "tiles",
                }]),
                "f": "json",
            })
            features = data.get("features") or []
            if features:
                break
        if latest and features:
            newest = max(int(f["attributes"].get("Year") or 0) for f in features)
            features = [f for f in features
                        if int(f["attributes"].get("Year") or 0) == newest]
        features.sort(key=lambda f: (f["attributes"].get("Year") or 0), reverse=True)
        rows = []
        for feat in features[: int(limit)]:
            attr = feat["attributes"]
            mission = attr.get("Mission")
            year = attr.get("Year")
            name = attr.get("Name_1") or f"NOAA mission {mission}"
            resolution = attr.get("Resolution")
            sensor = attr.get("Sensor") or ""
            rows.append({
                "id": f"{mission}-{year}",
                "geometry": self._footprint(product, mission, aoi.bbox, service),
                "properties": {
                    "datetime": f"{int(year)}-01-01T00:00:00Z" if year else None,
                    "title": name,
                    "year": int(year) if year else None,
                    "resolution": resolution,
                    "sensor": sensor,
                    "tiles": attr.get("tiles"),
                },
                "assets": {},
                "noaa": {
                    "product": product,
                    "service": service,
                    "mission": mission,
                    "name": name,
                    "year": int(year) if year else None,
                    "resolution": resolution,
                },
            })
        return rows

    def _footprint(self, product, mission, bbox, service=None):
        """Bounding box of the tiles in this mission that meet the AOI."""
        if mission is None:
            return None
        minx, miny, maxx, maxy = bbox
        endpoint = (f"{NOAA_IMAGE}/{service}/ImageServer" if service
                    else self._service(product))
        try:
            data = self._get(endpoint + "/query", {
                "where": f"Category=1 AND Mission={int(mission)}",
                "geometry": f"{minx},{miny},{maxx},{maxy}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "returnExtentOnly": "true",
                "returnGeometry": "false",
                "outSR": "4326",
                "f": "json",
            })
        except (requests.RequestException, RuntimeError, ValueError):
            return None
        ext = data.get("extent") or {}
        try:
            x0, y0, x1, y1 = (float(ext[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
        except (KeyError, TypeError, ValueError):
            return None
        return {"type": "Polygon", "coordinates": [[
            [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}

    def preview_url(self, scene, aoi: AOI, max_px: int = 480) -> str:
        """JPEG of this dataset clipped to the area, small enough to show in a browser."""
        from urllib.parse import urlencode

        info = scene["noaa"]
        mission = int(info["mission"])
        minx, miny, maxx, maxy = aoi.bbox
        mid = math.radians((miny + maxy) / 2.0)
        width_m = max(maxx - minx, 1e-9) * 111_320 * max(math.cos(mid), 0.2)
        height_m = max(maxy - miny, 1e-9) * 111_320
        aspect = width_m / height_m
        if aspect >= 1:
            size_w, size_h = max_px, max(32, int(round(max_px / aspect)))
        else:
            size_h, size_w = max_px, max(32, int(round(max_px * aspect)))
        rule = {
            "mosaicMethod": "esriMosaicAttribute",
            "where": f"Mission={mission}",
            "mosaicOperation": "MT_FIRST",
        }
        service = info.get("service") or NOAA_PRODUCTS[info["product"]]
        params = {
            "bbox": f"{minx},{miny},{maxx},{maxy}",
            "bboxSR": "4326",
            "size": f"{size_w},{size_h}",
            "format": "jpg",
            "interpolation": "RSP_BilinearInterpolation",
            "mosaicRule": json.dumps(rule),
            "f": "image",
        }
        return f"{NOAA_IMAGE}/{service}/ImageServer/exportImage?{urlencode(params)}"

    def download(self, scene, aoi: AOI, out: Path):
        """Export a GeoTIFF of the AOI for one dataset. Returns (path, note)."""
        info = scene["noaa"]
        mission = int(info["mission"])
        res_m = float(info.get("resolution") or 1.0) or 1.0
        minx, miny, maxx, maxy = aoi.bbox
        mid = math.radians((miny + maxy) / 2.0)
        width_m = max(maxx - minx, 1e-9) * 111_320 * max(math.cos(mid), 0.2)
        height_m = max(maxy - miny, 1e-9) * 111_320
        px_w = max(1, int(round(width_m / res_m)))
        px_h = max(1, int(round(height_m / res_m)))
        scale = max(px_w / NOAA_MAX_WIDTH, px_h / NOAA_MAX_HEIGHT, 1.0)
        size_w = min(NOAA_MAX_WIDTH, max(32, int(px_w / scale)))
        size_h = min(NOAA_MAX_HEIGHT, max(32, int(px_h / scale)))
        rule = {
            "mosaicMethod": "esriMosaicAttribute",
            "where": f"Mission={mission}",
            "mosaicOperation": "MT_FIRST",
        }
        service = info.get("service") or NOAA_PRODUCTS[info["product"]]
        url = f"{NOAA_IMAGE}/{service}/ImageServer/exportImage"
        params = {
            "bbox": f"{minx},{miny},{maxx},{maxy}",
            "bboxSR": "4326",
            "size": f"{size_w},{size_h}",
            "format": "tiff",
            "interpolation": "RSP_BilinearInterpolation",
            "mosaicRule": json.dumps(rule),
            "f": "image",
        }
        dst = out / f"{safe_name(info['name'])[:90]}.tif"
        with requests.get(url, params=params, stream=True, timeout=180) as r:
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            if "json" in ctype or ctype.startswith("text/"):
                detail = r.text[:300]
                raise RuntimeError(detail or "NOAA export failed")
            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        note = ""
        if scale > 1.05:
            note = (f"{info['name']}: resampled to about {res_m * scale:.2f} m "
                    f"(native {res_m:g} m) so the clip fits NOAA's export limit.")
        return dst, note


# --------------------------------------------------------------------------- #
# USGS 3DEP lidar
# --------------------------------------------------------------------------- #
def _iso_day(value) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def _ms_day(ms) -> str | None:
    if not ms:
        return None
    try:
        when = datetime.fromtimestamp(int(ms) / 1000, timezone.utc)
    except (TypeError, ValueError, OSError):
        return None
    return when.strftime("%Y-%m-%d")


def _box_polygon(bounds: dict) -> dict | None:
    try:
        x0, y0 = float(bounds["minX"]), float(bounds["minY"])
        x1, y1 = float(bounds["maxX"]), float(bounds["maxY"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"type": "Polygon", "coordinates": [[
        [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def _esri_polygon(geometry: dict) -> dict | None:
    rings = (geometry or {}).get("rings") or []
    polygons = []
    for ring in rings:
        coords = [[float(pt[0]), float(pt[1])] for pt in ring if len(pt) >= 2]
        if len(coords) < 4:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        polygons.append([coords])
    if not polygons:
        return None
    if len(polygons) == 1:
        return {"type": "Polygon", "coordinates": polygons[0]}
    return {"type": "MultiPolygon", "coordinates": polygons}


def _intersects(aoi: AOI, geometry: dict | None) -> bool:
    if not geometry or not aoi.geometry:
        return True
    from shapely.geometry import shape

    try:
        return shape(aoi.geometry).intersects(shape(geometry))
    except Exception:  # noqa: BLE001
        return True


class USGS:
    """Search and download USGS 3DEP lidar. No account is required."""

    def search(self, collection, aoi: AOI, start, end, limit):
        if collection == "lidar-projects":
            return self.search_projects(aoi, start, end, limit)
        if collection == "lidar-lpc":
            return self.search_tiles(aoi, start, end, limit)
        raise ValueError(
            f"Unknown USGS collection {collection!r}. "
            f"Choose one of: {', '.join(LIDAR_COLLECTIONS)}")

    def search_tiles(self, aoi: AOI, start, end, limit, text=None,
                     url_contains=None):
        """LAZ tiles whose footprint meets the area, newest publication first."""
        want = max(1, int(limit))
        found, offset, total = [], 0, None
        while offset < 500:
            page = self._products(aoi, start, end, text, 50, offset)
            items = page.get("items") or []
            total = page.get("total")
            if not items:
                break
            for item in items:
                row = self._tile_row(item, total)
                if row is None:
                    continue
                url = row["lidar"]["download"]
                if url_contains and url_contains not in url:
                    continue
                if not _intersects(aoi, row["geometry"]):
                    continue
                found.append(row)
            offset += len(items)
            if total is not None and offset >= int(total):
                break
        found.sort(key=lambda row: row["properties"].get("datetime") or "",
                   reverse=True)
        chosen = found[:want]
        if chosen and (len(found) > want or (total and offset < int(total))):
            for row in chosen:
                row["properties"]["truncated"] = True
        return chosen

    def search_projects(self, aoi: AOI, start, end, limit):
        """3DEP lidar work units that were collected in the date range."""
        want = max(1, int(limit))
        start_day, end_day = _iso_day(start), _iso_day(end)
        if end_day < start_day:
            start_day, end_day = end_day, start_day
        where = (f"collect_start <= DATE '{end_day}' AND "
                 f"collect_end >= DATE '{start_day}'")
        minx, miny, maxx, maxy = aoi.bbox
        found, offset = [], 0
        more = False
        while len(found) < want and offset < 5000:
            data = self._index({
                "where": where,
                "geometry": f"{minx},{miny},{maxx},{maxy}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": ("workunit,project,ql,spec,p_method,collect_start,"
                              "collect_end,lpc_pub_date,lpc_link,metadata_link,"
                              "lpc_category,horiz_crs,vert_crs,geoid"),
                "returnGeometry": "true",
                "outSR": "4326",
                "orderByFields": "collect_start DESC",
                "resultRecordCount": 100,
                "resultOffset": offset,
                "f": "json",
            })
            features = data.get("features") or []
            if not features:
                break
            for feat in features:
                row = self._project_row(feat)
                if row and _intersects(aoi, row["geometry"]):
                    found.append(row)
                    if len(found) >= want:
                        break
            offset += len(features)
            more = bool(data.get("exceededTransferLimit"))
            if not more:
                break
        if len(found) >= want and more:
            for row in found:
                row["properties"]["truncated"] = True
        self._attach_previews(found, aoi)
        return found

    def download(self, scene, aoi: AOI, out: Path, tile_limit=8):
        """Save one LAZ tile, or the tiles of one project that meet the area."""
        info = scene.get("lidar") or {}
        out.mkdir(parents=True, exist_ok=True)
        if info.get("kind") == "project":
            return self._download_project(scene, aoi, out, tile_limit)
        path = self._save_tile(info.get("download") or "", out)
        return [path], ""

    def _download_project(self, scene, aoi: AOI, out: Path, tile_limit):
        info = scene["lidar"]
        workunit = info["workunit"]
        tiles = self.search_tiles(
            aoi, "1990-01-01", _iso_day(date.today()), max(1, int(tile_limit)),
            text=workunit, url_contains=f"/{workunit}/")
        paths = []
        for tile in tiles:
            paths.append(self._save_tile(tile["lidar"]["download"], out))
        record = {
            "workunit": workunit,
            "project": info.get("project"),
            "quality": info.get("quality"),
            "collected": scene["properties"].get("datetime"),
            "point_cloud": info.get("folder"),
            "metadata": info.get("metadata"),
            "explorer": LIDAR_EXPLORER,
            "tiles": [path.name for path in paths],
        }
        (out / f"{safe_name(workunit)}.json").write_text(json.dumps(record, indent=1))
        note = f"{workunit}: saved {len(paths)} LAZ tile(s) that meet this area."
        if info.get("folder"):
            note += f" The rest of the project is at {info['folder']}"
        return paths, note

    def _save_tile(self, url: str, out: Path) -> Path:
        from urllib.parse import unquote, urlparse

        if not url:
            raise RuntimeError("This lidar result has no download link.")
        name = safe_name(unquote(urlparse(url).path.rsplit("/", 1)[-1])) or "lidar.laz"
        dst = out / name
        if dst.exists() and dst.stat().st_size > 0:
            return dst
        # rockyweb often accepts the connection slowly, then streams steadily.
        try:
            http_download(url, dst, timeout=(30, 180))
        except requests.RequestException as exc:
            raise RuntimeError(
                "USGS did not finish sending this LAZ file from rockyweb.usgs.gov. "
                "Try the download again in a moment."
            ) from exc
        return dst

    def _products(self, aoi: AOI, start, end, text, page, offset):
        minx, miny, maxx, maxy = aoi.bbox
        params = {
            "datasets": "Lidar Point Cloud (LPC)",
            "bbox": f"{minx},{miny},{maxx},{maxy}",
            "prodFormats": "LAZ",
            "max": int(page),
            "offset": int(offset),
            "outputFormat": "JSON",
        }
        if start and end:
            params["dateType"] = "Publication"
            params["start"] = _iso_day(start)
            params["end"] = _iso_day(end)
        if text:
            params["q"] = text
        response = requests.get(TNM_PRODUCTS, params=params, timeout=90)
        response.raise_for_status()
        data = response.json()
        errors = data.get("errors") or []
        if errors:
            raise RuntimeError("; ".join(str(err) for err in errors))
        return data

    def _index(self, params):
        response = requests.get(LIDAR_INDEX, params=params, timeout=90)
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            message = data["error"].get("message") or str(data["error"])
            raise RuntimeError(message)
        return data

    def _tile_row(self, item, total):
        url = item.get("downloadLazURL") or item.get("downloadURL")
        bounds = _box_polygon(item.get("boundingBox") or {})
        if not url or bounds is None:
            return None
        from urllib.parse import unquote, urlparse

        name = unquote(urlparse(url).path.rsplit("/", 1)[-1])
        published = (item.get("publicationDate") or "")[:10]
        size = item.get("sizeInBytes") or 0
        return {
            "id": safe_name(name) or safe_name(item.get("title") or "lidar"),
            "geometry": bounds,
            "properties": {
                "datetime": f"{published}T00:00:00Z" if published else None,
                "title": item.get("title") or name,
                "format": item.get("format") or "LAZ",
                "size_mb": round(size / 1e6, 1),
                "catalogue_total": total,
            },
            "lidar": {
                "kind": "tile",
                "download": url,
                "preview": item.get("previewGraphicURL"),
                "metadata": item.get("metaUrl") or item.get("vendorMetaUrl"),
            },
        }

    def _project_row(self, feature):
        attr = feature.get("attributes") or {}
        workunit = attr.get("workunit")
        if not workunit:
            return None
        collected = _ms_day(attr.get("collect_start"))
        return {
            "id": safe_name(str(workunit)),
            "geometry": _esri_polygon(feature.get("geometry") or {}),
            "properties": {
                "datetime": f"{collected}T00:00:00Z" if collected else None,
                "title": workunit,
                "project": attr.get("project"),
                "quality": attr.get("ql"),
                "specification": attr.get("spec"),
                "method": attr.get("p_method"),
                "category": attr.get("lpc_category"),
                "published": _ms_day(attr.get("lpc_pub_date")),
            },
            "lidar": {
                "kind": "project",
                "workunit": workunit,
                "project": attr.get("project"),
                "quality": attr.get("ql"),
                "folder": attr.get("lpc_link"),
                "metadata": attr.get("metadata_link"),
                "preview": None,
            },
        }

    def _attach_previews(self, rows, aoi: AOI):
        if not rows:
            return

        def one(row):
            workunit = row["lidar"]["workunit"]
            try:
                page = self._products(aoi, None, None, workunit, 8, 0)
            except (requests.RequestException, RuntimeError, ValueError):
                return None
            fallback = None
            for item in page.get("items") or []:
                preview = item.get("previewGraphicURL")
                if not preview:
                    continue
                url = item.get("downloadLazURL") or item.get("downloadURL") or ""
                if f"/{workunit}/" in url:
                    return preview
                fallback = fallback or preview
            return fallback

        with ThreadPoolExecutor(max_workers=6) as pool:
            previews = list(pool.map(one, rows))
        for row, preview in zip(rows, previews):
            row["lidar"]["preview"] = preview


def print_lidar_table(rows):
    if rows and (rows[0].get("lidar") or {}).get("kind") == "project":
        print(f"{'#':>3}  {'collected':<12} {'quality':<8}  work unit")
        for i, row in enumerate(rows, 1):
            props = row["properties"]
            when = (props.get("datetime") or "")[:10]
            print(f"{i:3}  {when:<12} {props.get('quality') or '-':<8}  "
                  f"{props.get('title')}")
        return
    print(f"{'#':>3}  {'published':<12} {'MB':>8}  tile")
    for i, row in enumerate(rows, 1):
        props = row["properties"]
        when = (props.get("datetime") or "")[:10]
        print(f"{i:3}  {when:<12} {props.get('size_mb') or 0:8}  {props.get('title')}")


def print_noaa_table(rows):
    print(f"{'#':>3}  {'year':<6} {'res m':>6} {'tiles':>6}  dataset")
    for i, row in enumerate(rows, 1):
        props = row["properties"]
        print(f"{i:3}  {props.get('year') or '-':<6} "
              f"{props.get('resolution') or 0:6} {props.get('tiles') or 0:6}  "
              f"{props.get('title')}")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def print_stac_table(items):
    print(f"{'#':>3}  {'date':<20} {'cloud':>6}  id")
    for i, it in enumerate(items, 1):
        cc = it.properties.get("eo:cloud_cover")
        cc = f"{cc:5.1f}" if isinstance(cc, (int, float)) else "   - "
        when = it.datetime.strftime("%Y-%m-%d %H:%M") if it.datetime else "-"
        print(f"{i:>3}  {when:<20} {cc:>6}  {it.id}")


def print_cdse_table(products):
    print(f"{'#':>3}  {'date':<20} {'cloud':>6} {'size MB':>8}  name")
    for i, p in enumerate(products, 1):
        cc = cdse_attr(p, "cloudCover")
        cc = f"{cc:5.1f}" if isinstance(cc, (int, float)) else "   - "
        size = (p.get("ContentLength") or 0) / 1e6
        print(f"{i:>3}  {p['ContentDate']['Start'][:16]:<20} {cc:>6} {size:8.0f}  "
              f"{p['Name']}")


def write_summary(out: Path, items, failures):
    rows = [{"id": it.id,
             "datetime": it.datetime.isoformat() if it.datetime else None,
             "cloud_cover": it.properties.get("eo:cloud_cover"),
             "platform": it.properties.get("platform")} for it in items]
    (out / "scenes.json").write_text(json.dumps(
        {"scenes": rows, "failures": [dict(zip(("id", "band", "error"), f))
                                      for f in failures]}, indent=1))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Search and download Sentinel, Landsat, NOAA imagery, and USGS 3DEP lidar.",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("Examples")[1].split("Install")[0]
                                if "Examples" in __doc__ else None)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--source",
                        choices=["earth-search", "planetary", "cdse", "noaa", "usgs"],
                        default="earth-search")
        sp.add_argument("--collection", help="default: sentinel-2-l2a (STAC) / "
                        "SENTINEL-2 (cdse) / noaa-rgb8 (noaa) / lidar-lpc (usgs)")
        g = sp.add_mutually_exclusive_group(required=True)
        g.add_argument("--bbox", nargs=4, type=float,
                       metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
        g.add_argument("--aoi", help="GeoJSON file (EPSG:4326)")
        sp.add_argument("--start", required=True, help="YYYY-MM-DD")
        sp.add_argument("--end", required=True, help="YYYY-MM-DD")
        sp.add_argument("--cloud", type=float,
                        help="max cloud cover %% (Sentinel-2 and Landsat)")
        sp.add_argument("--product-type",
                        help="cdse only, e.g. S2MSI2A, OL_2_LFR___, L2__NO2___, L1TP")
        sp.add_argument("--limit", type=int, default=50, help="max scenes (default 50)")

    sp_search = sub.add_parser("search", help="list matching scenes")
    common(sp_search)
    sp_search.add_argument("--json", help="also save results to this JSON file")

    sp_dl = sub.add_parser("download", help="download matching scenes")
    common(sp_dl)
    sp_dl.add_argument("--bands", nargs="+", default=["red", "green", "blue", "nir"],
                       help="band names (STAC sources). S2 aliases: coastal blue green "
                            "red rededge1-3 nir nir08 nir09 swir16 swir22 scl visual; "
                            "Landsat 8/9: coastal blue green red nir swir16 swir22 "
                            "lwir11 qa_pixel; S1: vv vh. Sentinel-3 and Sentinel-5P "
                            "use native asset keys (otci, no2, …).")
    sp_dl.add_argument("--mode", choices=["clip", "full"], default="clip")
    sp_dl.add_argument("--stack", nargs="+", metavar="BAND",
                       help="also write stack.tif of these bands (clip mode)")
    sp_dl.add_argument("--out", default="sentinel_data", help="output folder")
    sp_dl.add_argument("--workers", type=int, default=4)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    aoi = AOI.from_geojson(args.aoi) if args.aoi else AOI.from_bbox(args.bbox)
    collection = args.collection or DEFAULT_COLLECTION[args.source]

    if args.source == "cdse":
        cdse = CDSE()
        products = cdse.search(collection, aoi, args.start, args.end, args.cloud,
                               args.product_type, args.limit)
        if not products:
            print("No products found.")
            return 0
        print_cdse_table(products)
        if args.cmd == "search":
            if args.json:
                Path(args.json).write_text(json.dumps(products, indent=1))
            return 0
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        failed = 0
        for p in products:  # sequential: CDSE limits concurrent downloads per user
            try:
                path = retry(lambda p=p: cdse.download(p, out), what=p["Name"])
                print(f"  ✓ {path}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  ✗ {p['Name']}: {exc}", file=sys.stderr)
        (out / "products.json").write_text(json.dumps(products, indent=1))
        return 1 if failed else 0

    if args.source == "noaa":
        rows = NOAA().search(collection, aoi, args.start, args.end, args.limit)
        if not rows:
            print("No NOAA coastal imagery intersects this area and year range.")
            return 0
        print_noaa_table(rows)
        if args.cmd == "search":
            if args.json:
                Path(args.json).write_text(json.dumps(rows, indent=1))
            return 0
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        noaa = NOAA()
        failed = 0
        for row in rows:
            try:
                path, note = retry(lambda row=row: noaa.download(row, aoi, out),
                                   what=row["properties"].get("title") or row["id"])
                print(f"  ✓ {path}")
                if note:
                    print(f"    {note}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  ✗ {row['id']}: {exc}", file=sys.stderr)
        return 1 if failed else 0

    if args.source == "usgs":
        rows = USGS().search(collection, aoi, args.start, args.end, args.limit)
        if not rows:
            print("No USGS 3DEP lidar meets this area and date range.")
            return 0
        print_lidar_table(rows)
        if args.cmd == "search":
            if args.json:
                Path(args.json).write_text(json.dumps(rows, indent=1))
            return 0
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        usgs = USGS()
        failed = 0
        for row in rows:
            label = row["properties"].get("title") or row["id"]
            try:
                paths, note = retry(lambda row=row: usgs.download(
                    row, aoi, out, args.limit), what=label)
                print(f"  ✓ {', '.join(str(path) for path in paths)}")
                if note:
                    print(f"    {note}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  ✗ {label}: {exc}", file=sys.stderr)
        return 1 if failed else 0

    if args.source == "earth-search" and collection == "naip":
        items = naip_search(aoi, args.limit)
        if not items:
            print("No USDA NAIP imagery covers this area.")
            return 0
        print_stac_table(items)
        if args.cmd == "search":
            if args.json:
                Path(args.json).write_text(json.dumps([i.to_dict() for i in items],
                                                      indent=1))
            return 0
        failures = naip_download(items, aoi, Path(args.out))
        if failures:
            for item_id, err in failures:
                print(f"  ✗ {item_id}: {err}", file=sys.stderr)
            return 1
        print(f"Done → {Path(args.out).resolve()}")
        return 0

    items = stac_search(args.source, collection, aoi, args.start, args.end,
                        args.cloud, args.limit)
    if not items:
        print("No scenes found.")
        return 0
    print_stac_table(items)
    if args.cmd == "search":
        if args.json:
            Path(args.json).write_text(json.dumps([i.to_dict() for i in items],
                                                  indent=1))
        return 0

    failures = stac_download(items, args.source, collection, aoi, args.bands,
                             args.mode, Path(args.out), args.stack, args.workers)
    if failures:
        print(f"{len(failures)} band(s) failed — see {args.out}/scenes.json",
              file=sys.stderr)
        return 1
    print(f"Done → {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
