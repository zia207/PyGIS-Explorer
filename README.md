# PyGIS Explorer

Search and download Sentinel, Landsat, USDA NAIP, and NOAA coastal imagery. The Streamlit app draws an area on a map, lists the matching scenes with thumbnails, and saves the files you select. The same catalogues are available from `image_downloader.py`.

## Run the app

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL Streamlit prints. Downloads are written under `pygis_runs/`.

Draw a rectangle or polygon on the map, or type a bounding box (`minlon,minlat,maxlon,maxlat`) or upload a GeoJSON file in EPSG:4326. Set the source and collection in the sidebar, then search. Tick the rows you want and download them.

For USDA NAIP, the page shows which years were flown in each state before you search. A download uses the newest year that covers the area. After any search, the scene browser shows a thumbnail of each result. **View** opens that scene larger.

## Sources

| Source | Account | What you get |
| --- | --- | --- |
| Earth Search (AWS) | none | Sentinel-2 L2A and L1C, Landsat 8–9 Collection 2 Level-2, latest USDA NAIP |
| Planetary Computer | none | Sentinel-2, Sentinel-1 RTC and GRD, Landsat 8–9, Sentinel-3 and Sentinel-5P NetCDF |
| Copernicus Data Space | free registration | Sentinel-1, 2, 3, and 5P, plus Landsat 8 and 9, as full products |
| NOAA Digital Coast | none | Coastal natural color, color infrared, and the newest coastal NAIP mosaic |

`planetary-computer` is only required for the Planetary Computer source.

Copernicus downloads need a free account from [dataspace.copernicus.eu](https://dataspace.copernicus.eu). Set the credentials in the sidebar, or in the environment:

```bash
export CDSE_USER=you@example.com
export CDSE_PASSWORD=...
```

NOAA coverage is coastal. The NAIP availability map on that page is the nationwide USDA catalogue. The NOAA download itself is the coastal mosaic from the [Data Access Viewer](https://coast.noaa.gov/dataviewer/#/imagery/search).

Landsat Collection 2 Level-2 searches are limited to Landsat 8 and Landsat 9. Sentinel-5 is not in these catalogues. Use Sentinel-5P.

## Download modes

Earth Search and Planetary Computer can clip a scene to the area of interest, or save the whole file. Clip is the default. A clip can also write a multi-band `stack.tif`.

USDA NAIP from Earth Search is always a clip with red, green, blue, and near infrared. NOAA exports a GeoTIFF of the area. Large areas are resampled to stay within NOAA's export size. Copernicus saves the whole product.

Sentinel-2 and Landsat clips can be inspected in the app as a true-colour image and an NDVI map.

## Command line

```bash
python image_downloader.py search --bbox -76.6 42.4 -76.4 42.5 \
    --start 2026-06-01 --end 2026-09-30 --cloud 10

python image_downloader.py download --bbox -76.6 42.4 -76.4 42.5 \
    --start 2026-06-01 --end 2026-09-30 --cloud 10 \
    --bands red green blue nir scl --stack red green blue nir --out s2_ithaca

python image_downloader.py download --source planetary \
    --collection sentinel-1-rtc --aoi field.geojson \
    --start 2026-07-01 --end 2026-07-31 --bands vv vh --out s1_rtc

python image_downloader.py download --source noaa --collection noaa-rgb8 \
    --bbox -79.95 32.75 -79.85 32.82 \
    --start 2016-01-01 --end 2024-12-31 --out noaa_charleston

python image_downloader.py download --source earth-search --collection naip \
    --bbox -79.95 32.75 -79.85 32.82 \
    --start 2010-01-01 --end 2026-10-15 --out naip_charleston
```

`--bbox` is `MINLON MINLAT MAXLON MAXLAT`. `--aoi` is a GeoJSON file. `--source` is `earth-search` (default), `planetary`, `cdse`, or `noaa`. `--mode` is `clip` or `full`. NAIP ignores the date window and keeps the newest year over the area. For NOAA, the dates are a year range.

`--collection` defaults to `sentinel-2-l2a`, `SENTINEL-2`, or `noaa-rgb8`, depending on the source. NOAA collections are `noaa-rgb8`, `noaa-rgb16`, `noaa-rgbn8`, `noaa-cir8`, and `noaa-naip`.
