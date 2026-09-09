# 1. Data to Insight — เหตุความรุนแรงในครอบครัว (Domestic Violence)

Part of the [Super AI Engineer Season 6](../README.md) hackathon series — a 1-week "Data to Insight" data
cleansing and analysis challenge.

## Dataset

**ชุดข้อมูลเหตุความรุนแรงในครอบครัว** (Domestic Violence Incidents in Thailand), published by the
กรมกิจการสตรีและสถาบันครอบครัว (Department of Women's Affairs and Family Development) via
[thackle.or.th/dataset/84](https://www.thackle.or.th/th/dataset/84). It contains statistics on family
violence incidents in Thailand — broken down by area, incident type, and time of day — for analyzing
social-trend problems and access to support services.

The raw data is split across three related CSV tables:

| File | Rows × Cols | Description |
|---|---|---|
| `ข้อมูลการแจ้งเหตุความรุนแรงในครอบครัว.csv` | 877 × 11 | Incident reports — region, province, district, sub-district, source/channel of report, period of day, locale |
| `ข้อมูลผู้กระทำความรุนแรงในครอบครัว.csv` | 564 × 19 | Perpetrators — demographics (gender, age, age range) and risk factors (alcohol, drugs, authoritative behavior, rage, jealousy, divorce, health/mental problems, gambling, economic stress) |
| `ข้อมูลผู้ถูกกระทำความรุนแรงในครอบครัว.csv` | 597 × 20 | Victims — same demographics/risk factors as perpetrators, plus `Relation Type` (relationship to the perpetrator) |

<details>
<summary><b>Column schema (click to expand)</b> — re-rendered from the notebook's <code>1.png</code>/<code>2.png</code>/<code>3.png</code> in one consistent style</summary>
<br>

<img src="assets/schema/01_incidents_schema.png" width="100%" alt="Incidents table schema">
<img src="assets/schema/02_perpetrators_schema.png" width="100%" alt="Perpetrators table schema">
<img src="assets/schema/03_victims_schema.png" width="100%" alt="Victims table schema">

</details>

Supporting reference data used for enrichment and mapping:

- `province.csv` — Thai/English province names + region (source: [dataset 83](https://www.thackle.or.th/th/datatset/83))
- `th-adm1-province-centroid.geojson`, `th-adm2-district-centroid.geojson` — province/district centroid points
- `tha_admbnda_adm1_rtsd_20190221.*` — Thailand admin-1 (province) boundary shapefile, from
  [Open Development Mekong](https://data.opendevelopmentmekong.net/th/dataset/8f3fa1b8-cb5c-48c8-9fd7-d3c213ea23db)
- `thsarabunnew-webfont.ttf` — Thai font used for notebook chart labels

None of the three tables contain null values, so no imputation was needed — the cleansing work centered on
consistent category ordering (age ranges, time periods), joining incident counts to province geometry, and
standardizing labels for charting.

## Analysis notebook

[`Domestic_Violence.ipynb`](Domestic_Violence.ipynb) walks through:

1. **Exploration** — nulls check, unique values per column across all three tables.
2. **Victim profile** — gender split (~80% female), age distribution (risk rises with age, peaks in the
   36–59 middle-aged group), and risk-factor prevalence.
3. **Perpetrator profile** — gender split (~80% male), same age pattern, dominant relationship to victim is
   spouse.
4. **Victim vs. perpetrator comparison** — side-by-side age/gender breakdown and normalized (%) risk-factor
   comparison, since the two populations differ in size. Top risk factors for both groups: substance use,
   alcohol, and displays of authority/aggression.
5. **Area & time analysis** — incidents by region (ภาคกลาง > ภาคตะวันออกเฉียงเหนือ > ภาคเหนือ > ภาคใต้), by
   time of day (afternoon–evening peak), by locale (mostly private residences), and a province choropleth
   map (via `geopandas`) showing Bangkok and its metro area as the hotspot, both overall and split by
   time-of-day.

## Visual walkthrough

The charts below are re-rendered from the notebook in one clean, minimal, single-accent theme (`assets/charts/`)
instead of copying the notebook's original mixed-color output, so the gallery reads as one consistent set —
same font, same accent color for the primary series, neutral gray for anything de-emphasized.

**Victim profile**

| Gender | Age range | Age × gender |
|---|---|---|
| ![Victim gender](assets/charts/01_victim_gender.png) | ![Victim age](assets/charts/02_victim_age.png) | ![Victim age × gender](assets/charts/03_victim_age_gender.png) |

![Victim risk factors](assets/charts/04_victim_risk.png)

**Perpetrator profile**

| Gender | Age range | Age × gender |
|---|---|---|
| ![Perpetrator gender](assets/charts/05_perp_gender.png) | ![Perpetrator age](assets/charts/06_perp_age.png) | ![Perpetrator age × gender](assets/charts/07_perp_age_gender.png) |

![Perpetrator risk factors](assets/charts/08_perp_risk.png)

**Victim vs. perpetrator comparison**

| Age pyramid (victim vs. perpetrator) | Risk factor comparison (%) |
|---|---|
| ![Age pyramid](assets/charts/09_age_pyramid.png) | ![Risk factor comparison](assets/charts/10_compare_risk.png) |

**Area & time analysis**

| By region | By time of day | By locale |
|---|---|---|
| ![Regional](assets/charts/11_regional.png) | ![Period](assets/charts/12_period.png) | ![Locale](assets/charts/13_locale.png) |

| Top 10 provinces | Bangkok by time of day |
|---|---|
| ![Top province](assets/charts/14_top_province.png) | ![Bangkok period](assets/charts/15_bangkok_period.png) |

**Province maps** (choropleth, rendered with `pyshp` + `matplotlib` instead of `geopandas`)

![Incident map, overall](assets/charts/16_map_overall.png)

![Incident map, by time of day](assets/charts/17_map_by_period.png)

**Co-occurring risk factors**

![Risk factor count histogram](assets/charts/18_risk_factor_hist.png)

## Dashboard

[`dashboard.html`](dashboard.html) is a **static, self-contained** HTML dashboard summarizing the notebook's
findings — no build step, backend, or CDN dependency needed, just open it in a browser. It embeds the
pre-aggregated dataset as inline JSON and bundles the [Chart.js](https://www.chartjs.org/) and
[Leaflet](https://leafletjs.com/) libraries directly in the file, so all charts and KPI cards render even
offline or behind a network that blocks third-party script CDNs. The only thing that still needs a live
internet connection is the OpenStreetMap map tiles behind the province map (and the Google Font, which falls
back to a system sans-serif font if unavailable).

It covers:

- KPI summary (incident/victim/perpetrator counts, top region/province)
- An interactive province map with a time-of-day filter (00:01–06:00 / 06:01–12:00 / 12:01–18:00 / 18:01–00:00)
- Region, time-of-day, locale, and top-province breakdowns for incidents, plus a Bangkok-specific time chart
- Victim and perpetrator profiles: gender, age range, age × gender, risk-factor prevalence, marital status /
  relationship to perpetrator, and co-occurring risk-factor count
- A side-by-side victim-vs-perpetrator risk-factor comparison
- A static analysis gallery at the bottom, embedding the same themed images from the walkthrough above
  (`assets/charts/`, `assets/schema/`) for an at-a-glance reference alongside the live charts

The dashboard data was regenerated directly from the three source CSVs (not copied from notebook output), so
the figures match what re-running the notebook would produce. The gallery section needs the `assets/` folder
to sit next to `dashboard.html`; everything else in the file works standalone.

## Repo layout note

This repository holds multiple independent Super AI Engineer Season 6 competition projects, each in its own
top-level folder — see the [repo README](../README.md) for the full list. This folder covers only the
Data Cleansing / Data-to-Insight challenge.
