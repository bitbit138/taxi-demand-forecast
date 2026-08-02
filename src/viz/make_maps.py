"""Render the result figures — each one carrying a finding, not just data.

Outputs to ``reports/``:

  ``k_selection.png``        elbow + silhouette, with the disagreement annotated.
                             This is proposal question #1 answered as a figure.
  ``zone_hour_heatmap.png``  223 zones x 168 hours of the week, rows grouped by
                             cluster, so the nightlife late-peak and the business
                             evening-peak are visible as bands.
  ``cluster_map.html``       Folium map of the 223 zones by cluster, legend labelled
                             with the derived characters.
  ``geojson/clusters.geojson``          static zone -> cluster assignment
  ``geojson/demand_<how>.geojson``      per-time-window predicted demand

Design notes. The elbow and silhouette are drawn as **two stacked panels sharing one
x-axis, never a dual-axis chart** — two y-scales on one plot invent a correlation by
arbitrary alignment. Cluster colours use the first three categorical slots, which are
the ones that clear the all-pairs colour-vision gates a choropleth needs; the singleton
cluster is deliberately neutral grey rather than a fourth hue, because it is an artifact
rather than a peer category. The heatmap is a single-hue sequential ramp, light to dark.

    python -m src.viz.make_maps
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import folium  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.stream.predict_live import DAY_NAMES, Forecaster, how_label  # noqa: E402

GEOJSON_DIR = config.REPORTS_DIR / "geojson"

# Choropleth colours must clear the *all-pairs* colour-vision gate, not just the
# adjacent-pair one. The documented palette's first three slots do; adding its fourth
# (yellow) puts yellow beside orange and fails the normal-vision floor at 13.7 (needs
# 15). Slots 1-3 plus violet, plus a dark gold re-stepped away from orange, was found
# by running scripts/validate_palette.js over candidates:
#   CVD 9.2 (deutan) · normal-vision 16.3 · all checks PASS at 5 slots, light mode.
# Aqua sits at 2.74:1 contrast, so the relief rule applies — every zone carries a text
# tooltip and the legend names each cluster, so identity is never colour-alone.
# The singleton, when one exists, takes a neutral instead of a hue.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#8c5a00"]
NEUTRAL = "#7a7975"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdbd6"
ACCENT = "#4a3aa7"  # annotation layer, kept distinct from the data layer

# Single-hue sequential ramp, light -> dark (blue 100..700).
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", [SURFACE, *BLUE_RAMP])

# Windows chosen to show the clusters diverging, not at random.
DEFAULT_WINDOWS = [
    1 * 24 + 8,    # Tue 08:00 — commuter morning
    2 * 24 + 18,   # Wed 18:00 — business evening
    5 * 24 + 1,    # Sat 01:00 — nightlife
    6 * 24 + 4,    # Sun 04:00 — city-wide trough
]


def cluster_style(forecaster: Forecaster) -> dict[int, dict]:
    """Colour + short label per cluster, ordered so the biggest clusters lead."""
    order = (
        forecaster.zones.groupby("cluster").size().sort_values(ascending=False).index
    )
    n_hued = sum(
        1 for c in order if forecaster.characters[int(c)]["n_zones"] > 1
    )
    if n_hued > len(SERIES):
        raise ValueError(
            f"{n_hued} clusters need a hue but only {len(SERIES)} validated slots "
            "exist. Do not wrap the palette — two clusters would share a colour and "
            "the map would be unreadable. Extend SERIES and re-run "
            "scripts/validate_palette.js with --pairs all before using it."
        )

    styles: dict[int, dict] = {}
    slot = 0
    for cluster in order:
        character = forecaster.characters[int(cluster)]
        singleton = character["n_zones"] == 1
        if singleton:
            colour, weight, dash = NEUTRAL, 2.5, "6,3"
        else:
            colour, weight, dash = SERIES[slot], 0.6, None
            slot += 1
        styles[int(cluster)] = {
            "color": colour,
            "weight": weight,
            "dash": dash,
            "singleton": singleton,
            "label": character["label"],
            "short": character["label"].split("—")[0].strip(),
            "peak": character["peak_label"],
            "n_zones": character["n_zones"],
        }

    # Two clusters can legitimately earn the same character — the full year splits
    # residential commute into two groups. A legend with two identically named
    # entries is unreadable, so collisions are suffixed with their peak hour, which
    # is what actually distinguishes them.
    from collections import Counter  # noqa: PLC0415

    counts = Counter(s["short"] for s in styles.values())
    for style in styles.values():
        if counts[style["short"]] > 1:
            style["short"] = f"{style['short']} ({style['peak']})"
    return styles


def held_out_metrics() -> dict[str, dict]:
    """Test-split metrics, so any accuracy shown on a figure keeps its caveat."""
    if not config.METRICS_CSV.exists():
        return {}
    metrics = pd.read_csv(config.METRICS_CSV)
    metrics = metrics[metrics["subset"] == "all"]
    return {row["method"]: row.to_dict() for _, row in metrics.iterrows()}


# --------------------------------------------------------------------------- #
# Figure 1 — K selection
# --------------------------------------------------------------------------- #
def plot_k_selection(chosen_k: int, metadata: dict) -> Path:
    """Elbow and silhouette as stacked panels, with the disagreement annotated."""
    sweep = pd.read_csv(config.KSWEEP_CSV)
    norm = sweep[sweep["variant"] == "normalized"].sort_values("k")

    k_elbow = int(metadata.get("k_suggested_by_elbow", 0))
    k_sil = int(metadata.get("k_suggested_by_silhouette", 0))

    fig, (ax_wcss, ax_sil) = plt.subplots(
        2, 1, figsize=(9, 7.5), sharex=True,
        gridspec_kw={"hspace": 0.18, "height_ratios": [1, 1]},
    )
    fig.patch.set_facecolor(SURFACE)

    for axis in (ax_wcss, ax_sil):
        axis.set_facecolor(SURFACE)
        axis.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=INK_MUTED, labelsize=9)

    # --- elbow -------------------------------------------------------------
    ax_wcss.plot(norm["k"], norm["wcss"], color=SERIES[0], linewidth=2,
                 marker="o", markersize=6, markerfacecolor=SERIES[0],
                 markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax_wcss.set_ylabel("WCSS (within-cluster sum of squares)", color=INK_MUTED,
                       fontsize=10)
    ax_wcss.set_title("Elbow — variance explained keeps improving past K=4",
                      color=INK, fontsize=11, loc="left", pad=8)

    # --- silhouette --------------------------------------------------------
    ax_sil.plot(norm["k"], norm["silhouette"], color=SERIES[1], linewidth=2,
                marker="o", markersize=6, markerfacecolor=SERIES[1],
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax_sil.axhline(0, color=GRID, linewidth=1)
    ax_sil.set_ylabel("Silhouette (higher = better separated)", color=INK_MUTED,
                      fontsize=10)
    ax_sil.set_xlabel("K (number of clusters)", color=INK_MUTED, fontsize=10)
    ax_sil.set_title("Silhouette — separation is best at the coarsest split",
                     color=INK, fontsize=11, loc="left", pad=8)

    # --- annotation layer, deliberately not in a series colour -------------
    for axis in (ax_wcss, ax_sil):
        axis.axvline(k_elbow, color=INK_MUTED, linewidth=1.2, linestyle=(0, (5, 4)))
        axis.axvline(k_sil, color=INK_MUTED, linewidth=1.2, linestyle=(0, (5, 4)))
        axis.axvline(chosen_k, color=ACCENT, linewidth=2.2)

    wcss_at = dict(zip(norm["k"], norm["wcss"]))
    sil_at = dict(zip(norm["k"], norm["silhouette"]))

    ax_wcss.annotate(
        f"WCSS knee\nK={k_elbow}", xy=(k_elbow, wcss_at[k_elbow]),
        xytext=(k_elbow + 0.5, wcss_at[k_elbow] + 0.45 * (max(norm['wcss']) - min(norm['wcss']))),
        color=INK, fontsize=9,
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=1),
    )
    sil_span = max(norm["silhouette"]) - min(norm["silhouette"])
    ax_sil.annotate(
        f"silhouette peak\nK={k_sil} ({sil_at[k_sil]:+.3f})",
        xy=(k_sil, sil_at[k_sil]),
        xytext=(k_sil + 0.35, sil_at[k_sil] - 0.42 * sil_span),
        color=INK, fontsize=9,
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=1),
    )
    ax_sil.annotate(
        f"K={chosen_k} chosen", xy=(chosen_k, sil_at[chosen_k]),
        xytext=(chosen_k + 0.35, sil_at[chosen_k] + 0.22 * sil_span),
        color=ACCENT, fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=1.2),
    )
    # K is discrete — label every value, not matplotlib's even-numbered default.
    ax_sil.set_xticks(list(norm["k"]))

    ax_wcss.legend(
        handles=[
            Line2D([], [], color=INK_MUTED, linestyle=(0, (5, 4)), linewidth=1.2,
                   label=f"criterion suggestions (K={k_sil}, K={k_elbow})"),
            Line2D([], [], color=ACCENT, linewidth=2.2, label=f"chosen K={chosen_k}"),
        ],
        loc="upper right", frameon=False, fontsize=9, labelcolor=INK_MUTED,
    )

    fig.suptitle(
        "Choosing K: the two criteria disagree",
        color=INK, fontsize=14, fontweight="bold", x=0.055, ha="left", y=0.98,
    )
    fig.text(
        0.055, 0.935,
        f"Elbow says K={k_elbow}, silhouette says K={k_sil}. K={chosen_k} was chosen after "
        f"inspecting cluster contents: it is the coarsest K that separates a\nnightlife "
        "group from the business core, which neither criterion can see. "
        "Normalised (shape) profiles, 223 zones, train split only.",
        color=INK_MUTED, fontsize=9.5, ha="left", va="top",
    )
    # Leave real space under the subtitle: at top=0.855 the first panel title
    # collided with the second line of it.
    fig.subplots_adjust(top=0.815, left=0.10, right=0.97, bottom=0.075)

    out = config.REPORTS_DIR / "k_selection.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Figure 2 — zone x hour-of-week heatmap
# --------------------------------------------------------------------------- #
def plot_heatmap(forecaster: Forecaster, styles: dict[int, dict]) -> Path:
    """223 zones x 168 hours, rows grouped by cluster, shares not raw counts.

    Rows are row-normalised so every zone contributes its *shape*; on raw counts
    Midtown would saturate the scale and the clusters would be invisible.
    """
    features = pd.read_parquet(
        config.FEATURES_PARQUET,
        columns=["zone_id", "hour_of_week", "trip_count", "is_train"],
    )
    train = features[features["is_train"]]
    profile = (
        train.groupby(["zone_id", "hour_of_week"])["trip_count"].mean().unstack(
            fill_value=0.0
        )
    )
    profile = profile.reindex(columns=range(config.HOURS_PER_WEEK), fill_value=0.0)
    shares = profile.div(profile.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    assign = forecaster.zones.set_index("zone_id")
    order = (
        assign.loc[shares.index, ["cluster", "zone_mean_demand"]]
        .sort_values(["cluster", "zone_mean_demand"], ascending=[True, False])
    )
    shares = shares.loc[order.index]

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    vmax = float(np.nanpercentile(shares.values, 99.5))
    image = ax.imshow(
        shares.values, aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=vmax,
        interpolation="nearest",
    )

    # Cluster bands: a rule between groups plus a labelled colour chip.
    boundaries, labels = [], []
    start = 0
    for cluster, group in order.groupby("cluster", sort=True):
        end = start + len(group)
        boundaries.append((start, end, int(cluster)))
        labels.append(int(cluster))
        start = end

    # A 1-row band cannot host a 2-line label at its own centre, so nudge label
    # positions apart while the colour chips stay on the true band extents.
    n_rows = len(order)
    min_gap = n_rows * 0.055
    centres: list[float] = []
    for begin, end, _ in boundaries:
        wanted = (begin + end) / 2 - 0.5
        if centres and wanted - centres[-1] < min_gap:
            wanted = centres[-1] + min_gap
        centres.append(wanted)

    for (begin, end, cluster), centre in zip(boundaries, centres):
        if begin:
            ax.axhline(begin - 0.5, color=INK, linewidth=1.4)
        style = styles[cluster]
        ax.add_patch(
            plt.Rectangle(
                (-4.6, begin - 0.5), 3.2, end - begin, clip_on=False,
                facecolor=style["color"], edgecolor="none",
            )
        )
        band_centre = (begin + end) / 2 - 0.5
        if abs(centre - band_centre) > 0.5:
            # Leader line back to the band the label belongs to.
            ax.plot(
                [-3.0, -1.6], [centre, band_centre], color=INK_MUTED,
                linewidth=0.8, clip_on=False,
            )
        plural = "zone" if style["n_zones"] == 1 else "zones"
        ax.text(
            -6.0, centre,
            f"c{cluster}  {style['short']}\n{style['n_zones']} {plural}",
            ha="right", va="center", fontsize=9, color=INK,
        )

    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xticks([d * 24 for d in range(7)])
    ax.set_xticklabels(DAY_NAMES, fontsize=9.5, color=INK_MUTED)
    ax.set_xticks([d * 24 + 12 for d in range(7)], minor=True)
    for day in range(1, 7):
        ax.axvline(day * 24 - 0.5, color=SURFACE, linewidth=0.8, alpha=0.5)
    ax.set_xlabel("hour of week (midnight tick at each day label)", color=INK_MUTED,
                  fontsize=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)

    bar = fig.colorbar(image, ax=ax, pad=0.015, fraction=0.026)
    bar.set_label("share of the zone's weekly demand falling in this hour",
                  color=INK_MUTED, fontsize=9.5)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8.5)
    bar.outline.set_visible(False)

    ax.set_title(
        "Weekly demand shape by zone, grouped by cluster",
        color=INK, fontsize=14, fontweight="bold", loc="left", pad=26,
    )
    ax.text(
        0, -0.085,
        "Each row is one zone, row-normalised to its own weekly total — so brightness is *when* a zone is busy, not how busy.\n"
        "The nightlife band peaks in the weekend small hours; the business-core band peaks on weekday evenings and empties at weekends.",
        transform=ax.transAxes, color=INK_MUTED, fontsize=9.5, va="top",
    )
    # right=0.99 clipped the colorbar tick labels off the canvas; left=0.20 clipped
    # cluster labels once collisions started carrying a "(Wed 07:00)" suffix.
    fig.subplots_adjust(left=0.245, right=0.925, top=0.90, bottom=0.16)

    out = config.REPORTS_DIR / "zone_hour_heatmap.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out, shares, order


# --------------------------------------------------------------------------- #
# Folium map + GeoJSON
# --------------------------------------------------------------------------- #
def zone_geometry() -> pd.DataFrame:
    """Zone polygons as GeoJSON geometry dicts."""
    from shapely import wkt  # noqa: PLC0415
    from shapely.geometry import mapping  # noqa: PLC0415

    zones = pd.read_parquet(config.ZONE_CENTROIDS_PARQUET)
    zones["geometry"] = zones["geometry_wkt"].apply(lambda w: mapping(wkt.loads(w)))
    return zones.drop(columns=["geometry_wkt"])


def build_map(forecaster: Forecaster, styles: dict[int, dict], metrics: dict) -> Path:
    """Folium choropleth of the 223 modelling zones by cluster."""
    geo = zone_geometry()
    served = forecaster.zones.merge(
        geo[["zone_id", "geometry", "area_km2"]], on="zone_id", how="inner"
    )

    fmap = folium.Map(
        location=[40.7300, -73.9350], zoom_start=11, tiles="cartodbpositron",
        control_scale=True,
    )

    for cluster in sorted(served["cluster"].unique()):
        style = styles[int(cluster)]
        group = folium.FeatureGroup(
            name=f"c{cluster} — {style['short']} ({style['n_zones']})", show=True
        )
        subset = served[served["cluster"] == cluster]

        for _, row in subset.iterrows():
            feature = {
                "type": "Feature",
                "geometry": row["geometry"],
                "properties": {
                    "zone_id": int(row["zone_id"]),
                    "zone_name": row["zone_name"],
                    "borough": row["borough"],
                    "cluster": int(cluster),
                    "cluster_character": style["label"],
                    "cluster_peak": style["peak"],
                    "zone_mean_demand": round(float(row["zone_mean_demand"]), 3),
                    "total_trips_train": int(row["total_trips"]),
                },
            }
            colour, weight, dash = style["color"], style["weight"], style["dash"]
            folium.GeoJson(
                feature,
                style_function=lambda _f, c=colour, w=weight, d=dash: {
                    "fillColor": c,
                    "color": "#3a3a38" if d else c,
                    "weight": w,
                    "dashArray": d,
                    "fillOpacity": 0.72,
                },
                highlight_function=lambda _f: {"weight": 3, "color": INK,
                                               "fillOpacity": 0.88},
                # Text identity on every mark — the relief rule for the aqua slot,
                # and it keeps identity from being colour-alone.
                tooltip=folium.GeoJsonTooltip(
                    fields=["zone_id", "zone_name", "borough", "cluster_character",
                            "cluster_peak", "zone_mean_demand"],
                    aliases=["Zone", "Name", "Borough", "Cluster", "Cluster peaks",
                             "Mean trips/hour"],
                    sticky=True,
                ),
            ).add_to(group)
        group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(legend_html(styles, metrics)))

    out = config.REPORTS_DIR / "cluster_map.html"
    fmap.save(str(out))
    return out


def legend_html(styles: dict[int, dict], metrics: dict) -> str:
    """Legend keyed by derived character, plus correctly-captioned accuracy."""
    rows = []
    for cluster in sorted(styles):
        style = styles[cluster]
        border = (
            f"border:2px dashed #3a3a38" if style["singleton"]
            else "border:1px solid rgba(0,0,0,.15)"
        )
        note = " &middot; singleton artifact" if style["singleton"] else ""
        rows.append(
            f'<div style="display:flex;align-items:flex-start;gap:8px;margin:5px 0">'
            f'<span style="flex:0 0 15px;height:15px;margin-top:2px;'
            f'background:{style["color"]};{border};border-radius:3px"></span>'
            f'<span><b>c{cluster} &mdash; {style["short"]}</b>{note}<br>'
            f'<span style="color:#52514e">{style["n_zones"]} zones &middot; '
            f'peaks {style["peak"]}</span></span></div>'
        )

    scaled = metrics.get("Cluster shape x zone level (K-Means)")
    hist = metrics.get("Historical avg (zone,hour,dow)")
    accuracy = ""
    if scaled and hist:
        accuracy = (
            f'<div style="margin-top:9px;padding-top:8px;'
            f'border-top:1px solid #dcdbd6;color:#52514e">'
            f'<b style="color:#0b0b0b">Held-out test split</b> '
            f'(2024-03-13&ndash;03-31)<br>'
            f'cluster shape &times; zone level &mdash; MAE '
            f'{scaled["mae"]:.2f}, WAPE {100 * scaled["wape"]:.1f}%<br>'
            f'per-zone hist. average &mdash; MAE {hist["mae"]:.2f}, '
            f'WAPE {100 * hist["wape"]:.1f}%<br>'
            f'<i>The clustering is the interpretable, live-serveable model, '
            f'not the most accurate one.</i></div>'
        )

    return f"""
    <div style="position:fixed;bottom:22px;left:22px;z-index:9999;max-width:340px;
                background:{SURFACE};padding:12px 14px;border-radius:8px;
                border:1px solid #dcdbd6;box-shadow:0 2px 10px rgba(0,0,0,.12);
                font:12.5px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;color:#0b0b0b">
      <div style="font-size:14px;font-weight:700;margin-bottom:2px">
        NYC taxi zones by weekly demand shape</div>
      <div style="color:#52514e;margin-bottom:7px">
        K-Means, K=4, on L1-normalised profiles &mdash; grouped by <i>when</i> demand
        happens, not how much. Train split only.</div>
      {''.join(rows)}
      {accuracy}
    </div>"""


def export_geojson(forecaster: Forecaster, styles: dict[int, dict],
                   windows: list[int]) -> list[Path]:
    """Static cluster assignment plus one file per time window."""
    GEOJSON_DIR.mkdir(parents=True, exist_ok=True)
    geo = zone_geometry()
    served = forecaster.zones.merge(geo[["zone_id", "geometry"]], on="zone_id")
    written: list[Path] = []

    def collection(features: list[dict], **extra) -> dict:
        return {
            "type": "FeatureCollection",
            "crs": {"type": "name",
                    "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "properties": {
                "source": "NYC TLC yellow taxi 2024-01..03, train split",
                "model": f"K-Means K={forecaster.metadata.get('chosen_k')} "
                         "on L1-normalised weekly profiles",
                "zones": len(features),
                **extra,
            },
            "features": features,
        }

    static = [
        {
            "type": "Feature",
            "geometry": row["geometry"],
            "properties": {
                "zone_id": int(row["zone_id"]),
                "zone_name": row["zone_name"],
                "borough": row["borough"],
                "cluster": int(row["cluster"]),
                "cluster_character": styles[int(row["cluster"])]["label"],
                "fill": styles[int(row["cluster"])]["color"],
                "zone_mean_demand": round(float(row["zone_mean_demand"]), 3),
            },
        }
        for _, row in served.iterrows()
    ]
    path = GEOJSON_DIR / "clusters.geojson"
    path.write_text(json.dumps(collection(static), indent=1))
    written.append(path)

    for how in windows:
        features = []
        for _, row in served.iterrows():
            zone_id = int(row["zone_id"])
            share = forecaster.shape[
                (forecaster.shape["cluster"] == int(row["cluster"]))
                & (forecaster.shape["hour_of_week"] == how)
            ]
            predicted = (
                float(row["zone_mean_demand"]) * float(share.iloc[0]["cluster_share"])
                * config.HOURS_PER_WEEK
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": row["geometry"],
                    "properties": {
                        "zone_id": zone_id,
                        "zone_name": row["zone_name"],
                        "cluster": int(row["cluster"]),
                        "cluster_character": styles[int(row["cluster"])]["label"],
                        "hour_of_week": how,
                        "window": how_label(how),
                        "predicted_demand": round(predicted, 2),
                        "fill": styles[int(row["cluster"])]["color"],
                    },
                }
            )
        name = how_label(how).replace(" ", "_").replace(":", "")
        path = GEOJSON_DIR / f"demand_{how:03d}_{name}.geojson"
        path.write_text(
            json.dumps(collection(features, window=how_label(how),
                                  hour_of_week=how), indent=1)
        )
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(forecaster: Forecaster, styles: dict[int, dict], outputs: list[Path],
             shares: pd.DataFrame, order: pd.DataFrame, metadata: dict) -> bool:
    ok = True
    print("\n" + "=" * 78)
    print("VALIDATION")
    print("=" * 78)

    expected = len(forecaster.zones)

    # --- map ------------------------------------------------------------------
    map_path = config.REPORTS_DIR / "cluster_map.html"
    html = map_path.read_text(encoding="utf-8")
    rendered = sum(html.count(f'"zone_id": {int(z)},') for z in forecaster.zones["zone_id"])
    print(f"\n1. Folium map")
    print(f"   zones expected / rendered : {expected} / {rendered}")
    ok = ok and rendered == expected
    for cluster, style in sorted(styles.items()):
        present = style["color"] in html
        print(f"   cluster {cluster} colour {style['color']} present: {present} "
              f"({style['n_zones']} zones, {style['short']})")
        ok = ok and present

    # Presence alone is not enough: a wrapped palette would put the same hex on two
    # clusters and still report "present" for both.
    colours = [s["color"] for s in styles.values()]
    distinct = len(set(colours)) == len(colours)
    print(f"   every cluster a distinct colour : {distinct}")
    if not distinct:
        dupes = {c for c in colours if colours.count(c) > 1}
        print(f"     COLLISION on {sorted(dupes)} — the map cannot be read")
    ok = ok and distinct

    labels = [s["short"] for s in styles.values()]
    unique_labels = len(set(labels)) == len(labels)
    print(f"   every cluster a distinct label  : {unique_labels}")
    ok = ok and unique_labels

    geo_ids = set(pd.read_parquet(config.ZONE_CENTROIDS_PARQUET)["zone_id"])
    missing_geom = set(forecaster.zones["zone_id"]) - geo_ids
    print(f"   zones without geometry    : {len(missing_geom)} {sorted(missing_geom) or ''}")
    ok = ok and not missing_geom

    # --- geojson --------------------------------------------------------------
    print(f"\n2. GeoJSON")
    for path in sorted(GEOJSON_DIR.glob("*.geojson")):
        payload = json.loads(path.read_text())
        valid_type = payload.get("type") == "FeatureCollection"
        n = len(payload.get("features", []))
        geometries = all(
            f.get("geometry", {}).get("type") in ("Polygon", "MultiPolygon")
            for f in payload["features"]
        )
        good = valid_type and n == expected and geometries
        ok = ok and good
        print(f"   {path.name:<34} {n:>4} features  "
              f"geometry {'ok' if geometries else 'BAD'}  "
              f"{'PASS' if good else 'FAIL'}")

    # --- plots read correctly against the numbers -----------------------------
    print(f"\n3. Plots agree with the numbers")
    sweep = pd.read_csv(config.KSWEEP_CSV)
    norm = sweep[sweep["variant"] == "normalized"]
    sil_peak = int(norm.loc[norm["silhouette"].idxmax(), "k"])
    k_sil = int(metadata.get("k_suggested_by_silhouette", -1))
    k_elbow = int(metadata.get("k_suggested_by_elbow", -1))
    chosen = int(metadata.get("chosen_k", -1))
    print(f"   silhouette peak in csv     : K={sil_peak}  (annotated K={k_sil})")
    print(f"   WCSS knee from metadata    : K={k_elbow}")
    print(f"   chosen K marked            : K={chosen}")
    # The chosen K is whatever the metadata records — it is a decision, not a
    # constant. An earlier version asserted chosen == 4 and failed the whole render
    # the moment the full-year run selected 5.
    n_clusters = len(styles)
    matches = chosen == n_clusters
    print(f"   clusters rendered          : {n_clusters} (matches chosen K: {matches})")
    ok = ok and sil_peak == k_sil and matches
    monotone = norm.sort_values("k")["wcss"].is_monotonic_decreasing
    print(f"   WCSS decreasing with K     : {monotone} "
          "(non-monotone steps are k-means|| restarts, not an error)")

    # --- heatmap --------------------------------------------------------------
    print(f"\n4. Heatmap")
    print(f"   matrix                     : {shares.shape[0]} zones x "
          f"{shares.shape[1]} hours")
    ok = ok and shares.shape == (expected, config.HOURS_PER_WEEK)
    row_sums = shares.sum(axis=1)
    print(f"   rows normalised to 1.0     : {bool(np.allclose(row_sums, 1.0))}")
    ok = ok and bool(np.allclose(row_sums, 1.0))
    grouped = order["cluster"].is_monotonic_increasing
    print(f"   rows grouped by cluster    : {grouped}")
    ok = ok and grouped

    # The visual claim the caption makes must actually hold.
    night = [d * 24 + h for d in (4, 5) for h in (23,)] + [
        d * 24 + h for d in (5, 6) for h in (0, 1, 2)
    ]
    evening = [d * 24 + h for d in range(5) for h in (17, 18)]
    per_cluster = shares.join(order["cluster"]).groupby("cluster")
    print(f"\n   {'cluster':<9}{'weekend-night share':>21}{'weekday-evening share':>23}")
    for cluster, group in per_cluster:
        block = group.drop(columns="cluster")
        print(f"   c{cluster:<8}{100 * block[night].sum(axis=1).mean():>20.2f}%"
              f"{100 * block[evening].sum(axis=1).mean():>22.2f}%")
    nightlife = max(styles, key=lambda c: styles[c]["label"].startswith("nightlife"))
    business = [c for c in styles if "business" in styles[c]["label"]]
    if business:
        n_share = per_cluster.get_group(nightlife).drop(columns="cluster")[night].sum(axis=1).mean()
        b_share = per_cluster.get_group(business[0]).drop(columns="cluster")[night].sum(axis=1).mean()
        claim = n_share > b_share
        print(f"   nightlife band brighter in weekend small hours than business band: {claim}")
        ok = ok and claim

    print("\n" + "=" * 78)
    print("  RESULT:", "PASS" if ok else "FAIL")
    print("=" * 78)
    return ok


def main() -> int:
    for path, hint in (
        (config.ZONE_CENTROIDS_PARQUET, "src.batch.geo_join"),
        (config.KSWEEP_CSV, "src.batch.train_kmeans"),
        (config.FEATURES_PARQUET, "src.batch.features"),
    ):
        if not Path(path).exists():
            print(f"ERROR: {path} missing — run: python -m {hint}", file=sys.stderr)
            return 2

    forecaster = Forecaster()
    metadata = forecaster.metadata
    chosen_k = int(metadata.get("chosen_k", 4))
    styles = cluster_style(forecaster)
    metrics = held_out_metrics()

    print("=" * 78)
    print("Rendering result figures")
    print("=" * 78)
    print(f"  zones {len(forecaster.zones)} | K={chosen_k} | "
          f"clusters {sorted(styles)}")

    outputs: list[Path] = []
    outputs.append(plot_k_selection(chosen_k, metadata))
    heatmap_path, shares, order = plot_heatmap(forecaster, styles)
    outputs.append(heatmap_path)
    outputs.append(build_map(forecaster, styles, metrics))
    outputs.extend(export_geojson(forecaster, styles, DEFAULT_WINDOWS))

    ok = validate(forecaster, styles, outputs, shares, order, metadata)

    print("\n" + "=" * 78)
    print("OUTPUT FILES")
    print("=" * 78)
    for path in outputs:
        size = path.stat().st_size
        rel = path.relative_to(config.ROOT)
        print(f"  {str(rel):<52} {size / 1024:>8,.1f} KB")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
