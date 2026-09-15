"""Germany 2018 wind-solar complementarity: small Streamlit demo over committed compact assets.

The app reads only files under app_assets/2018/. It performs no downloads and no GRIB decoding.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "app_assets" / "2018"

NOT_SHORTAGE = (
    "Low-output events are **resource-screening diagnostics** for a hypothetical equal-rated-capacity "
    "wind/solar portfolio. They are **not** electricity shortages, loss of load, or adequacy results: "
    "demand, storage, imports, transmission and dispatchable supply are not modelled."
)

st.set_page_config(page_title="From weather to renewable-energy signals: Germany 2018",
                   page_icon="🌬️", layout="wide", initial_sidebar_state="expanded")

# Consistent visual identity: one colour per physical quantity, one per data layer.
COLOR = {
    "wind": "#1f77b4", "solar": "#f28e2b", "hybrid": "#2ca02c", "observed": "#222222",
    "area": "#1f77b4", "capacity": "#9c755f",
}
LINE = {"observed": dict(color=COLOR["observed"], width=2.2), "area": dict(color=COLOR["area"], width=1.5),
        "capacity": dict(color=COLOR["capacity"], width=1.5), "wind": dict(color=COLOR["wind"], width=1.6),
        "solar": dict(color=COLOR["solar"], width=1.6), "hybrid": dict(color=COLOR["hybrid"], width=2.2)}
LABEL = {"area": "COSMO model · area-weighted", "capacity": "COSMO model · capacity-weighted (OPSD MW)",
         "observed": "SMARD observed", "wind": "Onshore wind", "solar": "Solar PV",
         "hybrid": "50/50 hybrid (hypothetical)"}
SHORT = {"area": "area-weighted", "capacity": "capacity-weighted"}
PLOT_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, Helvetica, Arial", size=13, color="#374151"),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, xanchor="left"),
    margin=dict(l=56, r=16, t=52, b=72),
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1.5rem; max-width: 1200px;}
      h1 {font-size: 2.0rem; font-weight: 700; letter-spacing: -0.01em; margin-bottom: 0.3rem;}
      h2 {font-size: 1.35rem; font-weight: 600; margin-top: 1.8rem; margin-bottom: 0.5rem; color: #1f2937;}
      h3 {font-size: 1.15rem; font-weight: 600; margin-top: 1.3rem; margin-bottom: 0.4rem; color: #374151;}
      h4 {font-size: 1.0rem; font-weight: 600; color: #4b5563;}
      div[data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
                                   padding: 0.6rem 0.8rem;}
      div[data-testid="stMetricLabel"] p {font-size: 0.78rem; color: #64748b; margin-bottom: 0.15rem;}
      div[data-testid="stMetricValue"] p {font-size: 1.05rem; font-weight: 600;}
      section[data-testid="stSidebar"] {background: #f8fafc;}
      .app-footer {color: #6b7280; font-size: 0.78rem; border-top: 1px solid #e5e7eb;
                   padding-top: 0.6rem; margin-top: 2rem;}
      .workflow {display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 0.6rem 0 1.0rem;}
      .wf {background: #eef4fb; border: 1px solid #cdd9e5; border-radius: 7px;
           padding: 5px 11px; font-size: 0.82rem; font-weight: 600; color: #1f3b57;}
      .wf-arrow {color: #64748b; font-weight: 700; padding: 0 2px;}
      .small {font-size: 0.85rem; color: #4b5563;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- data
@st.cache_data
def load_daily() -> pd.DataFrame:
    area = pd.read_parquet(ASSETS / "cosmo_rea6_dach_2018_daily_country_timeseries.parquet")
    area = area[area["country"] == "de"][["time", "wind_cf", "solar_cf", "combined_cf"]].copy()
    area["time"] = pd.to_datetime(area["time"]).dt.normalize()
    area = area.rename(columns={"wind_cf": "wind_area", "solar_cf": "solar_area", "combined_cf": "hybrid_area"})

    cap = pd.read_parquet(ASSETS / "cosmo_rea6_de_2018_daily_capacity_weighted.parquet")
    cap["time"] = pd.to_datetime(cap["time"]).dt.normalize()
    cap = cap.rename(columns={
        "wind_cf_capacity_weighted": "wind_capacity",
        "solar_cf_capacity_weighted": "solar_capacity",
        "combined_cf_capacity_weighted": "hybrid_capacity",
    })[["time", "wind_capacity", "solar_capacity", "hybrid_capacity"]]

    obs = pd.read_csv(ASSETS / "smard_de_2018_daily.csv", parse_dates=["date"])
    obs["time"] = pd.to_datetime(obs["date"]).dt.tz_localize(None).dt.normalize()
    obs = obs.rename(columns={"wind_onshore_cf": "wind_observed", "solar_cf": "solar_observed"})
    obs["hybrid_observed"] = 0.5 * obs["wind_observed"] + 0.5 * obs["solar_observed"]
    obs = obs[["time", "wind_observed", "solar_observed", "hybrid_observed"]]

    return area.merge(cap, on="time").merge(obs, on="time").sort_values("time").reset_index(drop=True)


@st.cache_data
def load_json(name: str) -> dict:
    with open(ASSETS / name, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_hourly() -> pd.DataFrame:
    df = pd.read_parquet(ASSETS / "cosmo_smard_de_201811_hourly_validation.parquet")
    df["interval_start_utc"] = pd.to_datetime(df["interval_start_utc"])
    return df.sort_values("interval_start_utc").reset_index(drop=True)


@st.cache_data
def load_n_germany_cells() -> int:
    fields = pd.read_parquet(ASSETS / "cosmo_rea6_dach_201811_november_field_means.parquet",
                             columns=["in_germany"])
    return int(fields["in_germany"].sum())


def metrics(observed: pd.Series, model: pd.Series) -> dict:
    err = model - observed
    return {
        "observed mean": observed.mean(),
        "model mean": model.mean(),
        "correlation": observed.corr(model),
        "bias": err.mean(),
        "MAE": err.abs().mean(),
        "RMSE": float(np.sqrt((err ** 2).mean())),
    }


def find_events(dates: pd.Series, values: pd.Series, threshold: float, min_days: int) -> pd.DataFrame:
    low = values.reset_index(drop=True) < threshold
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    groups = low.ne(low.shift()).cumsum()
    rows = []
    for _, idx in low[low].groupby(groups[low]).groups.items():
        idx = list(idx)
        if len(idx) >= min_days:
            seg = values.reset_index(drop=True).iloc[idx]
            rows.append({
                "start": dates.iloc[idx[0]].date(),
                "end": dates.iloc[idx[-1]].date(),
                "duration (days)": len(idx),
                "mean hybrid CF": round(seg.mean(), 3),
                "severity (CF-days)": round((threshold - seg).sum(), 3),
            })
    return pd.DataFrame(rows, columns=["start", "end", "duration (days)", "mean hybrid CF", "severity (CF-days)"])


def style_for(col: str) -> dict:
    """Pick a line style from the column name (layer wins over technology)."""
    for key in ("observed", "capacity", "area", "hybrid", "wind", "solar"):
        if key in col:
            return LINE[key]
    return dict(width=1.5)


def line_chart(df: pd.DataFrame, x: str, series: dict[str, str], y_title: str, title: str,
               height: int = 420, observed_mean: float | None = None,
               x_tickformat: str | None = None) -> go.Figure:
    """Consistent time-series figure: title top-left, legend below x-axis, units on axes."""
    fig = go.Figure()
    for col, label in series.items():
        line = style_for(col)
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col], mode="lines", name=label, line=line,
            hovertemplate=f"<b>%{{x}}</b><br>capacity factor: %{{y:.3f}}<extra>{label}</extra>",
        ))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.02, xanchor="left", y=0.98, yanchor="top",
                   font=dict(size=15)),
        yaxis_title=y_title,
        xaxis_title="",
        height=height,
        **PLOT_LAYOUT,
    )
    fig.update_yaxes(rangemode="tozero", gridcolor="#eef0f2", tickformat=".2f")
    fig.update_xaxes(showgrid=False)
    if x_tickformat:
        fig.update_xaxes(tickformat=x_tickformat)
    if observed_mean is not None:
        fig.add_hline(
            y=observed_mean, line_dash="dot", line_color=COLOR["observed"],
            annotation_text="observed mean", annotation_position="top right",
            annotation_font_size=11, annotation_font_color=COLOR["observed"],
        )
    return fig


def metric_table(rows: dict[str, dict], show_observed: bool = False) -> pd.DataFrame:
    table = pd.DataFrame(rows).T.round(3)
    if not show_observed and "observed mean" in table.columns:
        table = table.drop(columns="observed mean")
    return table


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """Render a row of metric cards: (label, value, help)."""
    cols = st.columns(len(items))
    for col, (label, value, helptext) in zip(cols, items):
        col.metric(label, value, help=helptext)


def section(title: str, subtitle: str | None = None) -> None:
    """Major section divider with optional explanatory subtitle."""
    st.divider()
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<p class='small'>{subtitle}</p>", unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        '<div class="app-footer">Data: DWD COSMO-REA6 · Bundesnetzagentur | SMARD.de (CC BY 4.0, underlying '
        'data ENTSO-E) · Open Power System Data plant register (ODbL). Code: MIT. The app reads only '
        'pre-computed <code>app_assets/2018/</code>; no downloads, no GRIB decoding.</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- pages
def page_overview() -> None:
    st.title("From weather to renewable-energy signals: Germany 2018")
    st.caption("How spatial weighting and observed-generation validation change our view of wind–solar variability")

    st.info(
        "**Can weather-derived wind and solar resource proxies reproduce observed German "
        "renewable-generation variability — and does accounting for where the installed fleet is located "
        "change the answer?**  "
        "*One year of daily data (2018) and one month of hourly data (November 2018).*"
    )

    daily = load_daily()
    n_de = load_n_germany_cells()
    m = {
        ("wind", "area"): metrics(daily["wind_observed"], daily["wind_area"]),
        ("wind", "capacity"): metrics(daily["wind_observed"], daily["wind_capacity"]),
        ("solar", "area"): metrics(daily["solar_observed"], daily["solar_area"]),
        ("solar", "capacity"): metrics(daily["solar_observed"], daily["solar_capacity"]),
    }
    hqc = load_json("cosmo_smard_de_201811_hourly_validation_qc.json")
    corr_ws = daily["wind_area"].corr(daily["solar_area"])
    monthly_mean = daily.groupby(daily["time"].dt.month)[["wind_area", "solar_area"]].transform("mean")
    anom_corr = (daily["wind_area"] - monthly_mean["wind_area"]).corr(
        daily["solar_area"] - monthly_mean["solar_area"])
    std_w = daily["wind_area"].std()
    std_s = daily["solar_area"].std()
    std_ha = daily["hybrid_area"].std()
    std_hc = daily["hybrid_capacity"].std()
    events_a = find_events(daily["time"], daily["hybrid_area"], 0.10, 2)
    events_c = find_events(daily["time"], daily["hybrid_capacity"], 0.10, 2)

    st.markdown("### Why this matters")
    st.markdown(
        "Wind and solar availability varies across Germany and across time. A spatially averaged weather "
        "signal describes the resource field, but an energy-system analysis should also ask what the existing "
        "installed fleet actually experiences."
    )

    st.markdown("### The analytical chain")
    st.caption("From atmospheric fields to a national stress test.")
    st.markdown(
        '<div class="workflow" style="margin-bottom:0.6rem;">'
        '<span class="wf" style="line-height:1.25;"><b>WEATHER</b><br>'
        '<span style="font-size:0.72em;color:#64748b;">COSMO-REA6</span></span>'
        '<span class="wf-arrow">→</span>'
        '<span class="wf" style="line-height:1.25;"><b>TECHNOLOGY PROXY</b><br>'
        '<span style="font-size:0.72em;color:#64748b;">Wind + Solar CF</span></span>'
        '<span class="wf-arrow">→</span>'
        '<span class="wf" style="line-height:1.25;"><b>SPATIAL WEIGHTING</b><br>'
        '<span style="font-size:0.72em;color:#64748b;">Area vs capacity</span></span>'
        '<span class="wf-arrow">→</span>'
        '<span class="wf" style="line-height:1.25;"><b>REALITY CHECK</b><br>'
        '<span style="font-size:0.72em;color:#64748b;">SMARD observed</span></span>'
        '<span class="wf-arrow">→</span>'
        '<span class="wf" style="line-height:1.25;"><b>PORTFOLIO</b><br>'
        '<span style="font-size:0.72em;color:#64748b;">Wind + Solar</span></span>'
        '<span class="wf-arrow">→</span>'
        '<span class="wf" style="line-height:1.25;"><b>STRESS TEST</b><br>'
        '<span style="font-size:0.72em;color:#64748b;">Low-output + hourly</span></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Question.** What can atmospheric reanalysis tell us about renewable generation?")
    st.markdown(
        "**Method.** Convert COSMO-REA6 wind and radiation fields into simplified technology-specific "
        "capacity-factor (CF) proxies."
    )
    st.caption(
        "**Evidence.** November 2018 monthly-mean COSMO-REA6 input fields over the analysis domain "
        f"({n_de:,} cells inside the Germany mask)."
    )
    st.image(str(ASSETS / "fig_201811_field_means.png"), use_container_width=True)
    with st.expander("About the input fields"):
        st.caption(
            "Temperature, wind speed, direct and diffuse shortwave irradiance. These are the atmospheric "
            "inputs that the turbine and PV conversion curves turn into capacity factors."
        )
    st.markdown(
        "**So what?** The result is a spatially resolved resource proxy — not turbine SCADA data — that can "
        "be compared with the installed fleet and with national observations."
    )

    section("Three representations of the same 2018 weather")
    st.markdown(
        "The app keeps three layers in front of the viewer at all times. They are **not three equivalent "
        "datasets** — they answer different questions."
    )

    v1, v2, v3 = st.columns(3)
    with v1:
        with st.container(border=True):
            st.markdown("**1. Area-weighted COSMO**")
            st.caption("Resource view")
            st.markdown("*What does the spatially averaged renewable resource look like?*")
    with v2:
        with st.container(border=True):
            st.markdown("**2. Capacity-weighted COSMO**")
            st.caption("Fleet view")
            st.markdown("*What does the resource look like where Germany's installed fleet is located?*")
    with v3:
        with st.container(border=True):
            st.markdown("**3. SMARD observed**")
            st.caption("Reality check")
            st.markdown("*How does the model-derived signal compare with observed national generation?*")

    st.markdown("**Same question, different representations of the system.**")
    st.caption(
        "Area weighting → the geographical resource  ·  "
        "Capacity weighting → the installed-fleet geography  ·  "
        "SMARD → the observed national generation"
    )
    st.caption(
        "COSMO is a weather-derived proxy, not turbine SCADA data. The proxy is **not fitted** to SMARD. "
        "Area weighting uses `cos(latitude)`. Capacity weighting uses mapped installed MW from the OPSD 2018 "
        "plant register. SMARD is measured grid-fed generation divided by fixed year-end capacity."
    )

    section("Why spatial weighting matters")
    st.markdown("**Question.** Does where the installed fleet is located change the national resource signal?")
    st.markdown(
        "**Method.** Map every OPSD plant with valid coordinates to the nearest COSMO cell and weight by "
        "its share of national MW. Compare with an area-weighted `cos(latitude)` average."
    )
    st.caption(
        "**Evidence.** Installed MW per COSMO grid cell (OPSD 2018). Colour is log(1 + MW) with a fixed "
        "0–5 MW scale. Wind clusters in the north and along coasts; PV in the south and west."
    )
    st.image(str(ASSETS / "fig_capacity_maps.png"), use_container_width=True)
    with st.expander("About the map"):
        st.caption("Coverage: ~80% of wind MW, ~97% of PV MW. Weights are a static 2018 snapshot.")
    st.markdown(
        '**So what?** Fleet geography matters because the resource the installed capacity "sees" is not the '
        'same as the resource the land area "sees".'
    )

    section("Validation — does the proxy reproduce observed generation?")
    st.markdown(
        "We first test the proxy against national SMARD generation. This tells us whether the weather-derived "
        "signal captures the timing and variability of observed generation well enough to support the "
        "subsequent diagnostics."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            line_chart(
                daily, "time",
                {"wind_area": LABEL["area"], "wind_capacity": LABEL["capacity"], "wind_observed": LABEL["observed"]},
                "Capacity factor (0–1)",
                "Daily wind CF: weather proxy vs. observed generation",
                height=380,
                observed_mean=m[("wind", "area")]["observed mean"],
                x_tickformat="%b",
            ),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            line_chart(
                daily, "time",
                {"solar_area": LABEL["area"], "solar_capacity": LABEL["capacity"], "solar_observed": LABEL["observed"]},
                "Capacity factor (0–1)",
                "Daily solar PV CF: weather proxy vs. observed generation",
                height=380,
                observed_mean=m[("solar", "area")]["observed mean"],
                x_tickformat="%b",
            ),
            use_container_width=True,
        )
    st.markdown("**How to read the validation metrics**")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown("**Correlation**")
        st.caption("Does the proxy get the *timing* right?")
    with q2:
        st.markdown("**Bias**")
        st.caption("Does the proxy get the *level* right?")
    with q3:
        st.markdown("**MAE**")
        st.caption("How large are typical errors?")
    with q4:
        st.markdown("**RMSE**")
        st.caption("How large are larger errors?")

    rows = {f"{t} / {SHORT[w]}": v for (t, w), v in m.items()}
    with st.expander("Full validation metrics table"):
        st.dataframe(metric_table(rows), use_container_width=True, hide_index=False)

    st.info(
        "**What changes when we account for the installed fleet?** "
        f"Wind mean bias shifts from {m[('wind','area')]['bias']:+.3f} (area) to "
        f"{m[('wind','capacity')]['bias']:+.3f} (capacity). Wind correlation is still high "
        f"({m[('wind','area')]['correlation']:.3f} → {m[('wind','capacity')]['correlation']:.3f}), but "
        f"RMSE barely changes ({m[('wind','area')]['RMSE']:.3f} → {m[('wind','capacity')]['RMSE']:.3f}). "
        "Solar changes are small because PV is already widely dispersed."
    )
    st.success(
        "**So what?** Capacity weighting changes the representation of the national wind signal "
        "substantially, but better fleet geography does not automatically produce better agreement across "
        "every metric."
    )

    section("What the analysis finds")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown("**Weather → useful national signal**")
        st.caption(
            f"Solar correlation ≈ {m[('solar','area')]['correlation']:.2f}; wind ≈ "
            f"{m[('wind','area')]['correlation']:.2f}. The proxies capture much of the observed variability."
        )
    with f2:
        st.markdown("**Fleet geography matters**")
        st.caption(
            f"Wind bias shrinks from {m[('wind','area')]['bias']:+.3f} to "
            f"{m[('wind','capacity')]['bias']:+.3f} CF, but does not uniformly improve skill."
        )
    with f3:
        st.markdown("**Complementarity, not adequacy**")
        st.caption(
            f"Wind and solar are negatively correlated (r = {corr_ws:.2f}), but "
            f"{len(events_c)} low-output events still occur under capacity weighting."
        )
    with f4:
        st.markdown("**Daily averages hide ramps**")
        st.caption(
            "November hourly: solar ramp agreement is strong; wind ramp agreement is much weaker."
        )

    section("Complementarity and low-output events")
    st.markdown(
        "**Question.** If wind and solar are negatively correlated, does combining them reduce low-output "
        "periods in a hypothetical 50/50 rated-capacity portfolio?"
    )
    st.markdown(
        "**Method.** Examine the daily wind–solar relationship, the smoothness of a 50/50 portfolio, and the "
        "frequency of sustained hybrid CF < 0.10 events."
    )
    daily["season"] = daily["time"].dt.month.map(
        {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
         6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
    fig = px.scatter(
        daily, x="wind_area", y="solar_area", color="season", opacity=0.75,
        labels={"wind_area": "Wind capacity factor (0–1)",
                "solar_area": "Solar capacity factor (0–1)", "season": "Season"},
        title="Wind vs solar daily CF: evidence of complementarity",
        category_orders={"season": ["DJF", "MAM", "JJA", "SON"]},
        color_discrete_sequence=["#4e79a7", "#59a14f", "#f28e2b", "#b07aa1"],
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.4, color="white")))
    fig.update_layout(
        height=380,
        **{**PLOT_LAYOUT, "hovermode": "closest"},
        title=dict(text="<b>Wind vs solar daily CF: evidence of complementarity</b>", x=0.02, xanchor="left", y=0.98,
                   yanchor="top", font=dict(size=15)),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Daily wind and solar are negatively correlated, but both can still be low at the same time. "
        "→ **See page 3: Does combining wind and solar reduce low-output periods?**"
    )

    st.markdown(
        "**Sustained low-output periods remain possible in a hypothetical 50/50 rated-capacity portfolio:** "
        f"capacity-weighted hybrid CF < 0.10 for ≥2 days produces **{len(events_c)} events**, longest "
        f"**{int(events_c['duration (days)'].max()) if len(events_c) else 0} days**, severity "
        f"**{events_c['severity (CF-days)'].sum():.2f} CF-days**. → **See page 3 for the interactive catalogue.**"
    )
    st.warning(
        "**Important:** these are resource-screening diagnostics for a hypothetical 50/50 rated-capacity "
        "portfolio. They are **NOT** electricity shortages, loss-of-load, or adequacy results. Demand, storage, "
        "imports, transmission, dispatchable generation and market behaviour are not modelled."
    )
    st.markdown(
        "**So what?** Combining technologies changes the variability of the aggregate resource, which "
        "motivates examining sustained low-output periods — but the diagnostics do not represent electricity "
        "shortages."
    )

    section("Hourly resolution: a November stress test")
    st.markdown("**Question.** What does daily aggregation hide about timing and ramp behaviour?")
    st.markdown(
        "**Method.** Compare hourly COSMO proxies with SMARD observations for the whole month of November 2018. "
        "Daily data reveal seasonal and sustained variability; hourly data reveal timing and ramps."
    )
    st.caption(
        "**November 2018 only.** This is a one-month higher-resolution case study, not a full-year hourly validation."
    )
    ramp_w_a = hqc["ramp_validation"]["area"]["wind"]["ramp_correlation"]
    ramp_w_c = hqc["ramp_validation"]["capacity"]["wind"]["ramp_correlation"]
    ramp_s = hqc["ramp_validation"]["area"]["solar"]["ramp_correlation"]
    kpi_row([
        ("Solar hourly correlation",
         f"{hqc['validation_metrics']['area']['solar']['correlation']:.3f}",
         "Area-weighted; capacity-weighted similar"),
        ("Solar ramp correlation", f"{ramp_s:.2f}", "CF change per hour"),
        ("Wind hourly correlation",
         f"{hqc['validation_metrics']['area']['wind']['correlation']:.3f} / "
         f"{hqc['validation_metrics']['capacity']['wind']['correlation']:.3f}",
         "Area / capacity weighting"),
        ("Wind ramp correlation", f"{ramp_w_a:.2f} / {ramp_w_c:.2f}",
         "Hour-to-hour wind changes are weak — visible only at hourly resolution"),
    ])
    st.caption(
        "Solar levels, timing and ramps are reproduced strongly at hourly resolution. Wind ramp representation "
        "is much weaker, and capacity weighting does **not** solve it. → **See page 5: Do daily averages hide "
        "hourly behaviour?**"
    )
    st.markdown(
        "**So what?** Daily averages can conceal timing and ramp behaviour that matters at shorter "
        "timescales, particularly for wind."
    )

    section("What we learned — and what this does not answer")
    st.success(
        "**What the analysis supports.** The weather-derived proxies reproduce much of the observed temporal "
        "variability, especially for solar. Capacity weighting substantially changes the wind mean level, but "
        "does not automatically improve every validation metric. Wind and solar are negatively correlated "
        f"(r ≈ {corr_ws:.2f}), so a combined portfolio is smoother than either source alone, yet sustained "
        "low-output periods still occur. The hourly case study reveals that daily averages can hide ramp "
        "behaviour, particularly for wind."
    )
    st.info(
        "**What this analysis does not answer:** future climate impacts, electricity-system adequacy, storage "
        "sizing, grid congestion, bankable energy-yield forecasts, plant-level performance, or real-time "
        "forecasting skill. Low-output events are resource-screening diagnostics for a hypothetical equal-rated "
        "portfolio, not estimates of electricity shortages."
    )
    footer()


def page_daily() -> None:
    st.title("Does the weather proxy reproduce observed generation?")
    st.info(
        "**We first test the weather-derived capacity-factor proxy against national SMARD generation.** "
        "This tells us whether the weather signal captures the timing and variability of observed generation "
        "well enough to support the subsequent diagnostics."
    )
    st.caption("Compare wind, solar and a hypothetical 50/50 hybrid for area-weighted, capacity-weighted and SMARD-observed daily series.")
    daily = load_daily()
    cap_qc = load_json("cosmo_rea6_de_2018_capacity_weighting_qc.json")
    wind_cap = int(cap_qc["published_end_2018_capacity_mw"]["wind"])
    solar_cap = int(cap_qc["published_end_2018_capacity_mw"]["solar"])

    c1, c2 = st.columns([1, 2])
    tech = c1.radio("Technology", ["wind", "solar", "hybrid"], horizontal=True, format_func=LABEL.get)
    layers = c2.multiselect("Series", ["area", "capacity", "observed"], default=["area", "capacity", "observed"],
                            format_func=LABEL.get)
    if tech == "hybrid":
        st.info(
            "Observed hybrid = 0.5 × observed wind CF + 0.5 × observed solar CF. It is a normalised "
            "reference portfolio, not Germany's actual combined generation."
        )

    observed_mean = daily[f"{tech}_observed"].mean() if f"{tech}_observed" in daily.columns else None
    series = {f"{tech}_{layer}": LABEL[layer] for layer in layers}
    fig = line_chart(
        daily, "time", series, "Capacity factor (0–1)",
        f"Daily {LABEL[tech]} CF: model vs. observed", height=420,
        observed_mean=observed_mean, x_tickformat="%b",
    )
    st.plotly_chart(fig, use_container_width=True)

    section("Validation metrics", "Model proxy vs. SMARD observed")
    rows = {}
    if "area" in layers:
        rows[SHORT["area"]] = metrics(daily[f"{tech}_observed"], daily[f"{tech}_area"])
    if "capacity" in layers:
        rows[SHORT["capacity"]] = metrics(daily[f"{tech}_observed"], daily[f"{tech}_capacity"])
    if rows:
        first = next(iter(rows.values()))
        kpi_row([("SMARD observed mean CF", f"{first['observed mean']:.3f}", "0–1, dimensionless")] +
                [(f"{name} — r / bias",
                  f"{v['correlation']:.3f} / {v['bias']:+.3f}",
                  "correlation unitless; bias in CF units") for name, v in rows.items()])
        st.dataframe(metric_table(rows), use_container_width=True, hide_index=False)
        interp = []
        if SHORT["area"] in rows:
            a = rows[SHORT["area"]]
            interp.append(
                f"area weighting: bias {a['bias']:+.3f}, r {a['correlation']:.3f}, RMSE {a['RMSE']:.3f}")
        if SHORT["capacity"] in rows:
            c = rows[SHORT["capacity"]]
            interp.append(
                f"capacity weighting: bias {c['bias']:+.3f}, r {c['correlation']:.3f}, RMSE {c['RMSE']:.3f}")
        if interp:
            st.success(
                "**Result.** " + "; ".join(interp) +
                ". Capacity weighting mainly improves the wind mean level; solar timing is already strong."
            )
    if tech == "hybrid":
        st.caption("Hybrid metrics compare the hypothetical equal-capacity model portfolio against the equally "
                   "hypothetical observed portfolio; they are not a validation of real combined generation.")
    st.caption("All metrics in dimensionless CF units (0–1); correlation is unitless. Correlation measures "
               f"timing/co-variability; bias, MAE and RMSE measure level and error magnitude. High correlation "
               f"does not imply an unbiased level. Observed CF uses fixed end-2018 capacity "
               f"({wind_cap:,} MW onshore wind; {solar_cap:,} MW PV).")

    section("Monthly mean daily CF")
    monthly = daily.set_index("time")[[c for c in daily.columns if c.startswith(tech)]].resample("MS").mean()
    monthly.columns = [LABEL[c.replace(f"{tech}_", "")] for c in monthly.columns]
    monthly.index = monthly.index.strftime("%b")
    st.dataframe(monthly.round(3).T, use_container_width=True, hide_index=False)

    st.info(
        "**Limitation.** Daily national means and fixed year-end capacities do not capture plant availability, "
        "curtailment, storage, transmission or grid constraints."
    )
    footer()


def page_complementarity() -> None:
    st.title("Does combining wind and solar reduce low-output periods?")
    st.info(
        "**If wind and solar are less correlated than either resource with itself, combining them can smooth "
        "the aggregate renewable signal. This page examines both statistical complementarity and sustained "
        "low-output periods in a hypothetical 50/50 rated-capacity portfolio.**"
    )
    st.caption("Relationship → seasonal behaviour → portfolio mix → sustained low-output periods.")
    daily = load_daily()
    st.warning(NOT_SHORTAGE)

    weighting = st.radio("Weighting", ["area", "capacity"], horizontal=True, format_func=LABEL.get)
    w, s = f"wind_{weighting}", f"solar_{weighting}"
    monthly_mean = daily.groupby(daily["time"].dt.month)[[w, s]].transform("mean")
    anom_corr = (daily[w] - monthly_mean[w]).corr(daily[s] - monthly_mean[s])
    shares = np.arange(0, 1.01, 0.1)
    sweep = pd.DataFrame({
        "wind share": shares,
        "mean CF": [(a * daily[w] + (1 - a) * daily[s]).mean() for a in shares],
        "std": [(a * daily[w] + (1 - a) * daily[s]).std() for a in shares],
        "5th percentile": [(a * daily[w] + (1 - a) * daily[s]).quantile(0.05) for a in shares],
    }).set_index("wind share")

    section("Relationship and seasonal behaviour")
    kpi_row([
        ("Full-year correlation", f"{daily[w].corr(daily[s]):.3f}", "Includes opposing seasonal cycles"),
        ("Monthly-anomaly correlation", f"{anom_corr:.3f}", "Weather-scale complementarity"),
    ])

    daily["season"] = daily["time"].dt.month.map(
        {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
         6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
    fig = px.scatter(
        daily, x=w, y=s, color="season", opacity=0.75,
        labels={w: "Wind capacity factor (0–1)", s: "Solar capacity factor (0–1)", "season": "Season"},
        title="Wind vs solar daily CF: evidence of complementarity",
        category_orders={"season": ["DJF", "MAM", "JJA", "SON"]},
        color_discrete_sequence=["#4e79a7", "#59a14f", "#f28e2b", "#b07aa1"],
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.4, color="white")),
                      hovertemplate="<b>%{x:.3f}</b> wind<br><b>%{y:.3f}</b> solar<extra>%{data.name}</extra>")
    fig.update_layout(
        height=420,
        **{**PLOT_LAYOUT, "hovermode": "closest"},
        title=dict(text="<b>Wind vs solar daily CF: evidence of complementarity</b>", x=0.02, xanchor="left", y=0.98,
                   yanchor="top", font=dict(size=15)),
        xaxis=dict(tickformat=".2f", showgrid=False),
        yaxis=dict(tickformat=".2f", gridcolor="#eef0f2"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Negative correlation means one source tends to offset the other, but both can still be low.")

    section("Portfolio mix sensitivity")
    c1, c2 = st.columns([1, 1])
    with c1:
        kpi_row([
            ("Std, 100% wind → 50/50", f"{sweep.loc[1.0, 'std']:.3f} → {sweep.loc[0.5, 'std']:.3f}",
             "Daily CF portfolio std"),
            ("Std, 100% solar → 50/50", f"{sweep.loc[0.0, 'std']:.3f} → {sweep.loc[0.5, 'std']:.3f}",
             "Daily CF portfolio std"),
        ])
    with c2:
        st.markdown(
            "A 50/50 portfolio is smoother than either source alone, but the 5th-percentile line shows that "
            "the low-output tail remains under every mix."
        )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep.index, y=sweep["mean CF"], mode="lines+markers", name="mean",
                             line=dict(color=COLOR["hybrid"], width=2)))
    fig.add_trace(go.Scatter(x=sweep.index, y=sweep["std"], mode="lines+markers", name="std",
                             line=dict(color=COLOR["capacity"], width=2)))
    fig.add_trace(go.Scatter(x=sweep.index, y=sweep["5th percentile"], mode="lines+markers",
                             name="5th percentile", line=dict(color=COLOR["wind"], width=2, dash="dot")))
    fig.update_layout(
        title=dict(text="<b>Portfolio mix: how does a 50/50 blend change variability?</b>", x=0.02, xanchor="left", y=0.98, yanchor="top",
                   font=dict(size=15)),
        xaxis_title="Wind share of rated capacity (0 = all solar, 1 = all wind)",
        yaxis_title="Daily capacity factor (0–1)",
        height=360,
        **PLOT_LAYOUT,
    )
    fig.update_xaxes(tickformat=".1f", showgrid=False)
    fig.update_yaxes(rangemode="tozero", gridcolor="#eef0f2", tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Statistical sensitivity, not an optimisation. The 5th-percentile line shows the low-output tail.")

    section("Sustained low-output periods in the hypothetical 50/50 portfolio")
    st.markdown(
        "Sustained low-output periods are identified for a hypothetical 50/50 rated-capacity portfolio. "
        "The sliders show how sensitive the event catalogue is to the threshold and minimum-duration definition."
    )
    c3, c4 = st.columns(2)
    threshold = c3.slider("Hybrid CF threshold", 0.05, 0.20, 0.10, 0.01)
    min_days = c4.slider("Minimum consecutive days", 1, 5, 2)

    hyb = 0.5 * daily[w] + 0.5 * daily[s]
    events = find_events(daily["time"], hyb, threshold, min_days)
    kpi_row([
        ("Events", f"{len(events)}", "CF < threshold for ≥ min days"),
        ("Longest event", f"{int(events['duration (days)'].max()) if len(events) else 0} days", None),
        ("Total severity", f"{events['severity (CF-days)'].sum() if len(events) else 0.0:.2f} CF-days",
         "Sum over event days of (threshold − hybrid CF)"),
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["time"], y=hyb, mode="lines", name=LABEL["hybrid"], line=LINE["hybrid"],
                             hovertemplate="<b>%{x}</b><br>hybrid CF: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=threshold, line_dash="dash", line_color=COLOR["capacity"],
                  annotation_text=f"threshold {threshold:.2f}", annotation_position="top left",
                  annotation_font_size=11, annotation_font_color=COLOR["capacity"])
    for _, ev in events.iterrows():
        fig.add_vrect(x0=str(ev["start"]),
                      x1=str(pd.Timestamp(ev["end"]) + pd.Timedelta(1, "D")), fillcolor=COLOR["capacity"],
                      opacity=0.12, line_width=0)
    fig.update_layout(
        title=dict(text=f"<b>Hypothetical 50/50 hybrid CF</b> — {LABEL[weighting].lower()}", x=0.02,
                   xanchor="left", y=0.98, yanchor="top", font=dict(size=15)),
        yaxis_title="Capacity factor (0–1)",
        height=420,
        **PLOT_LAYOUT,
    )
    fig.update_yaxes(rangemode="tozero", gridcolor="#eef0f2", tickformat=".2f")
    fig.update_xaxes(showgrid=False, tickformat="%b")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(events.sort_values("duration (days)", ascending=False), use_container_width=True, hide_index=True)
    st.success(
        "**Result.** Wind and solar are negatively correlated, so combining them smooths the daily signal, "
        "but sustained low-output periods still occur. The event count is sensitive to the threshold and "
        "minimum-duration definition."
    )
    st.warning(NOT_SHORTAGE)
    footer()


def page_capacity() -> None:
    st.title("Does the installed fleet change the answer?")
    st.markdown(
        "**Question.** Does mapping the installed renewable fleet onto the weather grid change the national "
        "resource signal?"
    )
    st.caption("Compare area-weighted, capacity-weighted and SMARD-observed daily capacity factors for wind and solar.")
    daily = load_daily()
    qc = load_json("cosmo_rea6_de_2018_capacity_weighting_qc.json")

    st.markdown(
        "**Method.** Both layers start from the same COSMO-REA6 capacity-factor fields; only the cell weights "
        "differ. Area weighting uses `cos(latitude)` to ask what weather German land experienced. Capacity "
        "weighting asks what weather the mapped installed fleet experienced."
    )

    st.markdown(
        "**From geography → to the installed fleet → to national generation.** The map shows where renewable "
        "capacity is actually located. The validation then tests whether that fleet-weighted signal is closer to "
        "observed national generation."
    )
    section("Mapped installed capacity")
    st.caption(
        "**Evidence.** Installed MW per COSMO grid cell (OPSD 2018). Colour is log(1 + MW) with a fixed 0–5 MW scale. "
        "Wind clusters in the north and along coasts; PV in the south and west."
    )
    st.image(str(ASSETS / "fig_capacity_maps.png"), use_container_width=True)

    st.markdown("### OPSD mapped-capacity coverage")
    cov = pd.DataFrame({
        "plant records": qc["plant_records"],
        "mapped MW": qc["mapped_capacity_mw"],
        "published end-2018 MW": qc["published_end_2018_capacity_mw"],
        "coverage": qc["mapped_capacity_coverage_fraction"],
        "occupied grid cells": qc["occupied_grid_cells"],
    }).loc[["wind", "solar"]]
    st.dataframe(cov, use_container_width=True, hide_index=False)
    st.caption("Wind coverage (~80%) is materially weaker than solar (~97%). Weights are a static 2018 snapshot.")

    section("Validation against SMARD", "Area-weighted vs. capacity-weighted vs. observed")
    wm = metrics(daily["wind_observed"], daily["wind_area"])
    wc = metrics(daily["wind_observed"], daily["wind_capacity"])
    sa = metrics(daily["solar_observed"], daily["solar_area"])
    sc = metrics(daily["solar_observed"], daily["solar_capacity"])
    kpi_row([
        ("Wind bias: area → capacity",
         f"{wm['bias']:+.3f} → {wc['bias']:+.3f}", "Mean bias largely removed"),
        ("Wind r / RMSE: area → capacity",
         f"{wm['correlation']:.3f} / {wm['RMSE']:.3f} → {wc['correlation']:.3f} / {wc['RMSE']:.3f}",
         "Daily skill changes little"),
        ("Solar bias: area → capacity",
         f"{sa['bias']:+.3f} → {sc['bias']:+.3f}", "PV is dispersed; changes are small"),
        ("Solar r / RMSE: area → capacity",
         f"{sa['correlation']:.3f} / {sa['RMSE']:.3f} → {sc['correlation']:.3f} / {sc['RMSE']:.3f}",
         "Already strong under area weighting"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            line_chart(
                daily, "time",
                {"wind_area": LABEL["area"], "wind_capacity": LABEL["capacity"], "wind_observed": LABEL["observed"]},
                "Capacity factor (0–1)",
                "Daily wind CF: area vs. capacity vs. observed", height=380,
                observed_mean=wm["observed mean"], x_tickformat="%b",
            ),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            line_chart(
                daily, "time",
                {"solar_area": LABEL["area"], "solar_capacity": LABEL["capacity"], "solar_observed": LABEL["observed"]},
                "Capacity factor (0–1)",
                "Daily solar PV CF: area vs. capacity vs. observed", height=380,
                observed_mean=sa["observed mean"], x_tickformat="%b",
            ),
            use_container_width=True,
        )

    rows = {"wind / area": wm, "wind / capacity": wc, "solar / area": sa, "solar / capacity": sc}
    st.caption(f"SMARD observed annual mean CF: wind {wm['observed mean']:.3f}, solar {sa['observed mean']:.3f} "
               "(0–1). All metrics dimensionless CF units; correlation unitless.")
    st.dataframe(metric_table(rows), use_container_width=True, hide_index=False)
    st.success(
        "**Result.** Capacity weighting removes most of the wind mean bias, showing that installed wind "
        "geography matters. It does not uniformly improve daily skill: wind correlation and RMSE change only "
        "slightly, and solar changes are small because PV is geographically dispersed."
    )

    section("Effect on low-output screening")
    ea = find_events(daily["time"], daily["hybrid_area"], 0.10, 2)
    ec = find_events(daily["time"], daily["hybrid_capacity"], 0.10, 2)
    st.dataframe(pd.DataFrame({
        LABEL["area"]: [len(ea), int(ea["duration (days)"].max()), round(ea["severity (CF-days)"].sum(), 2)],
        LABEL["capacity"]: [len(ec), int(ec["duration (days)"].max()), round(ec["severity (CF-days)"].sum(), 2)],
    }, index=["events", "longest (days)", "total severity (CF-days)"]), use_container_width=True, hide_index=False)
    st.caption(NOT_SHORTAGE)

    st.info(
        "**Limitation.** Capacity weights are a static 2018 OPSD snapshot. Wind coverage is only ~80%, so the "
        "capacity-weighted wind signal represents coordinate-valid plants, not necessarily the whole fleet."
    )
    footer()


def page_hourly() -> None:
    st.title("Do daily averages hide important hourly behaviour?")
    st.info(
        "**The annual analysis uses daily data to understand variability and sustained low-output periods. "
        "But daily averaging can hide intraday timing and ramp behaviour. November 2018 is therefore used as a "
        "higher-resolution case study.**"
    )
    st.caption("Hourly model–observed comparison for levels, ramps and low-output hours.")
    st.warning(
        "**One month only.** This is a resolution-sensitivity case study for November 2018, not a full-year "
        "hourly validation. 720 hours are aligned exactly in UTC (COSMO hour-ending labels minus 1 h)."
    )
    h = load_hourly()
    qc = load_json("cosmo_smard_de_201811_hourly_validation_qc.json")
    cqc = load_json("cosmo_rea6_de_201811_hourly_qc.json")

    section("Headline metrics")
    ramps = qc["ramp_validation"]
    kpi_row([
        ("Solar hourly correlation", f"{qc['validation_metrics']['area']['solar']['correlation']:.3f}",
         "Area-weighted; capacity-weighted similar"),
        ("Solar ramp correlation", f"{ramps['area']['solar']['ramp_correlation']:.3f}",
         "CF change per hour"),
        ("Wind hourly correlation",
         f"{qc['validation_metrics']['area']['wind']['correlation']:.3f} / "
         f"{qc['validation_metrics']['capacity']['wind']['correlation']:.3f}", "Area / capacity weighting"),
        ("Wind ramp correlation",
         f"{ramps['area']['wind']['ramp_correlation']:.3f} / "
         f"{ramps['capacity']['wind']['ramp_correlation']:.3f}",
         "Hour-to-hour wind changes are weak"),
    ])
    st.success(
        "**Result.** Solar levels, timing and ramps are reproduced strongly at hourly resolution. Capacity "
        "weighting improves wind level and correlation but does **not** solve the wind-ramp representation. "
        "These hourly comparisons do not fit or scale the proxies to SMARD; notebook 04 explores a same-year "
        "mean-ratio scaling separately as a diagnostic sensitivity experiment."
    )

    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between;'
        'margin:0.5rem 0 1rem 0;">'
        '<div style="flex:1;min-width:120px;text-align:center;background:#f8fafc;border:1px solid #e2e8f0;'
        'border-radius:6px;padding:5px 4px;font-size:0.82rem;font-weight:600;">DAILY VIEW</div>'
        '<div style="color:#64748b;font-weight:700;">→</div>'
        '<div style="flex:1;min-width:120px;text-align:center;background:#f8fafc;border:1px solid #e2e8f0;'
        'border-radius:6px;padding:5px 4px;font-size:0.82rem;font-weight:600;">HOURLY VIEW</div>'
        '<div style="color:#64748b;font-weight:700;">→</div>'
        '<div style="flex:1;min-width:120px;text-align:center;background:#f8fafc;border:1px solid #e2e8f0;'
        'border-radius:6px;padding:5px 4px;font-size:0.82rem;font-weight:600;">TIMING</div>'
        '<div style="color:#64748b;font-weight:700;">→</div>'
        '<div style="flex:1;min-width:120px;text-align:center;background:#f8fafc;border:1px solid #e2e8f0;'
        'border-radius:6px;padding:5px 4px;font-size:0.82rem;font-weight:600;">RAMPS</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    section("Hourly capacity factor")
    c1, c2 = st.columns([1, 2])
    tech = c1.radio("Technology", ["wind", "solar", "hybrid"], horizontal=True, format_func=LABEL.get)
    window = c2.select_slider("Window", options=["Full month", "1-9 November (fixed event window)"],
                              value="Full month")
    df = h if window == "Full month" else h[h["interval_start_utc"].between("2018-11-01", "2018-11-09 23:00")]
    fig = line_chart(
        df, "interval_start_utc",
        {f"observed_{tech}_cf": LABEL["observed"], f"{tech}_cf_area": LABEL["area"], f"{tech}_cf_capacity": LABEL["capacity"]},
        "Capacity factor (0–1)",
        f"Hourly {LABEL[tech]} CF: model vs. observed, UTC", height=420,
    )
    if tech == "hybrid":
        fig.add_hline(y=0.10, line_dash="dash", line_color=COLOR["capacity"], annotation_text="0.10",
                      annotation_position="top left", annotation_font_size=11,
                      annotation_font_color=COLOR["capacity"])
    st.plotly_chart(fig, use_container_width=True)

    section("Hourly timing and ramp agreement")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Level and timing")
        rows = []
        for wgt, d in qc["validation_metrics"].items():
            for t, mm in d.items():
                rows.append({"weighting": wgt, "technology": t, **{k: v for k, v in mm.items() if k != "n"}})
        st.dataframe(pd.DataFrame(rows).set_index(["weighting", "technology"]).round(3), use_container_width=True, hide_index=False)
        st.caption("CF values dimensionless (0–1); correlation unitless; bias/MAE/RMSE in CF units.")
    with c2:
        st.markdown("#### Ramps (719 consecutive-hour pairs)")
        rows = []
        for wgt, d in ramps.items():
            for t, mm in d.items():
                rows.append({"weighting": wgt, "technology": t,
                             "ramp correlation": mm["ramp_correlation"], "ramp MAE": mm["ramp_mae"],
                             "ramp RMSE": mm["ramp_rmse"]})
        st.dataframe(pd.DataFrame(rows).set_index(["weighting", "technology"]).round(3), use_container_width=True, hide_index=False)
        st.caption("Ramps = CF change per hour (0–1); correlation unitless.")
    st.markdown(
        "**So what?** Solar timing and ramp agreement are strong. Wind ramp correlation is much weaker, and "
        "capacity weighting does **not** solve it — an honest sign that the proxy's hourly wind dynamics are not fully "
        "captured."
    )

    section("Hourly vs. daily processing routes")
    st.markdown(
        "Converting hourly wind to CF and then averaging gives a systematically higher daily CF than converting "
        "the daily-mean wind, because the turbine curve is nonlinear. Solar agrees almost exactly."
    )
    st.dataframe(pd.DataFrame(cqc["daily_vs_existing_daily"]["metrics"]).T.round(4), use_container_width=True, hide_index=False)
    st.caption("Resolution-sensitivity check; all values are dimensionless CF or CF differences.")

    section("Low-output classification (hybrid CF < 0.10)")
    rows = []
    for wgt, x in qc["low_output_classification"].items():
        rows.append({"weighting": wgt, **x["confusion"], "precision": x["precision"], "recall": x["recall"],
                     "agreement": x["agreement"],
                     "observed longest run (h)": x["observed_run_summary"]["longest_hours"],
                     "model longest run (h)": x["model_run_summary"]["longest_hours"]})
    st.dataframe(pd.DataFrame(rows).set_index("weighting").round(3), use_container_width=True, hide_index=False)
    st.caption("tp/fp/fn/tn are hour counts; precision/recall/agreement are fractions (0–1); longest runs in hours.")
    st.warning("Nighttime solar zeros create repeated low-output hours, so hourly runs are not equivalent to the "
               "daily ≥2-day event catalogue. " + NOT_SHORTAGE)
    footer()


# --------------------------------------------------------------------------- shell
PAGES = {
    "1. Overview — story": page_overview,
    "2. Does the proxy reproduce observed generation?": page_daily,
    "3. Does combining wind and solar reduce low-output periods?": page_complementarity,
    "4. Does the installed fleet change the answer?": page_capacity,
    "5. Do daily averages hide hourly behaviour?": page_hourly,
}

st.sidebar.markdown("### Germany 2018")
st.sidebar.markdown("**From weather to renewable-energy signals**")
st.sidebar.caption("COSMO-REA6 → weather-to-power proxies → SMARD validation")
choice = st.sidebar.radio("Chapter", list(PAGES), label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(
    "This app is a guided scientific story. It reads only pre-computed `app_assets/2018/` — "
    "no downloads, no GRIB decoding, no research-pipeline execution.\n\n"
    "Notebooks `00`–`06` contain the full reproducible analysis."
)
PAGES[choice]()
