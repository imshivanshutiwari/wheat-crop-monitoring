"""Plotting: choropleths, VHI time-series, yield bars, phenology curves,
hotspot overlays, correlation scatter."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sown_area_choropleth(states_gdf, area_col="area_lakh_ha",
                         title="Wheat Sown Area (lakh ha)"):
    """State-wise choropleth from a GeoDataFrame."""
    ax = states_gdf.plot(column=area_col, cmap="YlGn", legend=True,
                         edgecolor="black", linewidth=0.4,
                         figsize=(9, 9))
    ax.set_title(title)
    ax.set_axis_off()
    return ax


def vhi_timeseries(df, date_col="date", vhi_col="VHI", state_col="state",
                   stress_threshold=40, ax=None):
    """Fortnightly VHI lines per state with stress threshold band."""
    ax = ax or plt.subplots(figsize=(11, 5))[1]
    for state, d in df.groupby(state_col):
        ax.plot(d[date_col], d[vhi_col], marker="o", label=state)
    ax.axhline(stress_threshold, color="red", ls="--",
               label=f"Stress (<{stress_threshold})")
    ax.set_ylabel("VHI")
    ax.set_title("Vegetation Health Index — fortnightly")
    ax.legend(ncol=3, fontsize=8)
    return ax


def yield_forecast_bars(df, admin_col="district", value_col="yield_pred",
                        title="District-wise Yield Forecast (t/ha)"):
    df = df.sort_values(value_col, ascending=False)
    ax = df.plot.bar(x=admin_col, y=value_col, figsize=(12, 5),
                     color="seagreen", legend=False)
    ax.set_ylabel("Yield (t/ha)")
    ax.set_title(title)
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    return ax


def phenology_backscatter_curve(ts_df, phenology, band="VH",
                                title="SAR Backscatter vs Wheat Phenology"):
    """VH backscatter time-series with shaded phenology stages."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts_df["date"], ts_df[band], "o-", color="navy", label=band)
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(phenology)))
    year = ts_df["date"].dt.year.min()
    for (stage, (s, e)), c in zip(phenology.items(), colors):
        sm = int(s[:2])
        yr = year if sm >= 11 else year + 1
        ax.axvspan(pd.Timestamp(f"{yr}-{s}"), pd.Timestamp(f"{yr}-{e}"),
                   color=c, alpha=0.5, label=stage)
    ax.set_ylabel(f"{band} backscatter (dB)")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8)
    return ax


def correlation_scatter(df, x_col, y_col, metrics=None,
                        title="Satellite vs Ground-Truth"):
    """1:1 scatter with optional R2/RMSE/bias annotation box."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df[x_col], df[y_col], color="darkorange",
               edgecolor="black", alpha=0.8)
    lims = [min(df[x_col].min(), df[y_col].min()) * 0.95,
            max(df[x_col].max(), df[y_col].max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    ax.set_xlim(lims), ax.set_ylim(lims)
    ax.set_xlabel(x_col), ax.set_ylabel(y_col)
    ax.set_title(title)
    if metrics:
        txt = "\n".join(f"{k.upper()}: {v:.3f}" for k, v in metrics.items())
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", fc="lightyellow"))
    ax.legend()
    return ax
