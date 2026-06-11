"""Yield forecasting: 5 km grid features → scikit-learn regression →
district/state aggregation."""
import ee
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error


def make_grid(aoi, cell_size_m=5000):
    """5 km x 5 km grid over an AOI as ee.FeatureCollection."""
    proj = ee.Projection("EPSG:4326").atScale(cell_size_m)
    grid_img = ee.Image.pixelCoordinates(proj).floor()
    cell_id = grid_img.select("x").multiply(100000) \
        .add(grid_img.select("y")).rename("cell").int64()
    return cell_id.clip(aoi)


def extract_grid_features(images_dict, grid_img, aoi, scale=1000):
    """Reduce each feature image per grid cell → pandas DataFrame.

    images_dict: {"feature_name": ee.Image (single band)}
    """
    stack = ee.Image.cat(
        [img.rename(name) for name, img in images_dict.items()])
    stats = stack.addBands(grid_img).reduceRegion(
        reducer=ee.Reducer.mean().group(
            groupField=len(images_dict), groupName="cell"),
        geometry=aoi, scale=scale, maxPixels=1e12, bestEffort=True)
    groups = stats.getInfo()["groups"]
    rows = []
    names = list(images_dict.keys())
    for g in groups:
        row = {"cell": g["cell"]}
        means = g["mean"]
        for i, n in enumerate(names):
            row[n] = means[i] if isinstance(means, list) else means
        rows.append(row)
    return pd.DataFrame(rows)


def build_models(cfg):
    """Instantiate RF + GBM regressors from config dict."""
    rf_p = cfg["models"]["random_forest"]
    gb_p = cfg["models"]["gradient_boosting"]
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=rf_p["n_estimators"],
            max_depth=rf_p["max_depth"], random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=gb_p["n_estimators"],
            learning_rate=gb_p["learning_rate"],
            max_depth=gb_p["max_depth"], random_state=42),
    }


def cross_validate(models, X, y, folds=5):
    """K-fold CV R2 per model → DataFrame."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
        rows.append({"model": name, "r2_mean": scores.mean(),
                     "r2_std": scores.std()})
    return pd.DataFrame(rows)


def fit_and_predict(model, X_train, y_train, X_pred):
    model.fit(X_train, y_train)
    return model.predict(X_pred)


def evaluate(y_true, y_pred):
    """R2, RMSE, bias evaluation dict."""
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "bias": float(np.mean(np.asarray(y_pred) - np.asarray(y_true))),
    }


def aggregate_to_admin(grid_df, mapping_df, value_col="yield_pred",
                       admin_col="district"):
    """Area-weighted mean of grid predictions per district/state."""
    merged = grid_df.merge(mapping_df, on="cell", how="inner")
    w = merged.get("wheat_fraction", pd.Series(1.0, index=merged.index))
    merged["_w"] = w
    out = (merged.groupby(admin_col)
           .apply(lambda d: np.average(d[value_col], weights=d["_w"]),
                  include_groups=False)
           .rename(value_col).reset_index())
    return out
