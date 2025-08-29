# Stock_Predictor.py
import yfinance as yf
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

def _walk_forward_generic(model, X_all, y_all, dates_all, init_train=1):
    n = len(X_all)
    preds_walk = []
    y_test = []
    dates_test = []

    if n <= init_train:
        return [], [], []

    for i in range(init_train, n):
        X_tr = X_all[:i]
        y_tr = y_all[:i]
        X_pred = X_all[i].reshape(1, -1)

        model.fit(X_tr, y_tr)
        preds_walk.append(float(model.predict(X_pred)[0]))
        y_test.append(float(y_all[i]))
        dates_test.append(dates_all[i])

    return dates_test, y_test, preds_walk


def train_and_predict(ticker, start_date, end_date, window_size=20):
    """
    Returns (results_dict, error_str). results_dict includes 'forecast_date' (YYYY-MM-DD).
    NOTE: end_date is treated inclusively by adding +1 calendar day when calling yfinance.
    """
    # ensure inclusive range for yfinance (yfinance end param is effectively exclusive here)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    raw = yf.download(ticker, start=pd.to_datetime(start_date).strftime("%Y-%m-%d"),
                      end=end_dt.strftime("%Y-%m-%d"))
    if raw.empty:
        return None, "No data found for the given range"

    # get closes and canonical dates (raw.index is authoritative)
    # raw may have multiindex columns when using certain tickers; keep only 'Close'
    if "Close" in raw.columns:
        closes = raw["Close"].values
    else:
        # defensive: try to find close column
        closes = raw.iloc[:, raw.columns.get_level_values(1) == "Close"].iloc[:, 0].values

    dates = list(raw.index)  # authoritative

    if len(closes) < 2:
        return None, "Not enough data (need >=2 trading days)."

    # Build samples
    X_list = []
    y_list = []
    dates_for_targets = []

    if window_size > 0:
        if len(closes) <= window_size:
            return None, f"Not enough data for window_size={window_size}. Need > {window_size} trading days."
        for t in range(window_size, len(closes)):
            X_list.append(closes[t-window_size:t].copy())
            y_list.append(closes[t])
            dates_for_targets.append(dates[t])
    else:
        for t in range(1, len(closes)):
            X_list.append(np.array([closes[t-1]]))
            y_list.append(closes[t])
            dates_for_targets.append(dates[t])

    X = np.array(X_list)
    y = np.array(y_list)
    X_flat = X.reshape(X.shape[0], -1)

    init_train = 1

    regressors = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    regression_results = {}
    next_day_preds = {}
    forecast_date = None

    for name, model in regressors.items():
        dates_test, y_test, preds_walk = _walk_forward_generic(
            model, X_flat, y, dates_for_targets, init_train=init_train
        )

        if len(y_test) > 0:
            mae = mean_absolute_error(y_test, preds_walk)
            rmse = np.sqrt(mean_squared_error(y_test, preds_walk))
            r2 = r2_score(y_test, preds_walk)
            rel_err = (mae / np.mean(y_test)) * 100 if np.mean(y_test) != 0 else 0.0
        else:
            mae = rmse = r2 = rel_err = 0.0

        # retrain on full data and forecast using last window from raw closes
        try:
            model.fit(X_flat, y)
            if window_size > 0:
                last_window = closes[-window_size:]
            else:
                last_window = np.array([closes[-1]])
            last_window = np.array(last_window).reshape(1, -1)
            next_pred = float(model.predict(last_window)[0])
        except Exception:
            next_pred = None

        # compute forecast_date from raw's last available index (next business day)
        last_raw_idx = pd.to_datetime(dates[-1])
        forecast_date_candidate = (last_raw_idx + BDay(1)).normalize()
        # ensure forecast_date consistent across models
        if forecast_date is None:
            forecast_date = forecast_date_candidate

        next_day_preds[name] = next_pred

        regression_results[name] = {
            "Backtest": {
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "Relative_Error(%)": float(rel_err),
                "y_test": [float(x) for x in y_test],
                "preds": [float(x) for x in preds_walk],
                "dates": [str(d) for d in dates_test]
            },
            "Forecast": {
                "next_day_pred": float(next_pred) if next_pred is not None else None,
                "forecast_date": str(forecast_date.date()) if forecast_date is not None else None
            }
        }

    # choose best model by lowest MAE
    best_model = None
    best_mae = None
    for name, info in regression_results.items():
        mae = info["Backtest"].get("mae", None)
        if best_model is None:
            best_model = name
            best_mae = mae
        else:
            if mae is not None and (best_mae is None or mae < best_mae):
                best_model = name
                best_mae = mae

    valid_preds = [v for v in next_day_preds.values() if v is not None]
    avg_pred = float(np.mean(valid_preds)) if len(valid_preds) > 0 else None

    # ---------------- Classification ----------------
    labels = []
    for xi, yi in zip(X, y):
        prev_close = xi[-1]
        labels.append(1 if yi > prev_close else 0)
    labels = np.array(labels)

    log_model = LogisticRegression(max_iter=1000)
    n_samples = X_flat.shape[0]

    class_preds = []
    class_probs = []
    class_dates = []
    class_y = []

    if n_samples > init_train:
        for i in range(init_train, n_samples):
            X_tr = X_flat[:i]
            y_tr = labels[:i]
            X_pred = X_flat[i].reshape(1, -1)

            unique_classes = np.unique(y_tr)
            if len(unique_classes) == 1:
                pred_class = int(unique_classes[0])
                pred_conf = 1.0
            else:
                log_model.fit(X_tr, y_tr)
                pred_class = int(log_model.predict(X_pred)[0])
                pred_conf = float(log_model.predict_proba(X_pred)[0].max())

            class_preds.append(pred_class)
            class_probs.append(pred_conf)
            class_y.append(int(labels[i]))
            class_dates.append(dates_for_targets[i])

        if len(class_y) > 0:
            class_accuracy = float(accuracy_score(class_y, class_preds))
            class_last_pred = int(class_preds[-1])
            class_last_actual = int(class_y[-1])
            class_last_conf = float(class_probs[-1])
        else:
            class_accuracy = 0.0
            class_last_pred = None
            class_last_actual = None
            class_last_conf = None

        # forecast next-day class using last window and the same forecast_date
        full_unique = np.unique(labels)
        if len(full_unique) == 1:
            next_day_class = int(full_unique[0])
            next_day_conf = 1.0
        else:
            try:
                log_model.fit(X_flat, labels)
                if window_size > 0:
                    last_window = closes[-window_size:]
                else:
                    last_window = np.array([closes[-1]])
                next_day_class = int(log_model.predict(np.array(last_window).reshape(1, -1))[0])
                next_day_conf = float(log_model.predict_proba(np.array(last_window).reshape(1, -1))[0].max())
            except Exception:
                next_day_class = None
                next_day_conf = None
    else:
        class_accuracy = 0.0
        class_last_pred = None
        class_last_actual = None
        class_last_conf = None
        class_preds = []
        class_probs = []
        class_dates = []
        class_y = []
        next_day_class = None
        next_day_conf = None

    classification_results = {
        "accuracy": class_accuracy,
        "last_pred": class_last_pred,
        "last_actual": class_last_actual,
        "confidence": class_last_conf,
        "all_preds": class_preds,
        "all_confidences": class_probs,
        "dates": [str(d) for d in class_dates],
        "y_test": [int(x) for x in class_y],
        "next_day_pred": next_day_class,
        "next_day_confidence": next_day_conf,
        "forecast_date": str(forecast_date.date()) if forecast_date is not None else None
    }

    return {
        "last_close": float(closes[-1]),
        "regression_results": regression_results,
        "classification_results": classification_results,
        "best_model": best_model,
        "best_model_mae": float(best_mae) if best_mae is not None else None,
        "avg_prediction": float(avg_pred) if avg_pred is not None else None,
        "forecast_date": str(forecast_date.date()) if forecast_date is not None else None,
    }, None
