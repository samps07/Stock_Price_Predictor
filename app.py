# app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from Stock_Predictor import train_and_predict

st.set_page_config(page_title="Stock Price Prediction App", layout="wide")
st.title("📈 Stock Price Prediction App", help="A Machine Learning based web app to predict approximate prices and future trends of stocks.")

# Sidebar
st.sidebar.header("Input Options")
prebuilt = st.sidebar.selectbox(
    "Select Period",
    ["1 Month", "2 Months", "3 Months", "6 Months", "Custom"],
    help="Period of days to consider for training and prediction.\n\nEnsure that Period>window size + 2 , excluding non trading days."
)

today = datetime.today()
if prebuilt == "1 Month":
    start_date = today - timedelta(days=30)
    end_date = today
elif prebuilt == "2 Months":
    start_date = today - timedelta(days=60)
    end_date = today
elif prebuilt == "3 Months":
    start_date = today - timedelta(days=90)
    end_date = today
elif prebuilt == "6 Months":
    start_date = today - timedelta(days=180)
    end_date = today
else:  # Custom
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", today)

ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL",
                              help="You can get the Ticker code of a Stock from Google")

window_size = st.sidebar.slider(
    "Select Window Size (days)", min_value=5, max_value=50, value=5, step=1,
    help="Number of past days used to predict tomorrow's movement.\nRecommended range is 5-23 depending on training period"
)

# helper
def safe_float(x, default=0.0):
    try:
        return float(x) if x is not None else default
    except Exception:
        return default

if st.sidebar.button("Run Prediction"):
    with st.spinner("Training models... Please wait ⏳... \nYour Prediction is Almost here..."):
        results = None
        error = None
        try:
            results, error = train_and_predict(
                ticker,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                window_size=window_size
            )
        except Exception as e:
            st.error("An exception occurred while running the predictor. See details below.")
            st.exception(e)
            results = None
            error = str(e)

    if error:
        st.error(error)
    elif results is None:
        st.error("No results returned from train_and_predict(). Check logs/exception above.")
    else:
        st.subheader(f"📊 Results for {ticker}")

        # Top boxes: Best model prediction, Average prediction, Classification result
        best_model = results.get("best_model")
        best_pred = None
        avg_pred = results.get("avg_prediction")
        if best_model:
            best_pred = results.get("regression_results", {}).get(best_model, {}).get("Forecast", {}).get("next_day_pred")

        class_info = results.get("classification_results", {})
        class_next = class_info.get("next_day_pred")
        class_conf = class_info.get("next_day_confidence", 0.0)

        col1, col2, col3 = st.columns([1.2, 1.2, 1.0])
        with col1:
            if best_pred is not None:
                st.metric(label=f"Best model ({best_model}) \n\nPredicted price tomorrow:", value=f"{safe_float(best_pred):.2f}")
                st.caption(f"Chosen by lowest MAE (backtest). MAE: {safe_float(results.get('best_model_mae')):.4f}")
            else:
                st.info("Best model prediction not available.")

        with col2:
            if avg_pred is not None:
                st.metric(label="Average prediction \n\n(all models)", value=f"{safe_float(avg_pred):.2f}")
                st.caption("Mean of available model forecasts.")
            else:
                st.info("Average prediction not available.")

        with col3:
            if class_next is not None:
                verb = "Up (1)" if int(class_next) == 1 else "Down (0)"
                st.metric(label="Classification forecast \n\n(Up/Down)", value=verb, delta=f"{safe_float(class_conf)*100:.2f}%")
                st.caption("Forecast Confidence")
            else:
                st.info("Classification forecast not available.")
        st.write(f"**Last Closing Price for {ticker}:** {results.get('last_close', 0):.2f}")

        # ---------------- Regression Results ----------------
        st.markdown("## 🔹 Regression Models (Next-day Price Prediction)")
        for model_name, info in results.get("regression_results", {}).items():
            back = info.get("Backtest", {})
            forecast = info.get("Forecast", {})

            next_day = forecast.get("next_day_pred", None)
            st.markdown(f"""
            ### {model_name}
            - **Next-day Prediction:** {safe_float(next_day):.2f}  
            - **Test MAE (Mean Absolute Error):** {safe_float(back.get('mae')):.4f}  
            - **Relative Error (% of Avg Price):** {safe_float(back.get('Relative_Error(%)')):.2f}%  
            - **Test RMSE (Root Mean Squared Error):** {safe_float(back.get('rmse')):.4f}  
            - **Test R² Score:** {safe_float(back.get('r2')):.4f}  
            """)

            dates = back.get("dates", [])
            y_actual = back.get("y_test", [])
            y_pred = back.get("preds", [])

            if dates and len(dates) == len(y_actual) == len(y_pred):
                dates = pd.to_datetime(dates)
                df_chart = pd.DataFrame({"Actual": y_actual, "Predicted": y_pred}, index=dates)
                try:
                    future_date = dates[-1] + pd.Timedelta(days=1)
                    df_future = pd.DataFrame({"Actual": [None], "Predicted": [next_day]}, index=[future_date])
                    df_chart = pd.concat([df_chart, df_future])
                except Exception:
                    pass
                st.line_chart(df_chart)
            else:
                st.write("Backtest data length mismatch or not available. Showing raw arrays:")
                st.write({"dates": dates, "y_test_len": len(y_actual), "preds_len": len(y_pred)})

        # ---------------- Classification Results ----------------
        st.markdown("## 🔹 Classification Model (Up/Down Movement)")
        acc = class_info.get("accuracy", 0.0)
        st.markdown(f"- **Accuracy:** {acc:.4f}  ")

        # Build classification dataframe with Actual and Predicted to show both lines
        class_dates = class_info.get("dates", [])
        class_preds = class_info.get("all_preds", [])
        class_confidences = class_info.get("all_confidences", [])
        class_actuals = class_info.get("y_test", [])

        if class_dates and len(class_dates) == len(class_preds) == len(class_actuals):
            dates = pd.to_datetime(class_dates)
            df_class = pd.DataFrame({
                "Actual": class_actuals,
                "Predicted": class_preds,
                "Confidence": class_confidences
            }, index=dates)
            st.line_chart(df_class)
        else:
            # fallback: show arrays
            st.write("Classification backtest arrays:")
            st.write({
                "dates": class_dates,
                "actuals": class_actuals,
                "preds": class_preds,
                "confidences": class_confidences
            })

        st.success("✅ Prediction Complete!")
