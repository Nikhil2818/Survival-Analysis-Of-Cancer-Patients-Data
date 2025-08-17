import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

# For survival curve plotting
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for Flask
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# --- Model Loading ---
model_path = os.path.join(os.path.dirname(__file__), "rsf_model (4).pkl")
columns_path = os.path.join(os.path.dirname(__file__), "model_columns (4).pkl")

try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

try:
    model_columns = joblib.load(columns_path)
    print("Model columns loaded successfully!")
except Exception as e:
    print(f"Error loading model columns: {e}")
    raise e

# --- Feature Preparation ---
final_model_features = model_columns.get("final_features", [])
categorical_options = {
    k: v for k, v in model_columns.items() if k != "final_features"
}

form_features = list(categorical_options.keys())

# Infer numerical features
all_ohe_columns = []
for base_feature, options in categorical_options.items():
    for option in options:
        all_ohe_columns.append(f"{base_feature}_{option}")

numerical_features = [
    col for col in final_model_features
    if col not in all_ohe_columns and col not in categorical_options
]

# Final form order: numerical first
form_features = numerical_features + form_features

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_msg = None
    plot_url = None

    if request.method == "POST":
        try:
            form_data = {}

            # --- Collect Form Data ---
            for col in form_features:
                val = request.form.get(col)
                if val is None or val.strip() == "":
                    val = 0 if col in numerical_features else categorical_options[col][0]

                if col in numerical_features:
                    try:
                        val = float(val)
                    except ValueError:
                        error_msg = f"Invalid input for {col}. Please enter a number."
                        break
                form_data[col] = val

            if not error_msg:
                # --- Create DataFrame for Model ---
                df = pd.DataFrame([form_data])
                df = pd.get_dummies(df)
                df = df.reindex(columns=final_model_features, fill_value=0)

                # --- Predict ---
                surv_funcs = model.predict_survival_function(df)
                fn = surv_funcs[0]

                # Median survival time
                median_time = np.interp(0.5, fn.y[::-1], fn.x[::-1])
                prediction = round(float(median_time), 2)

                # --- Plot Survival Curve ---
                fig, ax = plt.subplots()
                ax.step(fn.x, fn.y, where="post", label="Survival Function")
                ax.set_xlabel("Time (months)")
                ax.set_ylabel("Survival Probability")
                ax.set_title("Predicted Survival Curve")
                ax.grid(True)
                ax.legend()

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plot_url = base64.b64encode(buf.getvalue()).decode()
                plt.close(fig)

        except Exception as e:
            error_msg = f"Prediction error: Please check your inputs. ({str(e)})"

    return render_template(
        "index.html",
        form_features=form_features,
        categorical_options=categorical_options,
        prediction=prediction,
        plot_url=plot_url,
        error_msg=error_msg,
        form_data=request.form
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
