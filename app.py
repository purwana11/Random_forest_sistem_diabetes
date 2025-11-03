from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# === Load model Random Forest dan feature names ===
model = joblib.load("model_diabetes_rf.pkl")
feature_names = joblib.load("feature_names.pkl")
df = pd.read_csv("diabetes.csv")

# Nilai akurasi dan AUC dari hasil training sebelumnya
MODEL_ACCURACY = 0.82
MODEL_AUC = 0.88

@app.route("/")
def home():
    return render_template("index.html", accuracy=MODEL_ACCURACY, auc=MODEL_AUC)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(request.form.get(f)) for f in feature_names]
        data = np.array([values])
        pred = model.predict(data)[0]

        result_text = "Positif Diabetes" if pred == 1 else "Tidak Diabetes"
        color = "#dc3545" if pred == 1 else "#198754"  # merah / hijau

        return render_template("result.html", 
                               result=result_text, 
                               color=color, 
                               accuracy=MODEL_ACCURACY,
                               auc=MODEL_AUC)
    except Exception as e:
        return f"Terjadi kesalahan: {e}"

@app.route("/dataset")
def dataset():
    data_html = df.to_html(classes="table table-striped table-hover", index=False)
    return render_template("dataset.html", data=data_html)

if __name__ == "__main__":
    app.run(debug=True)
