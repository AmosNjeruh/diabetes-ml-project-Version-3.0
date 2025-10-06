import os
import uuid
from datetime import datetime
import tempfile

import pandas as pd
import numpy as np
import joblib
import gradio as gr

import shap
import lime.lime_tabular

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging
import json
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time

# Configure logging to file app.log
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# === Globals ===
DF = None
X = None
y = None
model = None
scaler = None
shap_explainer = None
lime_explainer = None
model_comparison_df = None
model_combo = "Combo1"

# Global toggle for hyperparameter tuning
ENABLE_TUNING = False

# === Model Combinations ===
combos = {
    "Combo1": [RandomForestClassifier(), LogisticRegression(max_iter=1000), GradientBoostingClassifier(), ExtraTreesClassifier()],
    "Combo2": [AdaBoostClassifier(), GradientBoostingClassifier(), LogisticRegression(max_iter=1000), ExtraTreesClassifier()],
    "Combo3": [RandomForestClassifier(), AdaBoostClassifier(), LogisticRegression(max_iter=1000), GradientBoostingClassifier()]
}

# === GEMMA setup (optional, may fail on CPU or without HF token) ===
gemma_pipeline = None
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    model_id = "google/gemma-1.1-2b-it"
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
        gemma_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=None, use_auth_token=hf_token)
        gemma_pipeline = pipeline("text-generation", model=gemma_model, tokenizer=tokenizer)
    else:
        # No token provided: do not try to load the large model
        gemma_pipeline = None
except Exception as e:
    # If any import or load fails, keep pipeline None and continue. Spaces may not have resources needed.
    gemma_pipeline = None

# Utility: default PIMA URL
DEFAULT_PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# === Load and Train Model ===
def load_and_train(url):
    global DF, X, y, model, scaler, shap_explainer, lime_explainer, model_comparison_df, model_combo
    try:
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(url, names=columns)

        # Basic validation
        if not all(col in df.columns for col in columns):
            return "Error: Missing required columns in the dataset."

        DF = df.copy()
        X = DF.drop('Outcome', axis=1)
        y = DF['Outcome']

        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Pick models from combo
        selected_models = combos.get(model_combo, combos["Combo1"])
        results = []

        trained_models = []
        for m in selected_models:
            # Apply tuning if ENABLE_TUNING = True
            if isinstance(m, LogisticRegression):
                tuned_model = tune_model(m, {"C": [0.01, 0.1, 1, 10]}, X_scaled, y)
            elif isinstance(m, RandomForestClassifier):
                tuned_model = tune_model(m, {"n_estimators": [50, 100, 200]}, X_scaled, y)
            elif isinstance(m, GradientBoostingClassifier):
                tuned_model = tune_model(m, {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}, X_scaled, y)
            elif isinstance(m, SVC):
                tuned_model = tune_model(m, {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}, X_scaled, y)
            else:
                tuned_model = tune_model(m, {}, X_scaled, y)  # fallback

            trained_models.append(tuned_model)

            # Evaluate model
            preds = tuned_model.predict(X_scaled)
            results.append({
                "Model": type(tuned_model).__name__,
                "Accuracy": round(accuracy_score(y, preds), 4),
                "Precision": round(precision_score(y, preds), 4),
                "Recall": round(precision_score(y, preds), 4),
                "F1 Score": round(f1_score(y, preds), 4)
            })

        # Pick first model for explainability
        model = trained_models[0]

        # SHAP: only works for tree/linear models with predict_proba
        try:
            shap_explainer = shap.TreeExplainer(model)
        except Exception:
            shap_explainer = None

        # LIME: always works if scaling was successful
        try:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_scaled,
                feature_names=X.columns.tolist(),
                class_names=["No", "Yes"],
                mode='classification'
            )
        except Exception:
            lime_explainer = None

        model_comparison_df = pd.DataFrame(results)
        globals()['scaler'] = scaler  # persist scaler for predictions
        return "Dataset loaded and model trained successfully!"

    except Exception as e:
        return f"Error during loading or training: {str(e)}"



   

def get_models():
    return {
        "logreg": LogisticRegression(max_iter=500),
        "rf": RandomForestClassifier(),
        "gb": GradientBoostingClassifier(),
        "svm": SVC(probability=True)
    }


def tune_model(model, param_grid, X_train, y_train):
    global ENABLE_TUNING
    if ENABLE_TUNING:
        grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)
        return grid.best_estimator_
    else:
        model.fit(X_train, y_train)
        return model


def train_models(X_train, y_train, combo="Combo1"):
    models = get_models()
    trained = {}

    if combo == "Combo1":
        trained["logreg"] = tune_model(
            models["logreg"],
            {"C": [0.01, 0.1, 1, 10]},
            X_train, y_train
        )
        trained["rf"] = tune_model(
            models["rf"],
            {"n_estimators": [50, 100, 200]},
            X_train, y_train
        )

    elif combo == "Combo2":
        trained["rf"] = tune_model(
            models["rf"],
            {"n_estimators": [50, 100, 200]},
            X_train, y_train
        )
        trained["gb"] = tune_model(
            models["gb"],
            {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
            X_train, y_train
        )

    elif combo == "Combo3":
        trained["logreg"] = tune_model(
            models["logreg"],
            {"C": [0.01, 0.1, 1, 10]},
            X_train, y_train
        )
        trained["svm"] = tune_model(
            models["svm"],
            {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
            X_train, y_train
        )

    return trained
def toggle_tuning(enable):
    global ENABLE_TUNING
    ENABLE_TUNING = enable
    return f"Tuning is now {'ENABLED' if enable else 'DISABLED'} for all combinations."
    
def load_and_train_from_df(dataframe):
    global df, X, y, model, scaler, shap_explainer, lime_explainer, model_comparison_df
    try:
        df = dataframe.copy()

        # Split features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']

        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Select model combo
        selected_models = combos.get(model_combo, combos["Combo1"])
        results = []

        trained_models = []
        for m in selected_models:
            # Apply tuning if ENABLE_TUNING = True
            if isinstance(m, LogisticRegression):
                tuned_model = tune_model(m, {"C": [0.01, 0.1, 1, 10]}, X_scaled, y)
            elif isinstance(m, RandomForestClassifier):
                tuned_model = tune_model(m, {"n_estimators": [50, 100, 200]}, X_scaled, y)
            elif isinstance(m, GradientBoostingClassifier):
                tuned_model = tune_model(m, {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}, X_scaled, y)
            elif isinstance(m, SVC):
                tuned_model = tune_model(m, {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}, X_scaled, y)
            else:
                tuned_model = tune_model(m, {}, X_scaled, y)  # fallback, no tuning

            trained_models.append(tuned_model)

            preds = tuned_model.predict(X_scaled)
            results.append({
                "Model": type(tuned_model).__name__,
                "Accuracy": round(accuracy_score(y, preds), 4),
                "Precision": round(precision_score(y, preds), 4),
                "Recall": round(recall_score(y, preds), 4),
                "F1 Score": round(f1_score(y, preds), 4)
            })

        # Pick first model as default for explainability
        model = trained_models[0]

        # SHAP + LIME
        shap_explainer = shap.TreeExplainer(model) if hasattr(model, "predict_proba") else None
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_scaled,
            feature_names=X.columns.tolist(),
            class_names=["No", "Yes"],
            mode='classification'
        )

        model_comparison_df = pd.DataFrame(results)
        return "Dataset loaded and model trained successfully!"

    except Exception as e:
        return f"Error during loading or training: {str(e)}"



  

# === Upload Dataset ===
def upload_dataset(file=None, url=None):
    global df
    if file is not None:
        # Read file with headers assumed present
        df = pd.read_csv(file.name)  # <-- reads header row
        # Now check columns and proceed to train
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        # Validate columns exist
        if not all(col in df.columns for col in columns):
            return "Error: Uploaded dataset missing required columns.", None

        # Call train on the dataframe directly
        # Since load_and_train expects URL, refactor it to accept df
        return load_and_train_from_df(df)
    elif url is not None:
        return load_and_train(url)
    else:
        return "No file or URL provided.", None


# === admin login function===
def admin_login(username, password):
    # Simple demo credentials (change for production)
    if username == "Amos" and password == "Kenya2025":
        return gr.update(visible=True), gr.update(visible=False), True
    else:
        return gr.update(visible=False), gr.update(visible=True), False

# === Imputation Method ===
def choose_imputation_method(method):
    global X
    if X is None:
        return "No dataset loaded."
    if method == "Median":
        imputer = SimpleImputer(strategy='median')
    elif method == "KNN":
        imputer = KNNImputer()
    else:
        return "Invalid method selected."

    X_imputed = imputer.fit_transform(X)
    # update global X to imputed values
    globals()['X'] = pd.DataFrame(X_imputed, columns=X.columns)
    return "Imputation method applied successfully!"

# Outlier Detection using Z-score or IQR
def handle_outliers(method):
    global DF
    if DF is None or DF.empty:
        return "Dataset is empty"

    df_cleaned = DF.copy()
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    if "Outcome" in numeric_cols:
        numeric_cols.remove("Outcome")

    if method == "Z-score":
        z_scores = np.abs((df_cleaned[numeric_cols] - df_cleaned[numeric_cols].mean()) / df_cleaned[numeric_cols].std())
        df_cleaned = df_cleaned[(z_scores < 3).all(axis=1)]
    elif method == "IQR":
        Q1 = df_cleaned[numeric_cols].quantile(0.25)
        Q3 = df_cleaned[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((df_cleaned[numeric_cols] < (Q1 - 1.5 * IQR)) | (df_cleaned[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        df_cleaned = df_cleaned[mask]
    else:
        return "Invalid method selected"

    DF = df_cleaned.copy()
    return f"Outliers handled using {method}. Remaining rows: {len(DF)}"

# VIF Calculation
def calculate_vif():
    global DF
    if DF is None or DF.empty:
        return pd.DataFrame(columns=["Feature", "VIF"])

    X_vif = DF.drop(columns=["Outcome"])
    X_scaled = StandardScaler().fit_transform(X_vif)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    return vif_data

# Feature Importance
def feature_importance_and_correlation():
    global DF
    if DF is None or DF.empty:
        return None, None

    data = DF.copy()
    X_local = data.drop(columns=["Outcome"])
    y_local = data["Outcome"]

    model_local = RandomForestClassifier()
    model_local.fit(X_local, y_local)
    importance = pd.DataFrame({'Feature': X_local.columns, 'Importance': model_local.feature_importances_})

    # Feature importance plot
    fig1, ax1 = plt.subplots()
    importance.sort_values(by="Importance", ascending=True).plot.barh(x="Feature", y="Importance", ax=ax1)
    ax1.set_title("Feature Importance")

    # Correlation plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    ax2.set_title("Correlation Matrix")

    return fig1, fig2

# === Model Evaluation ===
def evaluate_model(combo, test_size=0.2, random_state=42):
    global X, y
    if X is None or y is None:
        return pd.DataFrame(), None, None, None, None

    selected_models = combos.get(combo, combos["Combo1"])
    results = []
    auc_results = []

    X_vals = X.values if hasattr(X, 'values') else X
    X_train, X_test, y_train, y_test = train_test_split(
        X_vals, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- ROC Plot (Test set) ---
    fig_roc, ax_roc = plt.subplots()

    # --- Confusion Matrices (Test set) ---
    n_models = len(selected_models)
    n_cols = 2
    n_rows = (n_models + 1) // n_cols
    fig_cm, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, m in enumerate(selected_models):
        m.fit(X_train, y_train)

        # Train & test preds
        train_preds = m.predict(X_train)
        test_preds = m.predict(X_test)

        # Probabilities for AUC
        proba_train = m.predict_proba(X_train)[:, 1] if hasattr(m, "predict_proba") else None
        proba_test = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else None

        # Metrics
        results.append({
            "Model": type(m).__name__,
            "Train Accuracy": round(accuracy_score(y_train, train_preds), 4),
            "Train Precision": round(precision_score(y_train, train_preds), 4),
            "Train Recall": round(recall_score(y_train, train_preds), 4),
            "Train F1": round(f1_score(y_train, train_preds), 4),
            "Test Accuracy": round(accuracy_score(y_test, test_preds), 4),
            "Test Precision": round(precision_score(y_test, test_preds), 4),
            "Test Recall": round(recall_score(y_test, test_preds), 4),
            "Test F1": round(f1_score(y_test, test_preds), 4),
        })

        # AUC values
        if proba_train is not None:
            auc_train = auc(*roc_curve(y_train, proba_train)[:2])
            auc_test = auc(*roc_curve(y_test, proba_test)[:2])
            auc_results.append((type(m).__name__, auc_train, auc_test))

            # Plot ROC for test
            fpr, tpr, _ = roc_curve(y_test, proba_test)
            ax_roc.plot(fpr, tpr, label=f"{type(m).__name__} (AUC={auc_test:.2f})")

        # Confusion matrix (test)
        cm = confusion_matrix(y_test, test_preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=axes[i])
        axes[i].set_title(f"{type(m).__name__}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig_cm.delaxes(axes[j])

    # Finalize ROC
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_title("Test ROC Curves")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")

    # --- AUC Bar Plot ---
    fig_auc, ax_auc = plt.subplots()
    models = [r[0] for r in auc_results]
    train_aucs = [r[1] for r in auc_results]
    test_aucs = [r[2] for r in auc_results]
    x = range(len(models))
    ax_auc.bar([i - 0.2 for i in x], train_aucs, width=0.4, label="Train AUC")
    ax_auc.bar([i + 0.2 for i in x], test_aucs, width=0.4, label="Test AUC")
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels(models, rotation=45)
    ax_auc.set_ylim(0.0, 1.0)
    ax_auc.set_title("Train vs Test AUC Comparison")
    ax_auc.set_ylabel("AUC Score")
    ax_auc.legend()

    return pd.DataFrame(results), fig_roc, fig_cm, fig_auc


# Prediction
def predict_diabetes(p, g, bp, st, i, bmi, dpf, age):
    global model, scaler

    if model is None or scaler is None:
        return "Model not loaded or trained yet.", None, None

    start_time = time.time()

    # Prepare input
    input_data = np.array([[p, g, bp, st, i, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    exec_time = round(time.time() - start_time, 3)

    return (
        "Positive (Diabetes)" if prediction == 1 else "Negative (No Diabetes)",
        float(prob),
        exec_time
    )

# LIME explanation
def explain_with_lime(p, g, bp, st, i, bmi, dpf, age):
    global lime_explainer, scaler, model

    if lime_explainer is None or model is None or scaler is None:
        return "<b>LIME components are not initialized.</b>"

    input_array = np.array([[p, g, bp, st, i, bmi, dpf, age]])
    input_scaled = scaler.transform(input_array)

    lime_exp = lime_explainer.explain_instance(input_scaled[0], model.predict_proba, num_features=5)
    html_exp = lime_exp.as_list()

    table_html = "<h4>LIME Explanation</h4><table border='1'><tr><th>Feature</th><th>Weight</th></tr>"
    for feature, weight in html_exp:
        color = "#90ee90" if weight < 0 else "#f08080"
        table_html += f"<tr style='background:{color}'><td>{feature}</td><td>{round(weight, 4)}</td></tr>"
    table_html += "</table>"
    return table_html

# SHAP explanation
def explain_with_shap(p, g, bp, st, i, bmi, dpf, age):
    global shap_explainer, X

    if shap_explainer is None or X is None:
        return "<b>SHAP explainer or dataset not initialized.</b>"

    try:
        shap_values = shap_explainer.shap_values(X)
    except Exception:
        return "<b>SHAP computation failed for this model.</b>"

    mean_abs_shap = np.abs(shap_values[1]).mean(axis=0).flatten()
    mean_raw_shap = np.array(shap_values[1]).mean(axis=0).flatten()

    directions = ["increases risk" if val > 0 else "decreases risk" if val < 0 else "neutral"
                  for val in mean_raw_shap]

    feature_impact = list(zip(X.columns.tolist(), mean_abs_shap, directions))
    feature_impact.sort(key=lambda x: x[1], reverse=True)

    html = "<h4>Top Features by Mean SHAP Impact</h4><table border='1' style='border-collapse:collapse;'>"
    html += "<tr><th>Rank</th><th>Feature</th><th>Mean SHAP Value</th><th>Effect</th></tr>"
    for i_, (feature, impact, direction) in enumerate(feature_impact, start=1):
        color = "#90ee90" if "decreases" in direction else "#f08080" if "increases" in direction else "#ffffff"
        html += f"<tr style='background-color:{color}'><td>{i_}</td><td>{feature}</td><td>{impact:.4f}</td><td>{direction}</td></tr>"
    html += "</table>"

    return html

# GEMMA explanation (uses pipeline if available)
def explain_with_gemma(g, bmi, i, age, p):
    prompt = f"""Explain how the following clinical features relate to diabetes prediction:\n    - Glucose: {g}\n    - BMI: {bmi}\n    - Insulin: {i}\n    - Age: {age}\n    - Pregnancies: {p}\n    """
    if gemma_pipeline is None:
        return "GEMMA model unavailable in this environment Use GPU."
    try:
        response = gemma_pipeline(prompt, max_new_tokens=150, do_sample=True)[0]['generated_text']
        return response[len(prompt):].strip()
    except Exception as e:
        return f"Error generating GEMMA response: {str(e)}"

# Download report
def download_report(name, pid, date, p, g, bp, st, i, bmi, dpf, age, gemma_output, clinician_comments):
    global model, scaler

    if model is None or scaler is None:
        logging.error("Report generation attempted before model was trained.")
        return None

    encounter_ref = str(uuid.uuid4())[:8].upper()
    input_data = np.array([[p, g, bp, st, i, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    result = "Positive (Diabetes)" if prediction == 1 else "Negative (No Diabetes)"

    report_text = f"""Diabetes Prediction Report\n\nVisit Encounter Reference : {encounter_ref}\nDate : {date}\n\nPatient: {name} (ID: {pid})\n\nPregnancies: {p}\nGlucose: {g}\nBlood Pressure: {bp}\nSkinThickness: {st}\nInsulin: {i}\nBMI: {bmi}\nDPF: {dpf}\nAge: {age}\n\nResult: {result}\n\nGEMMA Explanation:\n{gemma_output}\n\nClinician's Notes:\n{clinician_comments}\n"""

    # Log the report event
    logging.info(
        f"Report generated | Encounter: {encounter_ref} | Patient ID: {pid} | Result: {result}"
    )

    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as tmp_file:
        tmp_file.write(report_text)
        return tmp_file.name
  # Log admin viwe function the report event
def view_logs():
    try:
        with open("app.log", "r") as f:
            logs = f.read()
        return logs if logs else "Log file is empty."
    except FileNotFoundError:
        return "No logs found yet."
    except Exception as e:
        return f"Error reading logs: {e}"
        
def compute_sus_score(responses):
    """
    responses: list of 10 ints (1–5). SUS scoring:
    - Odd items (1,3,5,7,9): score = response - 1
    - Even items (2,4,6,8,10): score = 5 - response
    """
    if not responses or len(responses) != 10:
        return None
    score_sum = 0
    for idx, resp in enumerate(responses, start=1):
        try:
            resp = int(resp)
        except Exception:
            return None
        if idx % 2 == 1:
            score_sum += (resp - 1)
        else:
            score_sum += (5 - resp)
    return round(score_sum * 2.5, 2)


def save_survey_to_log(respondent_name, respondent_email, sus_answers,
                       likert_answers, open_q1, open_q2):
    """
    Save feedback into feedback.log (JSON lines).
    """
    try:
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "respondent_name": respondent_name.strip() if respondent_name else "",
            "respondent_email": respondent_email.strip() if respondent_email else "",
            "sus_answers": [int(x) for x in sus_answers],
            "sus_score": compute_sus_score(sus_answers),
            "likert_answers": {
                "easy_to_navigate": int(likert_answers[0]),
                "shap_lime_clear": int(likert_answers[1]),
                "gemma_increased_trust": int(likert_answers[2]),
                "useful_for_care_decisions": int(likert_answers[3]),
                "overall_trust": int(likert_answers[4]),
            },
            "open_qs": {
                "most_helpful": open_q1.strip() if open_q1 else "",
                "how_improve_explanations": open_q2.strip() if open_q2 else ""
            }
        }

        with open("feedback.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logging.info(f"Saved survey response | id: {record['id']} | SUS: {record['sus_score']}")
        return True, record["sus_score"]

    except Exception as e:
        logging.error(f"Failed to save survey: {e}")
        return False, str(e)


def load_feedback_for_admin():
    """
    Read feedback.log into DataFrame.
    Returns empty DataFrame if no feedback exists.
    """
    rows = []
    if not os.path.exists("feedback.log"):
        return pd.DataFrame()

    try:
        with open("feedback.log", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    flat = {
                        "id": obj.get("id"),
                        "timestamp": obj.get("timestamp"),
                        "name": obj.get("respondent_name"),
                        "email": obj.get("respondent_email"),
                        "sus_score": obj.get("sus_score"),
                    }
                    # Add individual SUS answers
                    sus = obj.get("sus_answers", [])
                    for i in range(10):
                        flat[f"sus_q{i+1}"] = sus[i] if i < len(sus) else None

                    # Add likert answers
                    lik = obj.get("likert_answers", {})
                    for key in ["easy_to_navigate", "shap_lime_clear", "gemma_increased_trust",
                                "useful_for_care_decisions", "overall_trust"]:
                        flat[key] = lik.get(key)

                    # Add open-ended responses
                    open_qs = obj.get("open_qs", {})
                    flat["most_helpful"] = open_qs.get("most_helpful", "")
                    flat["how_improve_explanations"] = open_qs.get("how_improve_explanations", "")

                    rows.append(flat)
                except json.JSONDecodeError:
                    logging.warning("Skipped malformed line in feedback.log")
    except Exception as e:
        logging.error(f"Error loading feedback: {e}")
        return pd.DataFrame()

    return pd.DataFrame(rows)

# === Build Gradio UI ===

today_str = datetime.today().strftime('%Y-%m-%d')

# === Functions used in callbacks ===
# Define all your functions here, e.g., predict_diabetes, explain_with_shap, explain_with_lime, explain_with_gemma,
# download_report, admin_login, load_and_train, upload_dataset, feature_importance_and_correlation,
# calculate_vif, choose_imputation_method, handle_outliers, evaluate_model, view_logs

# === Build Gradio UI ===
with gr.Blocks() as demo:

    # === Help / Survey Buttons ===
    with gr.Row():
        help_btn = gr.Button(" Help / Send Message", variant="secondary")
        survey_btn = gr.Button(" Take Survey", variant="primary")

    help_section = gr.Column(visible=False)
    survey_section = gr.Column(visible=False)

    # -------------------- HELP SECTION --------------------
    with help_section:
        with gr.Tab("Quick-start Guide"):
            with gr.Accordion("Quick-start Guide", open=False):
                gr.Markdown("""
                **Step 1:** Enter patient details (Name, ID, Date).  
                **Step 2:** Fill in the clinical measurements (Pregnancies, Glucose, etc.).  
                **Step 3:** Click **Predict** to get a result.  
                **Step 4:** Use **SHAP** or **LIME** tabs to understand the prediction.  
                **Step 5:** Click **Ask Gemma** to get a Gemma Explanation.  
                **Step 6:** Download a report if needed.  
                """)

        with gr.Tab("FAQ"):
            with gr.Accordion("What dataset does this app use?", open=False):
                gr.Markdown("It uses the **Pima Indians Diabetes Dataset** from Kaggle/UCI.")
            with gr.Accordion("Can I use my own dataset?", open=False):
                gr.Markdown("Yes! Log in as admin and upload your dataset in CSV format.")
            with gr.Accordion("How do I interpret SHAP/LIME outputs?", open=False):
                gr.Markdown("Green values reduce diabetes risk, red values increase it.")

        with gr.Tab("Contact Us"):
            name_cf = gr.Textbox(label="Your Name")
            email_cf = gr.Textbox(label="Your Email")
            msg_cf = gr.Textbox(label="Message", lines=4)

            with gr.Row():
                submit_cf = gr.Button("Submit")
                refresh_cf = gr.Button("Refresh")

            contact_status = gr.Markdown()

            def save_contact(name, email, message):
                try:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open("contact_messages.txt", "a") as f:
                        f.write(f"[{ts}] {name} ({email}): {message}\n")
                    return " Your message has been sent. Thank you!"
                except Exception as e:
                    return f" Failed to send message: {e}"

            submit_cf.click(save_contact, [name_cf, email_cf, msg_cf], contact_status)

            def refresh_form():
                return "", "", ""

            refresh_cf.click(refresh_form, None, [name_cf, email_cf, msg_cf])

        # Close Help button
        close_help_btn = gr.Button("Close Help", variant="secondary")

    # -------------------- SURVEY SECTION --------------------
    with survey_section:
        gr.Markdown("## Quick Survey — System Usability & Trust")

        survey_name = gr.Textbox(label="Your name (optional)")
        survey_email = gr.Textbox(label="Your email (optional)")

        # Example SUS Questions
        sus_q1 = gr.Radio(choices=["1","2","3","4","5"], label="SUS 1: I think I would use this system frequently.")
        sus_q2 = gr.Radio(choices=["1","2","3","4","5"], label="SUS 2: I find the system unnecessarily complex.")
        sus_q3 = gr.Radio(choices=["1","2","3","4","5"], label="SUS 3: I think the system is easy to use.")
        sus_q4 = gr.Radio(choices=["1","2","3","4","5"], label="SUS 4: I think I would need support to use this system.")
        sus_q5 = gr.Radio(choices=["1","2","3","4","5"], label="SUS 5: I find the system well integrated.")
        sus_q6 = gr.Radio(choices=["1","2","3","4","5"], label="SUS 6: I think there is too much inconsistency in this system.")
        sus_q7 = gr.Radio(choices=["1","2","3","4","5"], label="SUS 7: I would imagine most people would learn to use it quickly.")
        sus_q8 = gr.Radio(choices=["1","2","3","4","5"], label="SUS 8: I find the system very cumbersome to use.")
        sus_q9 = gr.Radio(choices=["1","2","3","4","5"], label="SUS 9: I feel very confident using the system.")
        sus_q10 = gr.Radio(choices=["1","2","3","4","5"], label="SUS 10: I needed to learn a lot before I could use the system.")

        # Open-ended Questions
        open_q1 = gr.Textbox(label="What do you like about the system?", lines=3)
        open_q2 = gr.Textbox(label="What can be improved?", lines=3)

        with gr.Row():
            submit_survey_btn = gr.Button("Submit Survey", variant="primary")
            clear_survey_btn = gr.Button("Clear Form", variant="secondary")

        survey_status = gr.Markdown()

        def submit_survey_click(name, email, *answers):
            try:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("feedback.log", "a") as f:   # <--- write to feedback.log
                    f.write(f"[{ts}] Name: {name}, Email: {email}, Answers: {answers}\n")
                return " Thank you — your survey has been saved."
            except Exception as e:
                return f" Error: {e}"
      
        submit_inputs = [
            survey_name, survey_email,
            sus_q1, sus_q2, sus_q3, sus_q4, sus_q5,
            sus_q6, sus_q7, sus_q8, sus_q9, sus_q10,
            open_q1, open_q2
        ]

        submit_survey_btn.click(submit_survey_click, submit_inputs, survey_status)

        def clear_survey():
            return "", "", "3","3","3","3","3","3","3","3","3","3","", ""

        clear_survey_btn.click(clear_survey, None, submit_inputs)

        # Close Survey button
        close_survey_btn = gr.Button("Close Survey", variant="secondary")

    # -------------------- TOGGLE LOGIC --------------------
    help_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        None, [help_section, survey_section]
    )

    close_help_btn.click(lambda: gr.update(visible=False), None, help_section)

    survey_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        None, [help_section, survey_section]
    )

    close_survey_btn.click(lambda: gr.update(visible=False), None, survey_section)


    # === User Interface ===
    gr.Markdown("## Diabetes Prediction System (User Interface)")

    with gr.Group():
        name = gr.Textbox(label="Patient Name")
        pid = gr.Textbox(label="Patient ID")
        date = gr.Textbox(label="Date (YYYY-MM-DD)", value=today_str)
        p = gr.Number(label="Pregnancies (0–20)", value=0, minimum=0, maximum=20)
        g = gr.Number(label="Glucose (0–200 mg/dL)", value=0, minimum=0, maximum=200)
        bp = gr.Number(label="Blood Pressure (0–150 mm Hg)", value=0, minimum=0, maximum=150)
        st = gr.Number(label="Skin Thickness (0–100 mm)", value=0, minimum=0, maximum=100)
        i = gr.Number(label="Insulin (0–900 mu U/ml)", value=0, minimum=0, maximum=900)
        bmi = gr.Number(label="BMI (0–70)", value=0.0, minimum=0, maximum=70)
        dpf = gr.Number(label="DPF (0.0–2.5)", value=0.0, minimum=0.0, maximum=2.5)
        age = gr.Number(label="Age (1–120 years)", value=0, minimum=1, maximum=120)

        with gr.Tabs():
            with gr.TabItem("1 Predict"):
                out_pred = gr.Textbox(label="Prediction")
                out_prob = gr.Textbox(label="Probability of Diabetes")
                out_time = gr.Textbox(label="Execution Time (seconds)")

                gr.Button("Predict").click(
                    predict_diabetes,
                    inputs=[p, g, bp, st, i, bmi, dpf, age],
                    outputs=[out_pred, out_prob, out_time]
                )
            with gr.TabItem("2 SHAP"):
                out_shap = gr.HTML()
                gr.Button("Explain SHAP").click(explain_with_shap, [p, g, bp, st, i, bmi, dpf, age], out_shap)

            with gr.TabItem("3 LIME"):
                out_lime = gr.HTML()
                gr.Button("Explain LIME").click(explain_with_lime, [p, g, bp, st, i, bmi, dpf, age], out_lime)

            with gr.TabItem("4 GEMMA"):
                out_gemma = gr.Textbox()
                gr.Button("Ask GEMMA").click(explain_with_gemma, [g, bmi, i, age, p], out_gemma)

            with gr.TabItem("5 Report"):
                clinician_comments = gr.Textbox(label="Clinician Comments", lines=4, placeholder="Enter any relevant notes...")
                report = gr.File()
                gr.Button("Download Report").click(
                    download_report,
                    inputs=[name, pid, date, p, g, bp, st, i, bmi, dpf, age, out_gemma, clinician_comments],
                    outputs=report
                )

    # === Admin Interface ===
    gr.Markdown("## Admin Login")
    admin_user = gr.Textbox(label="Username")
    admin_pass = gr.Textbox(label="Password", type="password")
    login_btn = gr.Button("Login")
    login_fail = gr.Markdown(" Invalid credentials. Try again.", visible=False)

    # Admin Panel (hidden until login)
    admin_panel = gr.Group(visible=False)
    with admin_panel:
        gr.Markdown("## Admin Interface")

        with gr.Tab("Data Management"):
            gr.Markdown("### Load Dataset")
            load_output = gr.Textbox(label="Load Status", interactive=False)
            data_preview = gr.Dataframe(label="Loaded Dataset", interactive=False)

            dataset_url = DEFAULT_PIMA_URL
            load_url_button = gr.Button("Load Default Dataset from URL")
            load_url_button.click(fn=lambda: load_and_train(dataset_url), inputs=[], outputs=[load_output])

            file_upload = gr.File(label="Upload Your Own Dataset (CSV)")
            load_file_button = gr.Button("Upload from File")
            load_file_button.click(fn=upload_dataset, inputs=[file_upload, gr.Textbox(visible=False)], outputs=[load_output])

        with gr.Tab("EDA"):
            gr.Markdown("### Feature Importance & Correlation Matrix")
            eda_btn = gr.Button("Run EDA")
            feat_imp_plot = gr.Plot(label="Feature Importance")
            corr_plot = gr.Plot(label="Correlation Matrix")
            eda_btn.click(fn=feature_importance_and_correlation, inputs=[], outputs=[feat_imp_plot, corr_plot])

            gr.Markdown("### Variance Inflation Factor (VIF)")
            vif_btn = gr.Button("Calculate VIF")
            vif_output = gr.Dataframe(label="VIF Results", interactive=False)
            vif_btn.click(fn=calculate_vif, inputs=[], outputs=[vif_output])

        with gr.Tab("Data Preprocessing"):
            gr.Markdown("### Imputation Method")
            method = gr.Radio(choices=["Median", "KNN"], label="Select Imputation Method")
            apply_button = gr.Button("Apply")
            imputation_output = gr.Textbox(label="Imputation Status", interactive=False)
            apply_button.click(choose_imputation_method, [method], imputation_output)

            gr.Markdown("### Outlier Handling")
            outlier_method = gr.Radio(choices=["Z-score", "IQR"], label="Select Outlier Handling Method")
            outlier_status = gr.Textbox(label="Outlier Handling Status", interactive=False)
            gr.Button("Handle Outliers").click(handle_outliers, inputs=[outlier_method], outputs=[outlier_status])
        
        with gr.Tab("Hyperparameter Tuning"):
            enable_tuning = gr.Checkbox(label="Enable Hyperparameter Tuning (all combos)", value=ENABLE_TUNING)
            tuning_status = gr.Textbox(label="Tuning Status", interactive=False)

            enable_tuning.change(
                fn=toggle_tuning,
                inputs=[enable_tuning],
                outputs=[tuning_status]
            )    
        
        with gr.Tab("Model Combo Selection"):
            gr.Markdown(
                """
                ### Model Combination Options  
                **Combo 1**: Random Forest + Logistic Regression + Gradient Boosting + Extra Trees  
                **Combo 2**: AdaBoost + Gradient Boosting + Logistic Regression + Extra Trees  
                **Combo 3**: Random Forest + AdaBoost + Logistic Regression + Gradient Boosting
                """
            )

            # --- Dropdown for model combo ---
            model_combo_dropdown = gr.Dropdown(
                choices=["Combo1", "Combo2", "Combo3"],
                value="Combo1",
                label="Select Model Combination"
            )

            # --- Evaluate button ---
            evaluate_btn = gr.Button("Evaluate Models")

            # --- Outputs ---
            metrics_output = gr.Dataframe(label="Training vs Test Performance")
            roc_plot = gr.Plot(label="ROC Curves (Test Set)")
            cm_plot = gr.Plot(label="Confusion Matrices (Test Set)")
            auc_plot = gr.Plot(label="Train vs Test AUC Comparison")
            
            # --- Button wiring ---
            evaluate_btn.click(
                evaluate_model,
                inputs=[model_combo_dropdown],
                outputs=[metrics_output, roc_plot, cm_plot, auc_plot]
            )

            # Legend / explanation text
            legend_text = gr.Markdown(
                """
                ### Legend

                **AUC (Area Under the ROC Curve)**  
                - 0.5 = random guessing  
                - 0.7–0.8 = acceptable discrimination  
                - 0.8–0.9 = very good discrimination  
                - >0.9 = excellent discrimination  

                **Training Metrics**  
                - Show how well the model fits the data it has already seen.  
                - High training scores but much lower test scores → **Overfitting**.  
                - Both train & test low → **Underfitting**.  

                **Test Metrics (most important)**  
                - Reflect how the model performs on unseen data (generalization).  
                - ROC curve → ability to distinguish classes when deployed.  
                - Confusion matrix → shows real-world prediction errors (false positives/negatives).
                """
            )
   
        with gr.Tab("System Logs"):
            gr.Markdown("### Application Logs")
            log_btn = gr.Button("View Logs")
            log_output = gr.Textbox(
                label="Log Output",
                lines=15,
                interactive=False,
                placeholder="Click 'View Logs' to display log entries..."
            )
            log_btn.click(fn=view_logs, inputs=None, outputs=log_output)
        with gr.Tab("Feedback"):
            gr.Markdown("### Survey Feedback (Responses saved to `feedback.log`)")
            load_feedback_btn = gr.Button("Load Feedback")
            refresh_feedback_btn = gr.Button("Refresh")
            export_feedback_btn = gr.Button("Download feedback.log")
            feedback_df = gr.Dataframe(label="Survey Responses", interactive=False)
            feedback_text = gr.Textbox(label="Raw feedback.log (first 1000 chars)", lines=6, interactive=False)

            def load_feedback():
                df = load_feedback_for_admin()
                try:
                    with open("feedback.log", "r", encoding="utf-8") as f:
                        data = f.read(1000)
                        raw = data if data else "No feedback logged yet."
                except FileNotFoundError:
                    raw = "No feedback.log found."
                return df, raw

            load_feedback_btn.click(load_feedback, None, [feedback_df, feedback_text])
            refresh_feedback_btn.click(load_feedback, None, [feedback_df, feedback_text])
            export_feedback_btn.click(lambda: "feedback.log", None, gr.File(label="feedback file"))

               
    # Connect login button inside the Blocks context
    login_btn.click(admin_login, inputs=[admin_user, admin_pass], outputs=[admin_panel, login_fail, gr.State()])

# === Preload default dataset ===
try:
    load_and_train(DEFAULT_PIMA_URL)
except Exception:
    pass

# === Launch App ===
if __name__ == "__main__":
    demo.launch(share=True)








            
