from flask import Flask, render_template

from ml import (
    boosting as boosting_module,
    data_loader,
    dbscan_clustering as dbscan_module,
    decision_tree as decision_tree_module,
    eda as eda_module,
    ensemble_learning as ensemble_module,
    kmeans_clustering as kmeans_module,
    linear_regression as linear_regression_module,
    logistic_regression as logistic_regression_module,
    preprocessing as preprocessing_module,
)


app = Flask(__name__)


MODULE_STATUS = {
    "data_loading": {
        "name": "Data Loading",
        "short_name": "Data Loading",
        "description": "Load the placement CSV and preview rows, columns, data types, and missing values.",
        "endpoint": "data_loading",
        "implemented": data_loader.IS_IMPLEMENTED,
    },
    "eda": {
        "name": "Exploratory Data Analysis",
        "short_name": "EDA",
        "description": "Summary statistics, distributions, correlations, and class balance.",
        "endpoint": "eda",
        "implemented": eda_module.IS_IMPLEMENTED,
    },
    "preprocessing": {
        "name": "Preprocessing & Min-Max Normalization",
        "short_name": "Preprocessing",
        "description": "Imputation, one-hot encoding, Min-Max [0, 1] normalization, and train/test split.",
        "endpoint": "preprocessing",
        "implemented": preprocessing_module.IS_IMPLEMENTED,
    },
    "linear_regression": {
        "name": "Linear Regression & Accuracy",
        "short_name": "Linear Reg",
        "description": "Predict salary package with strict & standard tolerance accuracy, MAPE, and R².",
        "endpoint": "linear_regression",
        "implemented": linear_regression_module.IS_IMPLEMENTED,
    },
    "logistic_regression": {
        "name": "Logistic Regression",
        "short_name": "Logistic Reg",
        "description": "Predict whether a student will be placed with odds ratio and ROC analysis.",
        "endpoint": "logistic_regression",
        "implemented": logistic_regression_module.IS_IMPLEMENTED,
    },
    "decision_tree": {
        "name": "ID3 Decision Tree",
        "short_name": "ID3 Tree",
        "description": "Entropy and Information Gain based tree classifier with visual rule diagrams.",
        "endpoint": "decision_tree",
        "implemented": decision_tree_module.IS_IMPLEMENTED,
    },
    "boosting": {
        "name": "Boosting Algorithms",
        "short_name": "Boosting",
        "description": "Sequential boosting benchmark comparing AdaBoost and Gradient Boosting (GBM).",
        "endpoint": "boosting",
        "implemented": boosting_module.IS_IMPLEMENTED,
    },
    "ensemble_learning": {
        "name": "Ensemble Learning",
        "short_name": "Ensemble",
        "description": "Bagging (Random Forest), Soft Voting, and Stacking meta-models benchmark.",
        "endpoint": "ensemble_learning",
        "implemented": ensemble_module.IS_IMPLEMENTED,
    },
    "kmeans_clustering": {
        "name": "K-Means Clustering",
        "short_name": "K-Means",
        "description": "Unsupervised student persona profiling with Elbow method and Silhouette score.",
        "endpoint": "kmeans_clustering",
        "implemented": kmeans_module.IS_IMPLEMENTED,
    },
    "dbscan_clustering": {
        "name": "DBSCAN Clustering",
        "short_name": "DBSCAN",
        "description": "Density-based clustering finding natural groupings and isolating anomaly noise.",
        "endpoint": "dbscan_clustering",
        "implemented": dbscan_module.IS_IMPLEMENTED,
    },
}


@app.context_processor
def inject_module_status():
    return {"module_status": MODULE_STATUS}


@app.route("/")
def index():
    try:
        data = data_loader.load_data()
        completed_modules = sum(module["implemented"] for module in MODULE_STATUS.values())
        overview = {
            "rows": len(data),
            "columns": len(data.columns),
            "missing_values": int(data.isna().sum().sum()),
            "placed_percent": round(data["PlacementStatus"].mean() * 100, 1),
            "completed_modules": completed_modules,
            "total_modules": len(MODULE_STATUS),
        }
    except FileNotFoundError:
        overview = None
    return render_template("index.html", modules=MODULE_STATUS.values(), overview=overview)


@app.route("/data-loading")
def data_loading():
    try:
        data = data_loader.load_data()
        summary = data_loader.get_data_summary(data)
        return render_template("data_loading.html", summary=summary, error=None)
    except FileNotFoundError as error:
        return render_template("data_loading.html", summary=None, error=str(error))


@app.route("/eda")
def eda():
    try:
        return render_template("eda.html", result=eda_module.get_eda_results(), error=None)
    except FileNotFoundError as error:
        return render_template("eda.html", result=None, error=str(error))


@app.route("/preprocessing")
def preprocessing():
    try:
        return render_template(
            "preprocessing.html", result=preprocessing_module.get_preprocessing_summary(), error=None
        )
    except FileNotFoundError as error:
        return render_template("preprocessing.html", result=None, error=str(error))


@app.route("/linear-regression")
def linear_regression():
    try:
        return render_template(
            "linear_regression.html",
            result=linear_regression_module.run_linear_regression(),
            error=None,
        )
    except FileNotFoundError as error:
        return render_template("linear_regression.html", result=None, error=str(error))


@app.route("/logistic-regression")
def logistic_regression():
    try:
        return render_template(
            "logistic_regression.html",
            result=logistic_regression_module.run_logistic_regression(),
            error=None,
        )
    except FileNotFoundError as error:
        return render_template("logistic_regression.html", result=None, error=str(error))


@app.route("/decision-tree")
def decision_tree():
    try:
        return render_template(
            "decision_tree.html",
            result=decision_tree_module.run_id3_decision_tree(),
            error=None,
        )
    except FileNotFoundError as error:
        return render_template("decision_tree.html", result=None, error=str(error))


@app.route("/boosting")
def boosting():
    try:
        return render_template(
            "boosting.html",
            result=boosting_module.run_boosting(),
            error=None,
        )
    except FileNotFoundError as error:
        return render_template("boosting.html", result=None, error=str(error))


@app.route("/ensemble-learning")
def ensemble_learning():
    try:
        return render_template(
            "ensemble_learning.html",
            result=ensemble_module.run_ensemble_learning(),
            error=None,
        )
    except FileNotFoundError as error:
        return render_template("ensemble_learning.html", result=None, error=str(error))


@app.route("/kmeans-clustering")
def kmeans_clustering():
    try:
        return render_template(
            "kmeans_clustering.html",
            result=kmeans_module.run_kmeans_clustering(),
            error=None,
        )
    except FileNotFoundError as error:
        return render_template("kmeans_clustering.html", result=None, error=str(error))


@app.route("/dbscan-clustering")
def dbscan_clustering():
    try:
        return render_template(
            "dbscan_clustering.html",
            result=dbscan_module.run_dbscan_clustering(),
            error=None,
        )
    except FileNotFoundError as error:
        return render_template("dbscan_clustering.html", result=None, error=str(error))


if __name__ == "__main__":
    app.run(debug=True)
