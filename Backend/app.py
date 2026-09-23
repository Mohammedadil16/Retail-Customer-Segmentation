from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import os
import numpy as np
import pandas as pd
import joblib


# ============================================================
# FLASK APPLICATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "Frontend"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "Model files"
)


app = Flask(__name__)
CORS(app)


# ============================================================
# MODEL FILE PATHS
# ============================================================

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

KMEANS_PATH = os.path.join(
    MODEL_DIR,
    "kmeans.pkl"
)

HIERARCHICAL_PATH = os.path.join(
    MODEL_DIR,
    "hierarchical.pkl"
)

DBSCAN_PATH = os.path.join(
    MODEL_DIR,
    "dbscan.pkl"
)

CUSTOMER_DATA_PATH = os.path.join(
    MODEL_DIR,
    "customer_data.csv"
)


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURES = [
    "NumberOfOrders",
    "TotalQuantity",
    "TotalSpending",
    "AverageOrderValue"
]


SEGMENT_NAMES = [
    "Low Value Customer",
    "Regular Customer",
    "High Value Customer",
    "Premium Customer"
]


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 60)
print("LOADING RETAIL CUSTOMER SEGMENTATION MODELS")
print("=" * 60)


try:

    scaler = joblib.load(
        SCALER_PATH
    )

    kmeans = joblib.load(
        KMEANS_PATH
    )

    hierarchical_data = joblib.load(
        HIERARCHICAL_PATH
    )

    dbscan_data = joblib.load(
        DBSCAN_PATH
    )

    customer_data = pd.read_csv(
        CUSTOMER_DATA_PATH
    )

    print("Scaler loaded successfully")
    print("K-Means loaded successfully")
    print("Hierarchical model loaded successfully")
    print("DBSCAN model loaded successfully")
    print("Customer data loaded successfully")

except Exception as e:

    print()
    print("ERROR LOADING MODEL FILES")
    print(e)

    raise e


# ============================================================
# EXTRACT HIERARCHICAL MODEL
# ============================================================

if isinstance(hierarchical_data, dict):

    hierarchical = hierarchical_data["model"]

    hierarchical_centers = np.asarray(
        hierarchical_data["centers"]
    )

else:

    hierarchical = hierarchical_data

    hierarchical_centers = np.asarray(
        hierarchical.cluster_centers_
    )


# ============================================================
# EXTRACT DBSCAN MODEL
# ============================================================

if isinstance(dbscan_data, dict):

    dbscan = dbscan_data["model"]

    dbscan_centers = np.asarray(
        dbscan_data["centers"]
    )

else:

    dbscan = dbscan_data

    if hasattr(dbscan, "cluster_centers_"):

        dbscan_centers = np.asarray(
            dbscan.cluster_centers_
        )

    else:

        dbscan_centers = np.array([])


# ============================================================
# CREATE K-MEANS SEGMENT MAPPING
# ============================================================

def create_kmeans_mapping():

    if "KMeans_Cluster" not in customer_data.columns:

        return {}

    profile = (
        customer_data
        .groupby("KMeans_Cluster")[FEATURES]
        .mean()
    )

    sorted_clusters = (
        profile["TotalSpending"]
        .sort_values()
        .index
        .tolist()
    )

    mapping = {}

    for i, cluster in enumerate(sorted_clusters):

        if i < len(SEGMENT_NAMES):

            mapping[int(cluster)] = SEGMENT_NAMES[i]

        else:

            mapping[int(cluster)] = SEGMENT_NAMES[-1]

    return mapping


# ============================================================
# CREATE HIERARCHICAL SEGMENT MAPPING
# ============================================================

def create_hierarchical_mapping():

    if "Hierarchical_Cluster" not in customer_data.columns:

        return {}

    profile = (
        customer_data
        .groupby("Hierarchical_Cluster")[FEATURES]
        .mean()
    )

    sorted_clusters = (
        profile["TotalSpending"]
        .sort_values()
        .index
        .tolist()
    )

    mapping = {}

    for i, cluster in enumerate(sorted_clusters):

        if i < len(SEGMENT_NAMES):

            mapping[int(cluster)] = SEGMENT_NAMES[i]

        else:

            mapping[int(cluster)] = SEGMENT_NAMES[-1]

    return mapping


# ============================================================
# CREATE DBSCAN SEGMENT MAPPING
# ============================================================

def create_dbscan_mapping():

    if "DBSCAN_Cluster" not in customer_data.columns:

        return {}

    profile_data = customer_data[
        customer_data["DBSCAN_Cluster"] != -1
    ]

    if profile_data.empty:

        return {}

    profile = (
        profile_data
        .groupby("DBSCAN_Cluster")[FEATURES]
        .mean()
    )

    sorted_clusters = (
        profile["TotalSpending"]
        .sort_values()
        .index
        .tolist()
    )

    mapping = {}

    if len(sorted_clusters) == 1:

        mapping[
            int(sorted_clusters[0])
        ] = "Regular Customer"

    elif len(sorted_clusters) >= 2:

        mapping[
            int(sorted_clusters[0])
        ] = "Low Value Customer"

        mapping[
            int(sorted_clusters[-1])
        ] = "High Value Customer"

        for cluster in sorted_clusters[1:-1]:

            mapping[
                int(cluster)
            ] = "Regular Customer"

    return mapping


# ============================================================
# BUILD MAPPINGS
# ============================================================

kmeans_mapping = create_kmeans_mapping()

hierarchical_mapping = create_hierarchical_mapping()

dbscan_mapping = create_dbscan_mapping()


# ============================================================
# DISPLAY MAPPINGS
# ============================================================

print()
print("K-MEANS SEGMENT MAPPING")

for cluster, name in kmeans_mapping.items():

    print(
        f"Cluster {cluster + 1} -> {name}"
    )


print()
print("HIERARCHICAL SEGMENT MAPPING")

for cluster, name in hierarchical_mapping.items():

    print(
        f"Cluster {cluster + 1} -> {name}"
    )


print()
print("DBSCAN SEGMENT MAPPING")

for cluster, name in dbscan_mapping.items():

    print(
        f"Cluster {cluster + 1} -> {name}"
    )


# ============================================================
# FRONTEND
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


@app.route("/style.css")
def style_css():

    return send_from_directory(
        FRONTEND_DIR,
        "style.css"
    )


@app.route("/script.js")
def script_js():

    return send_from_directory(
        FRONTEND_DIR,
        "script.js"
    )


# ============================================================
# API STATUS
# ============================================================

@app.route("/api")
def api_status():

    return jsonify({

        "success": True,

        "message":
            "Retail Customer Segmentation API is running!",

        "algorithms": [

            "K-Means",
            "Hierarchical Clustering",
            "DBSCAN"

        ]

    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No input data received."

            }), 400


        # ----------------------------------------------------
        # INPUT VALUES
        # ----------------------------------------------------

        number_of_orders = float(
            data["number_of_orders"]
        )

        total_quantity = float(
            data["total_quantity"]
        )

        total_spending = float(
            data["total_spending"]
        )

        average_order_value = float(
            data["average_order_value"]
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if (
            number_of_orders <= 0
            or total_quantity <= 0
            or total_spending <= 0
            or average_order_value <= 0
        ):

            return jsonify({

                "success": False,

                "error":
                    "All values must be greater than zero."

            }), 400


        # ----------------------------------------------------
        # CREATE CUSTOMER DATAFRAME
        # ----------------------------------------------------

        customer = pd.DataFrame(

            [[
                number_of_orders,
                total_quantity,
                total_spending,
                average_order_value
            ]],

            columns=FEATURES

        )


        # ----------------------------------------------------
        # SCALE INPUT
        # ----------------------------------------------------

        customer_scaled = scaler.transform(
            customer
        )


        # ====================================================
        # K-MEANS
        # ====================================================

        kmeans_cluster = int(

            kmeans.predict(
                customer_scaled
            )[0]

        )

        kmeans_segment = (
            kmeans_mapping.get(
                kmeans_cluster,
                "Regular Customer"
            )
        )


        # ====================================================
        # HIERARCHICAL
        # ====================================================

        hierarchical_distances = np.linalg.norm(

            hierarchical_centers
            - customer_scaled,

            axis=1

        )

        hierarchical_cluster = int(

            np.argmin(
                hierarchical_distances
            )

        )

        hierarchical_segment = (
            hierarchical_mapping.get(
                hierarchical_cluster,
                "Regular Customer"
            )
        )


        # ====================================================
        # DBSCAN
        # ====================================================

        if len(dbscan_centers) > 0:

            dbscan_distances = np.linalg.norm(

                dbscan_centers
                - customer_scaled,

                axis=1

            )

            dbscan_cluster = int(

                np.argmin(
                    dbscan_distances
                )

            )

            dbscan_segment = (
                dbscan_mapping.get(
                    dbscan_cluster,
                    "Regular Customer"
                )
            )

        else:

            dbscan_cluster = -1

            dbscan_segment = "Noise"


        # ====================================================
        # FINAL CUSTOMER SEGMENT
        # ====================================================

        final_segment = kmeans_segment


        # ====================================================
        # RESPONSE
        # ====================================================

        return jsonify({

            "success": True,

            "input": {

                "number_of_orders":
                    number_of_orders,

                "total_quantity":
                    total_quantity,

                "total_spending":
                    total_spending,

                "average_order_value":
                    average_order_value

            },

            "prediction": {

                "kmeans":
                    f"Cluster {kmeans_cluster + 1}",

                "kmeans_segment":
                    kmeans_segment,

                "hierarchical":
                    f"Cluster {hierarchical_cluster + 1}",

                "hierarchical_segment":
                    hierarchical_segment,

                "dbscan":
                    (
                        f"Cluster {dbscan_cluster + 1}"
                        if dbscan_cluster >= 0
                        else "Noise"
                    ),

                "dbscan_segment":
                    dbscan_segment,

                "customer_type":
                    final_segment

            }

        })


    except KeyError as e:

        return jsonify({

            "success": False,

            "error":
                f"Missing input field: {str(e)}"

        }), 400


    except ValueError as e:

        return jsonify({

            "success": False,

            "error":
                f"Invalid input value: {str(e)}"

        }), 400


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# DASHBOARD API
# ============================================================

@app.route(
    "/dashboard",
    methods=["GET"]
)
def dashboard():

    try:

        df = pd.read_csv(
            CUSTOMER_DATA_PATH
        )


        # ====================================================
        # BASIC STATISTICS
        # ====================================================

        total_customers = len(df)

        total_orders = int(
            df["NumberOfOrders"].sum()
        )

        total_quantity = int(
            df["TotalQuantity"].sum()
        )

        total_spending = float(
            df["TotalSpending"].sum()
        )

        average_order_value = float(
            df["AverageOrderValue"].mean()
        )


        # ====================================================
        # K-MEANS DISTRIBUTION
        # ====================================================

        if "KMeans_Cluster" in df.columns:

            kmeans_counts = {}

            for cluster in sorted(
                df["KMeans_Cluster"].unique()
            ):

                kmeans_counts[
                    f"Cluster {int(cluster) + 1}"
                ] = int(
                    (
                        df["KMeans_Cluster"]
                        == cluster
                    ).sum()
                )

        else:

            scaled_data = scaler.transform(
                df[FEATURES]
            )

            labels = kmeans.predict(
                scaled_data
            )

            kmeans_counts = {}

            for cluster in sorted(
                np.unique(labels)
            ):

                kmeans_counts[
                    f"Cluster {int(cluster) + 1}"
                ] = int(
                    (labels == cluster).sum()
                )


        # ====================================================
        # HIERARCHICAL DISTRIBUTION
        # ====================================================

        if "Hierarchical_Cluster" in df.columns:

            hierarchical_counts = {}

            for cluster in sorted(
                df["Hierarchical_Cluster"].unique()
            ):

                hierarchical_counts[
                    f"Cluster {int(cluster) + 1}"
                ] = int(
                    (
                        df["Hierarchical_Cluster"]
                        == cluster
                    ).sum()
                )

        else:

            scaled_data = scaler.transform(
                df[FEATURES]
            )

            distances = np.linalg.norm(

                scaled_data[:, np.newaxis, :]
                -
                hierarchical_centers[
                    np.newaxis, :, :
                ],

                axis=2

            )

            labels = np.argmin(
                distances,
                axis=1
            )

            hierarchical_counts = {}

            for cluster in sorted(
                np.unique(labels)
            ):

                hierarchical_counts[
                    f"Cluster {int(cluster) + 1}"
                ] = int(
                    (labels == cluster).sum()
                )


        # ====================================================
        # DBSCAN DISTRIBUTION
        # ====================================================

        if "DBSCAN_Cluster" in df.columns:

            dbscan_counts = {}

            for cluster in sorted(
                df["DBSCAN_Cluster"].unique()
            ):

                if int(cluster) == -1:

                    label = "Noise"

                else:

                    label = (
                        f"Cluster {int(cluster) + 1}"
                    )

                dbscan_counts[label] = int(
                    (
                        df["DBSCAN_Cluster"]
                        == cluster
                    ).sum()
                )

        else:

            dbscan_counts = {}


        # ====================================================
        # SILHOUETTE SCORES
        # ====================================================

        silhouette_scores = {

            "K-Means": 0.9089,

            "Hierarchical": 0.9135,

            "DBSCAN": 0.8416

        }


        # ====================================================
        # RESPONSE
        # ====================================================

        return jsonify({

            "success": True,

            "statistics": {

                "total_customers":
                    total_customers,

                "total_orders":
                    total_orders,

                "total_quantity":
                    total_quantity,

                "total_spending":
                    round(
                        total_spending,
                        2
                    ),

                "average_order_value":
                    round(
                        average_order_value,
                        2
                    )

            },

            "kmeans":
                kmeans_counts,

            "hierarchical":
                hierarchical_counts,

            "dbscan":
                dbscan_counts,

            "silhouette_scores":
                silhouette_scores

        })


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("RETAIL CUSTOMER SEGMENTATION APPLICATION")
    print("=" * 60)

    print(
        "Website:"
        " http://127.0.0.1:5000/"
    )

    print(
        "API:"
        " http://127.0.0.1:5000/api"
    )

    print(
        "Status: Ready"
    )

    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )