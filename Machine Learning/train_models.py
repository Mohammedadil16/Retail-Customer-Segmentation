# ============================================================
# RETAIL CUSTOMER SEGMENTATION
# K-MEANS + HIERARCHICAL CLUSTERING + DBSCAN
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)
from sklearn.metrics import silhouette_score


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Datasets",
    "Online Retail.xlsx"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "Model files"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


print()
print("=" * 60)
print("RETAIL CUSTOMER SEGMENTATION")
print("=" * 60)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print()
print("Loading dataset...")

df = pd.read_excel(
    DATASET_PATH
)

print(
    "Original dataset shape:",
    df.shape
)


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

print()
print("Preprocessing data...")

# Remove customers with missing CustomerID
df = df.dropna(
    subset=["CustomerID"]
)

# Remove cancelled invoices
df = df[
    ~df["InvoiceNo"]
    .astype(str)
    .str.startswith("C")
]

# Keep only positive quantity and price
df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
]

# Remove duplicate rows
df = df.drop_duplicates()


print(
    "After preprocessing:",
    df.shape
)


# ============================================================
# 4. CALCULATE TOTAL AMOUNT
# ============================================================

df["TotalAmount"] = (
    df["Quantity"] *
    df["UnitPrice"]
)


# ============================================================
# 5. CUSTOMER LEVEL DATA
# ============================================================

print()
print("Creating customer-level features...")


customer_data = df.groupby(
    "CustomerID"
).agg({

    "InvoiceNo": "nunique",

    "Quantity": "sum",

    "TotalAmount": "sum"

}).reset_index()


# Rename columns

customer_data.columns = [

    "CustomerID",

    "NumberOfOrders",

    "TotalQuantity",

    "TotalSpending"

]


# ============================================================
# 6. AVERAGE ORDER VALUE
# ============================================================

customer_data[
    "AverageOrderValue"
] = (

    customer_data[
        "TotalSpending"
    ]

    /

    customer_data[
        "NumberOfOrders"
    ]

)


# ============================================================
# 7. SELECT FEATURES
# ============================================================

features = [

    "NumberOfOrders",

    "TotalQuantity",

    "TotalSpending",

    "AverageOrderValue"

]


X = customer_data[
    features
]


print()
print(
    "Number of customers:",
    len(customer_data)
)

print()
print("Features used:")

for feature in features:

    print(
        " -",
        feature
    )


# ============================================================
# 8. SCALE FEATURES
# ============================================================

print()
print("Scaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ============================================================
# 9. K-MEANS CLUSTERING
# ============================================================

print()
print("=" * 60)
print("K-MEANS CLUSTERING")
print("=" * 60)


kmeans = KMeans(

    n_clusters=4,

    random_state=42,

    n_init=10

)


kmeans_labels = (
    kmeans.fit_predict(
        X_scaled
    )
)


kmeans_score = silhouette_score(
    X_scaled,
    kmeans_labels
)


print(
    "Number of clusters:",
    len(
        set(kmeans_labels)
    )
)

print(
    "Silhouette Score:",
    round(
        kmeans_score,
        4
    )
)


# ============================================================
# 10. HIERARCHICAL CLUSTERING
# ============================================================

print()
print("=" * 60)
print("HIERARCHICAL CLUSTERING")
print("=" * 60)


hierarchical = AgglomerativeClustering(

    n_clusters=4

)


hierarchical_labels = (
    hierarchical.fit_predict(
        X_scaled
    )
)


hierarchical_score = (
    silhouette_score(
        X_scaled,
        hierarchical_labels
    )
)


print(
    "Number of clusters:",
    len(
        set(
            hierarchical_labels
        )
    )
)

print(
    "Silhouette Score:",
    round(
        hierarchical_score,
        4
    )
)


# ============================================================
# CREATE HIERARCHICAL CLUSTER CENTERS
# ============================================================

hierarchical_centers = []


for cluster in sorted(
    set(
        hierarchical_labels
    )
):

    points = X_scaled[
        hierarchical_labels == cluster
    ]

    center = points.mean(
        axis=0
    )

    hierarchical_centers.append(
        center
    )


hierarchical_centers = np.array(
    hierarchical_centers
)


# ============================================================
# 11. DBSCAN AUTOMATIC PARAMETER SEARCH
# ============================================================

print()
print("=" * 60)
print("DBSCAN CLUSTERING")
print("=" * 60)

print()
print(
    "Searching for suitable DBSCAN parameters..."
)


eps_values = [

    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.2,
    1.5,
    2.0

]


min_samples_values = [

    3,
    5,
    8,
    10

]


best_dbscan = None

best_dbscan_labels = None

best_dbscan_clusters = []

best_dbscan_score = -999

best_eps = None

best_min_samples = None


# ============================================================
# TRY DIFFERENT DBSCAN PARAMETERS
# ============================================================

for eps in eps_values:

    for min_samples in min_samples_values:

        model = DBSCAN(

            eps=eps,

            min_samples=min_samples

        )


        labels = model.fit_predict(
            X_scaled
        )


        # Remove noise
        valid_mask = (
            labels != -1
        )


        valid_labels = (
            labels[
                valid_mask
            ]
        )


        valid_data = (
            X_scaled[
                valid_mask
            ]
        )


        unique_clusters = set(
            valid_labels
        )


        # Need at least 2 clusters
        if len(
            unique_clusters
        ) < 2:

            continue


        # Need enough data
        if len(
            valid_data
        ) < 10:

            continue


        try:

            silhouette = (
                silhouette_score(
                    valid_data,
                    valid_labels
                )
            )

        except:

            continue


        # Calculate noise
        noise_percentage = (

            np.sum(
                labels == -1
            )

            /

            len(labels)

        )


        # Penalize too much noise
        final_score = (

            silhouette

            -

            (
                noise_percentage
                * 0.30
            )

        )


        if (
            final_score
            >
            best_dbscan_score
        ):

            best_dbscan_score = (
                final_score
            )

            best_dbscan = model

            best_dbscan_labels = (
                labels
            )

            best_dbscan_clusters = (
                sorted(
                    list(
                        unique_clusters
                    )
                )
            )

            best_eps = eps

            best_min_samples = (
                min_samples
            )


# ============================================================
# DBSCAN FALLBACK
# ============================================================

if best_dbscan is None:

    print(
        "Automatic DBSCAN search failed."
    )

    print(
        "Using default DBSCAN parameters."
    )


    best_eps = 0.8

    best_min_samples = 5


    best_dbscan = DBSCAN(

        eps=best_eps,

        min_samples=best_min_samples

    )


    best_dbscan_labels = (
        best_dbscan.fit_predict(
            X_scaled
        )
    )


    best_dbscan_clusters = sorted(

        [

            cluster

            for cluster in set(
                best_dbscan_labels
            )

            if cluster != -1

        ]

    )


# ============================================================
# DBSCAN CLUSTER CENTERS
# ============================================================

dbscan_centers = []


for cluster in (
    best_dbscan_clusters
):

    points = X_scaled[

        best_dbscan_labels
        == cluster

    ]


    center = points.mean(
        axis=0
    )


    dbscan_centers.append(
        center
    )


if len(
    dbscan_centers
) > 0:

    dbscan_centers = np.array(
        dbscan_centers
    )

else:

    dbscan_centers = np.empty(
        (
            0,
            X_scaled.shape[1]
        )
    )


# ============================================================
# DBSCAN RESULTS
# ============================================================

dbscan_noise = np.sum(

    best_dbscan_labels
    == -1

)


dbscan_noise_percentage = (

    dbscan_noise

    /

    len(
        best_dbscan_labels
    )

) * 100


print()
print(
    "Best EPS:",
    best_eps
)

print(
    "Best Min Samples:",
    best_min_samples
)

print(
    "Number of Clusters:",
    len(
        best_dbscan_clusters
    )
)

print(
    "Noise Points:",
    dbscan_noise
)

print(
    "Noise Percentage:",
    round(
        dbscan_noise_percentage,
        2
    ),
    "%"
)


# Final DBSCAN silhouette score

if len(
    best_dbscan_clusters
) >= 2:

    valid_mask = (

        best_dbscan_labels
        != -1

    )


    final_dbscan_score = (

        silhouette_score(

            X_scaled[
                valid_mask
            ],

            best_dbscan_labels[
                valid_mask
            ]

        )

    )


    print(
        "Silhouette Score:",
        round(
            final_dbscan_score,
            4
        )
    )


# ============================================================
# 12. SAVE CUSTOMER CLUSTER LABELS
# ============================================================

print()
print("=" * 60)
print("ADDING CLUSTER LABELS")
print("=" * 60)


customer_data[
    "KMeans_Cluster"
] = kmeans_labels


customer_data[
    "Hierarchical_Cluster"
] = hierarchical_labels


customer_data[
    "DBSCAN_Cluster"
] = best_dbscan_labels


# ============================================================
# 13. SAVE CUSTOMER DATA
# ============================================================

customer_data_path = os.path.join(

    MODEL_DIR,

    "customer_data.csv"

)


customer_data.to_csv(

    customer_data_path,

    index=False

)


print(
    "Customer data saved:"
)

print(
    customer_data_path
)


# ============================================================
# 14. SAVE SCALER
# ============================================================

print()
print(
    "Saving scaler..."
)


joblib.dump(

    scaler,

    os.path.join(

        MODEL_DIR,

        "scaler.pkl"

    )

)


# ============================================================
# 15. SAVE K-MEANS MODEL
# ============================================================

print(
    "Saving K-Means model..."
)


joblib.dump(

    kmeans,

    os.path.join(

        MODEL_DIR,

        "kmeans.pkl"

    )

)


# ============================================================
# 16. SAVE HIERARCHICAL MODEL
# ============================================================

print(
    "Saving Hierarchical model..."
)


joblib.dump(

    {

        "model":
            hierarchical,

        "centers":
            hierarchical_centers

    },

    os.path.join(

        MODEL_DIR,

        "hierarchical.pkl"

    )

)


# ============================================================
# 17. SAVE DBSCAN MODEL
# ============================================================

print(
    "Saving DBSCAN model..."
)


joblib.dump(

    {

        "model":
            best_dbscan,

        "centers":
            dbscan_centers,

        "clusters":
            best_dbscan_clusters

    },

    os.path.join(

        MODEL_DIR,

        "dbscan.pkl"

    )

)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)


print(
    "Customers:",
    len(customer_data)
)


print(
    "K-Means clusters:",
    len(
        set(
            kmeans_labels
        )
    )
)


print(
    "Hierarchical clusters:",
    len(
        set(
            hierarchical_labels
        )
    )
)


print(
    "DBSCAN clusters:",
    len(
        best_dbscan_clusters
    )
)


print(
    "DBSCAN noise points:",
    dbscan_noise
)


print()
print(
    "Model files created successfully!"
)


print()
print(
    "Files saved in:"
)

print(
    MODEL_DIR
)

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print("\n" + "="*60)
print("CUSTOMER LEVEL FEATURES")
print("="*60)

print(customer_data.head(10).to_string(index=False))