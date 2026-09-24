# Retail Customer Segmentation Using Purchase Behaviour

## 📌 Project Overview

Retail Customer Segmentation Using Purchase Behaviour is a machine learning project that groups customers into meaningful segments based on their purchasing behaviour.

The project uses customer-level purchasing features such as:

- Number of Orders
- Total Quantity Purchased
- Total Spending
- Average Order Value

Three unsupervised machine learning algorithms are implemented:

1. K-Means Clustering
2. Hierarchical Clustering
3. DBSCAN Clustering

The project also includes a Flask-based web application that allows users to enter customer purchase information and obtain a predicted customer segment.

---

## 🎯 Objectives

The main objectives of this project are:

- To analyze customer purchasing behaviour.
- To preprocess and clean retail transaction data.
- To generate customer-level purchasing features.
- To apply different clustering algorithms.
- To compare clustering performance using Silhouette Score.
- To identify meaningful customer segments.
- To develop a web-based customer segmentation application.
- To provide an interactive dashboard for viewing customer statistics and clustering results.

---

## 📊 Dataset

The project uses the **Online Retail Dataset**.

### Original Dataset

- Number of records: 541,909
- Number of features: 8

### Dataset Features

| Feature | Description |
|---|---|
| InvoiceNo | Invoice number |
| StockCode | Product code |
| Description | Product description |
| Quantity | Quantity purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per unit |
| CustomerID | Customer identification number |
| Country | Customer country |

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Removed records with missing CustomerID.
2. Removed cancelled invoices.
3. Removed records with Quantity less than or equal to zero.
4. Removed records with UnitPrice less than or equal to zero.
5. Removed duplicate records.
6. Calculated total transaction amount.

The formula used was:

```text
TotalAmount = Quantity × UnitPrice
