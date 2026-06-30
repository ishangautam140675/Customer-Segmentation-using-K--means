# CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING

## 1. Introduction

Customer segmentation is a technique used by businesses to divide customers into groups based on their behavior and characteristics. This helps companies understand their customers better and create targeted marketing strategies.

In this project, we use K-Means Clustering, an unsupervised machine learning algorithm, to group customers based on their annual income and spending score.

---

## 2. Objective

* To understand clustering in machine learning
* To group customers into different segments
* To apply the K-Means algorithm
* To determine the optimal number of clusters using the Elbow Method

---

## 3. Technology Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## 4. Dataset Description

The dataset contains:

* Customer ID
* Annual Income (k$)
* Spending Score (1–100)

---

## 5. Methodology

### Step 1: Load Dataset

### Step 2: Select Features

### Step 3: Use Elbow Method

### Step 4: Apply K-Means Clustering

### Step 5: Visualize Clusters

---

## 6. Algorithm Used: K-Means Clustering

K-Means is an unsupervised learning algorithm that groups data into K clusters based on similarity.

Steps:

1. Choose number of clusters (K)
2. Assign data points to nearest cluster
3. Update cluster centroids
4. Repeat until stable

---

## 7. Implementation (Code)

```python id="kmeanscode1"
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("customer_data.csv")

print("Dataset Preview:")
print(data.head())

# Select features
X = data[['AnnualIncome', 'SpendingScore']]

# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply K-Means (choose K=5)
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster to dataset
data['Cluster'] = y_kmeans

# Plot clusters
plt.scatter(X['AnnualIncome'], X['SpendingScore'], c=y_kmeans)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.show()
```

---

## 8. Results

The customers are divided into different clusters based on their income and spending behavior. The elbow method helps determine the optimal number of clusters.

---

## 9. Advantages

* Helps businesses target customers
* Simple and efficient
* Works well with large datasets

---

## 10. Limitations

* Need to choose K manually
* Sensitive to initial values
* Not suitable for non-spherical clusters

---

## 11. Conclusion

This project shows how clustering can be used to segment customers. It is useful in marketing, sales, and business decision-making.

---

## 12. Future Scope

* Use more features (age, gender)
* Apply advanced clustering algorithms
* Deploy as a web app
