# Customer Segmentation using K-Means Clustering and PCA

## Objective

Develop a K-Means clustering model to segment mall customers based on their annual income and spending behavior and visualize the clusters using Principal Component Analysis (PCA).

---

## Dataset

https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

---

## Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn

---

## Methodology

1. Load the dataset.
2. Explore the data.
3. Remove CustomerID.
4. Encode Gender.
5. Standardize the features.
6. Use the Elbow Method to determine the optimal K.
7. Train the K-Means clustering model.
8. Assign cluster labels.
9. Apply PCA for dimensionality reduction.
10. Visualize the customer segments.

---

## Results

- Optimal Number of Clusters: **5**
- Generated Elbow Curve.
- Generated Customer Cluster Visualization.
- Generated PCA Visualization.

---

## Observations

- The Elbow Method indicates that **K = 5** is the optimal number of customer groups.
- PCA effectively reduces high-dimensional customer data into two principal components while preserving most of the information.
- Customers with similar annual income and spending patterns are grouped into the same cluster.
- The identified customer segments can be used for personalized marketing strategies and business decision-making.

---

## Conclusion

This project successfully applied K-Means Clustering to segment customers based on demographic and purchasing characteristics. The Elbow Method suggested that five clusters provide a suitable balance between simplicity and cluster quality. PCA reduced the multidimensional dataset into two principal components, making cluster visualization easier to interpret. Customer segmentation enables businesses to identify groups such as high-income high-spending customers, budget-conscious customers, and average shoppers, allowing for targeted marketing campaigns. One limitation of K-Means is that the number of clusters must be specified beforehand and it is sensitive to initial centroid selection. PCA improves visualization by reducing dimensionality while retaining most of the important information.
