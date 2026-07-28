import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv("Mall_Customers.csv")

print("First Five Records")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nSummary Statistics")
print(df.describe())

print("\nNumerical Features")
print(df.select_dtypes(include=["int64", "float64"]).columns.tolist())

print("\nCategorical Features")
print(df.select_dtypes(include=["object"]).columns.tolist())

print("\nMissing Values")
print(df.isnull().sum())

df.drop("CustomerID", axis=1, inplace=True)

encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)

plt.savefig("elbow_curve.png")
plt.show()

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(scaled_data)

df["Cluster"] = clusters

plt.figure(figsize=(8,6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis"
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")

plt.savefig("customer_clusters.png")

plt.show()

pca = PCA(n_components=2)

components = pca.fit_transform(scaled_data)

plt.figure(figsize=(8,6))

plt.scatter(
    components[:,0],
    components[:,1],
    c=df["Cluster"],
    cmap="viridis"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization")

plt.savefig("pca_clusters.png")

plt.show()

print("\nObservations")
print("1. The Elbow Method suggests K = 5 as the optimal number of clusters.")
print("2. PCA reduces high-dimensional data into two dimensions for easier visualization.")
print("3. Customers are grouped based on similar income and spending behavior.")
print("4. Different clusters represent different marketing target groups.")
