import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load Iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Display first few rows
    print("First five rows of dataset:")
    print(df.head())

    # Check structure and missing values
    print("\nDataset info:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())

    # Clean dataset (no missing values in Iris, but code included)
    df = df.dropna()

except FileNotFoundError as e:
    print("Error loading dataset:", e)

# Task 2: Basic Data Analysis
print("\nBasic statistics:")
print(df.describe())

# Grouping by species and compute mean of numerical columns
grouped = df.groupby("target").mean()
print("\nMean values by species:")
print(grouped)

# Task 3: Data Visualization
# Line Chart: simulate trends over sample index for sepal length
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.title("Line Chart: Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.savefig("line_chart.png")
plt.close()

# Bar Chart: Average petal length per species
plt.figure(figsize=(8,5))
sns.barplot(x="target", y="petal length (cm)", data=df, ci=None)
plt.title("Bar Chart: Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.savefig("bar_chart.png")
plt.close()

# Histogram: Distribution of sepal width
plt.figure(figsize=(8,5))
plt.hist(df["sepal width (cm)"], bins=20, color="skyblue", edgecolor="black")
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.savefig("histogram.png")
plt.close()

# Scatter Plot: Sepal length vs Petal length
plt.figure(figsize=(8,5))
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"], alpha=0.7, c=df["target"], cmap="viridis")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.colorbar(label="Species")
plt.savefig("scatter_plot.png")
plt.close()

print("\nVisualizations saved as line_chart.png, bar_chart.png, histogram.png, scatter_plot.png")
