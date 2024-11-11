from flask import Flask, request, render_template, send_file
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    # Retrieve form data
    n_clusters = int(request.form.get("n_clusters", 3))  # Default to 3 clusters
    data_points = request.form.get("data_points", "1,1;2,2;3,3;8,8;9,9;10,10")

    # Parse data points
    data = np.array([list(map(float, point.split(','))) for point in data_points.split(';')])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette="viridis", s=100)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(f'K-Means Clustering with {n_clusters} Clusters')
    plt.legend()

    # Save the plot to a BytesIO object and serve it as an image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
