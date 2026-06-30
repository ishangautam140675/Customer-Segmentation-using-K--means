from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

app = Flask(__name__)

@app.route('/')
def home():
    data = pd.read_csv("customer_data.csv")
    X = data[['AnnualIncome', 'SpendingScore']]

    kmeans = KMeans(n_clusters=5)
    y_kmeans = kmeans.fit_predict(X)

    plt.scatter(X['AnnualIncome'], X['SpendingScore'], c=y_kmeans)
    plt.xlabel("Income")
    plt.ylabel("Score")
    plt.title("Customer Segments")
    plt.savefig("static/plot.png")
    plt.close()

    return render_template('customer.html')

if __name__ == "__main__":
    app.run(debug=True)