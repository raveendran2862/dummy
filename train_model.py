import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data.csv")

# Prepare features
X = df[["feature1", "feature2"]]
y = df["target"]

# Train model
model = LinearRegression()
model.fit(X, y)

print("Model trained successfully")
\nprint('R^2 score:',model.score(X,y))
