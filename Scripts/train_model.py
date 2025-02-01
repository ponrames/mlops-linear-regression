import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("../data/sampregdata.csv")

X = data[['x2']]  
y = data['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open("../Models/model_v1.pkl", "wb") as f:
    pickle.dump(model, f)
