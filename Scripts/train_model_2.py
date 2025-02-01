import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("../Data/sampregdata.csv")

X = data[['x2','x4','x1']]  
y = data['y']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the new model
model_v2 = LinearRegression()
model_v2.fit(X_train, y_train)

# Save the new model
with open("../Models/model_v2.pkl", "wb") as f:
    pickle.dump(model_v2, f)
