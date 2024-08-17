import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the dataset
filename = 'dataset.csv'  # Path to your CSV file
data = pd.read_csv(filename)



# Split the data into features (X) and target (y)
X = data.drop(columns=['lelah'])
y = data['lelah']

# Create the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X, y)