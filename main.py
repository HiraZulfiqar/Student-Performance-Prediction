import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("student_performance.csv")

# 2. Display the first 5 rows
print("First 5 rows:")
print(df.head())

# 3. Check number of rows and columns
print("\nShape:")
print(df.shape)

# 4. Check column names
print("\nColumns:")
print(df.columns)

# 5. Check data types and missing values
print("\nDataset Information:")
print(df.info())

# 6. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 7. Identify features and target
X = df.drop("Final_Score", axis=1)
y = df["Final_Score"]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)



# ==========================================
# 1. Load the dataset
# ==========================================

df = pd.read_csv("student_performance.csv")


# ==========================================
# 2. Separate Features (X) and Target (y)
# ==========================================

X = df.drop("Final_Score", axis=1)
y = df["Final_Score"]


# ==========================================
# 3. Split into Training and Testing Data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 4. Scale the Features
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. Check the results
# ==========================================

print("Original X shape:", X.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("\nY_train shape:", y_train.shape)
print("Y_test shape:", y_test.shape)

print("\nFirst 5 scaled training rows:")
print(X_train_scaled[:5])




# ==========================================
# 1. Create the Linear Regression model
# ==========================================

model = LinearRegression()


# ==========================================
# 2. Train the model
# ==========================================

model.fit(X_train_scaled, y_train)


# ==========================================
# 3. Make predictions
# ==========================================

y_pred = model.predict(X_test_scaled)


# ==========================================
# 4. Evaluate the model
# ==========================================

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


# ==========================================
# 5. Display results
# ==========================================

print("Linear Regression Results")
print("-------------------------")

print("RMSE:", rmse)
print("R² Score:", r2)




# Ridge
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# Lasso
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_scaled, y_train)

# ElasticNet
elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_model.fit(X_train_scaled, y_train)

print("All regularized models trained successfully!")


# Models
models = {
    "Linear Regression": model,
    "Ridge": ridge_model,
    "Lasso": lasso_model,
    "ElasticNet": elastic_model
}

# Store results
results = []

for name, current_model in models.items():

    # Predictions
    y_pred = current_model.predict(X_test_scaled)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append([name, rmse, r2])

# Print comparison
print("MODEL COMPARISON")
print("----------------")

for result in results:
    print(f"{result[0]:20} RMSE: {result[1]:.4f}   R²: {result[2]:.4f}")


print("\nCOEFFICIENT COMPARISON")
print("----------------------")

print("Linear Regression:", model.coef_)
print("Ridge:", ridge_model.coef_)
print("Lasso:", lasso_model.coef_)
print("ElasticNet:", elastic_model.coef_)
print("ElasticNet:", elastic_model.coef_)




# Make predictions using our best model
y_pred = elastic_model.predict(X_test_scaled)

# Plot Actual vs Predicted
plt.scatter(y_test, y_pred)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("ElasticNet: Actual vs Predicted Final Score")

plt.show()