import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Sample Credit Scoring Dataset
data = {
    'Income': [50000, 60000, 25000, 80000, 45000, 70000, 30000, 90000, 55000, 40000],
    'Debt': [10000, 5000, 15000, 2000, 12000, 4000, 18000, 1000, 7000, 14000],
    'Payment_History': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    'Credit_Status': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df[['Income', 'Debt', 'Payment_History']]
y = df['Credit_Status']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

try:
    print("ROC-AUC  :", roc_auc_score(y_test, y_pred))
except:
    print("ROC-AUC could not be calculated.")

# Predict New Customer
print("\n--- New Customer Prediction ---")

income = float(input("Enter Income: "))
debt = float(input("Enter Debt: "))
payment_history = int(input("Payment History (1=Good, 0=Bad): "))

new_customer = scaler.transform([[income, debt, payment_history]])

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("Creditworthy Customer (Loan Approved)")
else:
    print("Not Creditworthy (Loan Rejected)")