import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from datetime import datetime

# ========================== PART 1: Load and Preprocess Data ========================== #

# Load dataset
data_path = r"financial_wellbeing_data_full_proper.csv"
df = pd.read_csv(data_path)
print("Dataset loaded successfully. Shape:", df.shape)

def preprocess_data(df):
    """Preprocess the financial data and generate required features."""
    df_processed = df.copy()

    # Debt-to-Income Ratio
    if 'Debt-to-Income Ratio' not in df_processed.columns:
        df_processed['Debt-to-Income Ratio'] = np.minimum(
            df_processed['Existing Loans & Liabilities'] / df_processed['Monthly Income'], 1.0)

    # Income Stability Score
    if 'Income Stability Score' not in df_processed.columns:
        df_processed['Income Stability Score'] = np.minimum(
            df_processed['Savings & Investments'] / df_processed['Monthly Income'], 1.0)

    # Credit Utilization Rate
    if 'Credit Utilization Rate' not in df_processed.columns:
        df_processed['Credit Utilization Rate'] = np.minimum(
            df_processed['Existing Loans & Liabilities'] / df_processed['Monthly Income'], 1.0)

    # Monthly Savings Percentage
    if 'Monthly Savings Percentage' not in df_processed.columns:
        df_processed['Monthly Savings Percentage'] = np.maximum(
            (df_processed['Monthly Income'] - (df_processed['Fixed Expenses'] + df_processed['Variable Expenses']))
            / df_processed['Monthly Income'], 0)

    # Financial Volatility Index (default 0.5)
    if 'Financial Volatility Index' not in df_processed.columns:
        df_processed['Financial Volatility Index'] = 0.5

    # Financial Resilience Score
    if 'Financial Resilience Score' not in df_processed.columns:
        df_processed['Financial Resilience Score'] = np.minimum(
            df_processed['Savings & Investments'] / (df_processed['Fixed Expenses'] + df_processed['Variable Expenses'] + 1), 12)

    # Financial Stress Index
    if 'Financial Stress Index' not in df_processed.columns:
        df_processed['Financial Stress Index'] = np.minimum(
            df_processed['Existing Loans & Liabilities'] / (df_processed['Savings & Investments'] + 1), 5)

    # Target: Financial Well-being Score
    if 'Financial Well-being Score' not in df_processed.columns:
        df_processed['Financial Well-being Score'] = (
            df_processed['Income Stability Score'].fillna(0.5) * 0.35 +
            (1 - df_processed['Credit Utilization Rate'].fillna(0.5)) * 0.25 +
            df_processed['Monthly Savings Percentage'].fillna(0.5) * 0.25 +
            (1 - df_processed['Financial Volatility Index'].fillna(0.5)) * 0.15
        ).round(2)

    return df_processed

df_processed = preprocess_data(df)
print("Preprocessing complete. New shape:", df_processed.shape)

# ========================== PART 2: Encode, Split, and Train ========================== #

# Define features and target
categorical_columns = ['Spending Behavior', 'Investment Strategy']
numerical_features = [
    'Income Stability Score', 'Debt-to-Income Ratio', 'Credit Utilization Rate',
    'Monthly Savings Percentage', 'Financial Volatility Index',
    'Financial Resilience Score', 'Financial Stress Index'
]
target = 'Financial Well-being Score'

# Encode categorical features
label_encoders = {}
for col in categorical_columns:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col] = df_processed[col].astype(str)
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

# Split data (80% train, 20% test)
X = df_processed[numerical_features + categorical_columns]
y = df_processed[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Training complete!")

# ========================== PART 3: Model Evaluation ========================== #

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
accuracy = np.mean(np.abs(y_pred - y_test) <= 0.05) * 100  # Within 5% tolerance

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS".center(50))
print("="*50)
print(f"  Mean Absolute Error (MAE): {mae:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"  R-squared (R2) Score: {r2:.4f}")
print(f"  Accuracy (within 5% tolerance): {accuracy:.1f}%")
print("="*50)

# ========================== PART 4: Predict for a Sample Customer ========================== #

# Sample customer input
customer_input = {
    "Monthly Income": 50000,
    "Savings & Investments": 32500,
    "Fixed Expenses": 7500,
    "Variable Expenses": 5000,
    "Existing Loans & Liabilities": 7500,
    "Monthly Cash Flow Trends": 2000,
    "Spending Behavior": "Balanced",
    "Investment Strategy": "Moderate"
}

def process_customer_input(customer_input):
    """Process raw customer input into model features."""
    income = customer_input["Monthly Income"]
    savings = customer_input["Savings & Investments"]
    fixed_exp = customer_input["Fixed Expenses"]
    var_exp = customer_input["Variable Expenses"]
    loans = customer_input["Existing Loans & Liabilities"]
    cash_flow = customer_input["Monthly Cash Flow Trends"]

    return {
        "Income Stability Score": min(savings / income, 1.0),
        "Debt-to-Income Ratio": min(loans / income, 1.0),
        "Credit Utilization Rate": min(loans / income, 1.0),
        "Monthly Savings Percentage": max((income - (fixed_exp + var_exp)) / income, 0),
        "Financial Volatility Index": min(abs(cash_flow) / income, 1.0),
        "Financial Resilience Score": min(savings / (fixed_exp + var_exp + 1), 12),
        "Financial Stress Index": min(loans / (savings + 1), 5),
        "Spending Behavior": customer_input["Spending Behavior"],
        "Investment Strategy": customer_input["Investment Strategy"]
    }

def safe_label_transform(series, le):
    """Safely transform categorical values, handling unseen labels."""
    series = series.astype(str)
    safe_series = series.apply(lambda x: x if x in le.classes_ else le.classes_[0])
    return le.transform(safe_series)

# Process input and predict
customer_features = process_customer_input(customer_input)
input_df = pd.DataFrame({k: [v] for k, v in customer_features.items()})

for col in categorical_columns:
    if col in input_df.columns:
        input_df[col] = safe_label_transform(input_df[col], label_encoders[col])

# Scale numerical features for prediction
input_df[numerical_features] = scaler.transform(input_df[numerical_features])

predicted_score = model.predict(input_df)[0]
score_percentage = int(round(predicted_score * 100))

# Determine score category
if score_percentage >= 85:
    category = "Excellent"
elif score_percentage >= 70:
    category = "Very Good"
elif score_percentage >= 60:
    category = "Good"
elif score_percentage >= 50:
    category = "Fair"
elif score_percentage >= 40:
    category = "Needs Attention"
else:
    category = "Critical"

# ========================== PART 5: Display Results ========================== #

print("\n" + "="*50)
print("LOAN SCORE PREDICTION RESULT".center(50))
print("="*50)

print("\nCustomer Profile:")
print(f"  Monthly Income:       Rs.{customer_input['Monthly Income']:,.2f}")
print(f"  Savings & Investments:Rs.{customer_input['Savings & Investments']:,.2f}")
print(f"  Fixed Expenses:       Rs.{customer_input['Fixed Expenses']:,.2f}")
print(f"  Variable Expenses:    Rs.{customer_input['Variable Expenses']:,.2f}")
print(f"  Loans & Liabilities:  Rs.{customer_input['Existing Loans & Liabilities']:,.2f}")
print(f"  Spending Behavior:    {customer_input['Spending Behavior']}")
print(f"  Investment Strategy:  {customer_input['Investment Strategy']}")

print(f"\nPredicted Financial Well-being Score: {score_percentage}/100")
print(f"Category: {category}")

print("\nKey Metrics:")
print(f"  Income Stability:     {customer_features['Income Stability Score']*100:.1f}%")
print(f"  Debt-to-Income Ratio: {customer_features['Debt-to-Income Ratio']*100:.1f}%")
print(f"  Monthly Savings Rate: {customer_features['Monthly Savings Percentage']*100:.1f}%")
print(f"  Financial Stress:     {customer_features['Financial Stress Index']:.2f} / 5.00")

print("\n" + "="*50)
print(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}".center(50))
print("="*50)