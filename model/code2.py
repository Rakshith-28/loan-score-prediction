import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

NUMERICAL_FEATURES = [
    'Income Stability Score',
    'Debt-to-Income Ratio',
    'Credit Utilization Rate',
    'Monthly Savings Percentage',
    'Financial Volatility Index',
    'Financial Resilience Score',
    'Financial Stress Index'
]
CATEGORICAL_FEATURES = ['Spending Behavior', 'Investment Strategy']

DEFAULT_SPENDING_CLASSES = ['Conservative', 'Balanced', 'Aggressive']
DEFAULT_INVESTING_CLASSES = ['Conservative', 'Moderate', 'Aggressive']


# ------------------ Core Functions ------------------ #
def get_user_input():
    """Safely collect user financial data with validation."""
    print("\n" + "=" * 50)
    print("FINANCIAL HEALTH ASSESSMENT".center(50))
    print("=" * 50)

    while True:
        try:
            data = {
                'Monthly Income': float(input("\nMonthly Income (Rs): ")),
                'Savings & Investments': float(input("Savings & Investments (Rs): ")),
                'Fixed Expenses': float(input("Fixed Expenses (Rs): ")),
                'Variable Expenses': float(input("Variable Expenses (Rs): ")),
                'Existing Loans & Liabilities': float(input("Loans/Liabilities (Rs): ")),
                'Spending Behavior': input("Spending (Conservative/Balanced/Aggressive): ").title(),
                'Investment Strategy': input("Investments (Conservative/Moderate/Aggressive): ").title(),
            }

            if any(
                val < 0
                for val in [
                    data['Monthly Income'],
                    data['Savings & Investments'],
                    data['Fixed Expenses'],
                    data['Variable Expenses'],
                    data['Existing Loans & Liabilities'],
                ]
            ):
                raise ValueError("Negative values not allowed")

            if data['Spending Behavior'] not in DEFAULT_SPENDING_CLASSES:
                raise ValueError("Invalid spending behavior")

            if data['Investment Strategy'] not in DEFAULT_INVESTING_CLASSES:
                raise ValueError("Invalid investment strategy")

            return data

        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")


def calculate_metrics(data):
    """Calculate feature values with same names as training script."""
    income = max(data['Monthly Income'], 0.01)
    savings = data['Savings & Investments']
    fixed_expenses = data['Fixed Expenses']
    variable_expenses = data['Variable Expenses']
    liabilities = data['Existing Loans & Liabilities']

    return {
        'Income Stability Score': min(savings / income, 1.0),
        'Debt-to-Income Ratio': min(liabilities / income, 1.0),
        'Credit Utilization Rate': min(liabilities / income, 1.0),
        'Monthly Savings Percentage': max((income - (fixed_expenses + variable_expenses)) / income, 0),
        'Financial Volatility Index': 0.5,
        'Financial Resilience Score': min(savings / (fixed_expenses + variable_expenses + 1), 12),
        'Financial Stress Index': min(liabilities / (savings + 1), 5),
    }


def normalize_encoder_keys(label_encoders):
    """Support old saved key names used by previous script versions."""
    if 'Spending Behavior' not in label_encoders and 'Spending' in label_encoders:
        label_encoders['Spending Behavior'] = label_encoders['Spending']
    if 'Investment Strategy' not in label_encoders and 'Investing' in label_encoders:
        label_encoders['Investment Strategy'] = label_encoders['Investing']
    return label_encoders


def ensure_required_encoders(label_encoders):
    """Guarantee required encoders are available and valid."""
    if 'Spending Behavior' not in label_encoders:
        label_encoders['Spending Behavior'] = LabelEncoder().fit(DEFAULT_SPENDING_CLASSES)
    if 'Investment Strategy' not in label_encoders:
        label_encoders['Investment Strategy'] = LabelEncoder().fit(DEFAULT_INVESTING_CLASSES)
    return label_encoders


def safe_encode_value(encoder, value):
    """Encode value safely, falling back to known class for unseen labels."""
    value_str = str(value)
    safe_value = value_str if value_str in encoder.classes_ else encoder.classes_[0]
    return int(encoder.transform([safe_value])[0])


def initialize_model():
    """Initialize model from saved artifacts or fallback mini model."""
    base_dir = os.path.dirname(__file__)
    artifact_dir = os.path.join(base_dir, 'model')
    model_path = os.path.join(artifact_dir, 'financial_model.pkl')
    scaler_path = os.path.join(artifact_dir, 'scaler.pkl')
    encoders_path = os.path.join(artifact_dir, 'label_encoders.pkl')

    if os.path.exists(model_path) and os.path.exists(encoders_path):
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        encoders = ensure_required_encoders(normalize_encoder_keys(encoders))
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        return model, scaler, encoders

    print("\nNo training artifacts found. Using built-in assessment model")
    model = RandomForestRegressor(n_estimators=50, random_state=42)

    fallback_encoders = {
        'Spending Behavior': LabelEncoder().fit(DEFAULT_SPENDING_CLASSES),
        'Investment Strategy': LabelEncoder().fit(DEFAULT_INVESTING_CLASSES),
    }

    X = pd.DataFrame(
        {
            'Income Stability Score': [0.3, 0.6, 0.9],
            'Debt-to-Income Ratio': [0.5, 0.3, 0.1],
            'Credit Utilization Rate': [0.5, 0.3, 0.1],
            'Monthly Savings Percentage': [0.1, 0.2, 0.3],
            'Financial Volatility Index': [0.5, 0.5, 0.5],
            'Financial Resilience Score': [1.0, 2.0, 3.0],
            'Financial Stress Index': [1.0, 0.5, 0.2],
            'Spending Behavior': [
                safe_encode_value(fallback_encoders['Spending Behavior'], 'Conservative'),
                safe_encode_value(fallback_encoders['Spending Behavior'], 'Balanced'),
                safe_encode_value(fallback_encoders['Spending Behavior'], 'Balanced'),
            ],
            'Investment Strategy': [
                safe_encode_value(fallback_encoders['Investment Strategy'], 'Moderate'),
                safe_encode_value(fallback_encoders['Investment Strategy'], 'Moderate'),
                safe_encode_value(fallback_encoders['Investment Strategy'], 'Aggressive'),
            ],
        }
    )
    y = np.array([0.4, 0.7, 0.9])
    model.fit(X, y)

    return model, None, fallback_encoders


def generate_report(user_data, metrics, score):
    """Create comprehensive financial report."""
    report = f"""
    FINANCIAL HEALTH REPORT
    {'=' * 50}
    INCOME & EXPENSES:
    - Monthly Income: Rs.{user_data['Monthly Income']:,.2f}
    - Fixed Expenses: Rs.{user_data['Fixed Expenses']:,.2f}
    - Variable Expenses: Rs.{user_data['Variable Expenses']:,.2f}

    ASSETS & LIABILITIES:
    - Savings: Rs.{user_data['Savings & Investments']:,.2f}
    - Debt: Rs.{user_data['Existing Loans & Liabilities']:,.2f}

    KEY METRICS:
    - Income Stability: {metrics['Income Stability Score'] * 100:.1f}%
    - Debt-to-Income: {metrics['Debt-to-Income Ratio'] * 100:.1f}%
    - Savings Rate: {metrics['Monthly Savings Percentage'] * 100:.1f}%

    FINANCIAL SCORE: {score}/100
    {'=' * 50}
    RECOMMENDATIONS:
    """

    if metrics['Monthly Savings Percentage'] < 0.1:
        report += "1. [CRITICAL] Increase savings immediately (currently under 10%)\n"
    elif metrics['Monthly Savings Percentage'] < 0.2:
        report += "1. [WARNING] Boost savings to at least 20%\n"
    else:
        report += "1. [GOOD] Good savings rate - maintain this\n"

    if metrics['Debt-to-Income Ratio'] > 0.4:
        report += "2. [CRITICAL] Reduce debt (ratio over 40%)\n"
    elif metrics['Debt-to-Income Ratio'] > 0.3:
        report += "2. [WARNING] Lower debt ratio below 30%\n"

    if user_data['Spending Behavior'] == 'Aggressive':
        report += "3. [TIP] Track discretionary spending\n"

    report += f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    report += "\n" + "=" * 50

    return report


# ------------------ Main Program ------------------ #
def main():
    base_dir = os.path.dirname(__file__)
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    model, scaler, encoders = initialize_model()

    while True:
        print("\nOptions:")
        print("1. Analyze my finances")
        print("2. Exit")
        choice = input("Select (1/2): ").strip()

        if choice == '1':
            try:
                user_data = get_user_input()
                metrics = calculate_metrics(user_data)

                input_data = pd.DataFrame(
                    [
                        {
                            'Income Stability Score': metrics['Income Stability Score'],
                            'Debt-to-Income Ratio': metrics['Debt-to-Income Ratio'],
                            'Credit Utilization Rate': metrics['Credit Utilization Rate'],
                            'Monthly Savings Percentage': metrics['Monthly Savings Percentage'],
                            'Financial Volatility Index': metrics['Financial Volatility Index'],
                            'Financial Resilience Score': metrics['Financial Resilience Score'],
                            'Financial Stress Index': metrics['Financial Stress Index'],
                            'Spending Behavior': safe_encode_value(
                                encoders['Spending Behavior'], user_data['Spending Behavior']
                            ),
                            'Investment Strategy': safe_encode_value(
                                encoders['Investment Strategy'], user_data['Investment Strategy']
                            ),
                        }
                    ]
                )

                input_data = input_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

                if scaler is not None:
                    input_data[NUMERICAL_FEATURES] = scaler.transform(input_data[NUMERICAL_FEATURES])

                score = int(round(model.predict(input_data)[0] * 100))

                report = generate_report(user_data, metrics, score)
                filename = f"financial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                output_path = os.path.join(reports_dir, filename)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)

                print("\n" + "=" * 50)
                print(report)
                print(f"\nReport saved to {output_path}")
                print("=" * 50)

            except Exception as e:
                print(f"\nError: {str(e)}\nPlease try again")

        elif choice == '2':
            print("\nThank you for using the Financial Health Analyzer!")
            break

        else:
            print("\nInvalid choice. Please enter 1 or 2")


if __name__ == "__main__":
    main()
