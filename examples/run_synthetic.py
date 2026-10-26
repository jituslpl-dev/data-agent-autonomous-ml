"""
End-to-End Demonstration Script for DataAgent.
Simulates a real-world messy dataset (credit risk / customer churn),
triggers the autonomous ML pipeline, trains all 4 model families with Stratified CV,
computes SHAP explainability, and outputs full Diagnostic Reports.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from data_agent.pipeline import DataAgent

def generate_synthetic_credit_dataset(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    
    # Customer IDs
    customer_ids = [f"CUST_{10000 + i}" for i in range(n_samples)]
    
    # Dates
    dates = pd.date_range("2024-01-01", periods=100).astype(str).tolist()
    signup_dates = rng.choice(dates, size=n_samples)
    
    # Skewed continuous features (income, debt, loan_amount)
    annual_income = rng.exponential(scale=65000, size=n_samples) + 15000
    credit_score = rng.normal(loc=680, scale=80, size=n_samples).clip(300, 850)
    loan_amount = rng.lognormal(mean=9.5, sigma=0.8, size=n_samples)
    debt_to_income = (loan_amount / annual_income).clip(0, 5.0)
    
    # Categoricals
    home_ownership = rng.choice(["RENT", "OWN", "MORTGAGE"], size=n_samples, p=[0.45, 0.15, 0.40])
    loan_intent = rng.choice(["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"], size=n_samples)
    
    # Constant column (should be pruned)
    constant_col = ["US_REGION_EAST"] * n_samples
    
    # True latent log-odds of default
    z = (
        -0.00003 * annual_income
        - 0.015 * (credit_score - 600)
        + 0.00002 * loan_amount
        + 1.8 * debt_to_income
        + (home_ownership == "RENT").astype(float) * 0.4
        - 1.2
    )
    prob_default = 1.0 / (1.0 + np.exp(-z))
    default_label = (rng.uniform(size=n_samples) < prob_default).astype(int)
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date": signup_dates,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "debt_to_income": debt_to_income,
        "home_ownership": home_ownership,
        "loan_intent": loan_intent,
        "region_const": constant_col,
        "loan_default": default_label
    })
    
    # Inject missing values
    df.loc[rng.choice(n_samples, size=int(0.08 * n_samples), replace=False), "annual_income"] = np.nan
    df.loc[rng.choice(n_samples, size=int(0.05 * n_samples), replace=False), "credit_score"] = np.nan
    df.loc[rng.choice(n_samples, size=int(0.03 * n_samples), replace=False), "home_ownership"] = np.nan
    
    # Inject extreme outliers
    df.loc[rng.choice(n_samples, size=5, replace=False), "loan_amount"] *= 15
    
    return df

def main():
    print("=" * 80)
    print("   DataAgent: Autonomous Data Science Pipeline Execution")
    print("=" * 80)
    
    # 1. Generate messy dataset
    print("\n[Step 1/6] Synthesizing realistic messy financial dataset...")
    df = generate_synthetic_credit_dataset(n_samples=1200, random_state=42)
    print(f"-> Generated {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"-> Target distribution:\n{df['loan_default'].value_counts(normalize=True).to_dict()}")
    
    # Split train and test
    train_df = df.iloc[:1000].copy()
    test_df = df.iloc[1000:].copy()
    y_test_true = test_df["loan_default"].values
    
    # 2. Instantiate DataAgent
    agent = DataAgent(
        cv_splits=5,
        search_iterations=3,
        handle_outliers=True,
        create_interactions=True,
        random_state=42
    )
    
    # 3. Autonomous Pipeline Fit
    print("\n[Step 2/6] Running Autonomous Pipeline Fit...")
    agent.fit(train_df, target_col="loan_default")
    
    # 4. Model Tournament Results
    print("\n[Step 3/6] Model Tournament Leaderboard:")
    leaderboard = agent.tournament_report.leaderboard
    print(leaderboard.to_string(index=False))
    print(f"\n-> Best Champion Model: {agent.tournament_report.champion_model_name}")
    
    # 5. Out-of-sample Evaluation & Inference
    print("\n[Step 4/6] Evaluating on Holdout Test Set (200 samples)...")
    preds = agent.predict(test_df)
    probs = agent.predict_proba(test_df)[:, 1]
    
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    test_auc = roc_auc_score(y_test_true, probs)
    test_acc = accuracy_score(y_test_true, preds)
    print(f"-> Holdout Test ROC-AUC: {test_auc:.4f}")
    print(f"-> Holdout Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test_true, preds))
    
    # 6. Granular SHAP Explainability
    print("\n[Step 5/6] Inspecting SHAP Feature Attributions...")
    global_importance = agent.shap_report.global_importance
    print("Top 10 Most Influential Features:")
    for rank, (feat, score) in enumerate(list(global_importance.items())[:10], 1):
        print(f"  {rank:2d}. {feat:<30} SHAP Score: {score:.4f}")
    
    print("\nAutomated Insights:")
    for insight in agent.shap_report.insights:
        print(f"  * {insight}")
    
    # 7. Generate Comprehensive Diagnostic Reports (Markdown + HTML)
    print("\n[Step 6/6] Generating Comprehensive Diagnostic Reports...")
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    md_path = os.path.join(reports_dir, "diagnostic_report.md")
    html_path = os.path.join(reports_dir, "diagnostic_report.html")
    
    md_content = agent.generate_report(html_output_path=html_path)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"-> Diagnostic Markdown Report saved to: {os.path.abspath(md_path)}")
    print(f"-> Interactive HTML Report saved to: {os.path.abspath(html_path)}")
    print("\nAutonomous Data Science Agent pipeline completed successfully!")

if __name__ == "__main__":
    main()
