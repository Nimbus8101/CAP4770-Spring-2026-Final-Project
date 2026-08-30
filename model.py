from dotenv import load_dotenv
import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine, text
from feature_engineering import FeatureEngineer
from pipelines import build_lr_pipeline
from sklearn.linear_model import LogisticRegression


# ================== Setup ================== #
load_dotenv()

password = os.getenv("DB_PASSWORD")
if not password:
    raise ValueError("DB_PASSWORD not found")

engine = create_engine(
    f"mysql+mysqlconnector://root:{password}@localhost/employee_attrition_db"
)

TABLE_NAME = "employee_attrition"
DB_NAME = "employee_attrition_db"
TEST_TABLE_NAME = "employee_attrition_test"
TEST_DB_NAME = "test_set_db"

# ================== Load CSV ================== #
df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# ================== Build table ================== #
with engine.connect() as conn:

    # ---------- Check DB ----------
    db_exists = conn.execute(text(f"""
        SELECT SCHEMA_NAME
        FROM INFORMATION_SCHEMA.SCHEMATA
        WHERE SCHEMA_NAME = '{DB_NAME}'
    """)).fetchone()

    if not db_exists:
        raise ValueError("Database does not exist")

    # ---------- Drop table if exists ----------
    print("Dropping table if it exists...")
    conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_NAME}"))


    # ---------- Check Test DB ----------
    db_exists = conn.execute(text(f"""
        SELECT SCHEMA_NAME
        FROM INFORMATION_SCHEMA.SCHEMATA
        WHERE SCHEMA_NAME = '{TEST_DB_NAME}'
    """)).fetchone()

    if not db_exists:
        # raise ValueError("Test database does not exist")
        pass

    # ---------- Drop table if exists ----------
    print("Dropping table if it exists...")
    conn.execute(text(f"DROP TABLE IF EXISTS {TEST_TABLE_NAME}"))

# ================== Recreate table + load data ================== #
print("Recreating table and loading CSV...")

# Training set
df.to_sql(
    TABLE_NAME,
    con=engine,
    if_exists="replace",   # creates fresh table
    index=False
)

# Test set
test_df.to_sql(
    TEST_TABLE_NAME,
    con=engine,
    if_exists="replace",   # creates fresh table
    index=False
)

# ================== Add Primary Key ================== #
with engine.connect() as conn:

    print("Adding primary key...")

    conn.execute(text(f"""
        ALTER TABLE {TABLE_NAME}
        ADD PRIMARY KEY (`Employee ID`)
    """))



# ================== Setup ================== #
load_dotenv()

password = os.getenv("DB_PASSWORD")
if not password:
    raise ValueError("DB_PASSWORD not found")


# ========== Load the DataFrame into MySQL ========== #
engine = create_engine(
    f"mysql+mysqlconnector://root:{password}@localhost/employee_attrition_db"
)

df.to_sql("table_name", con=engine, if_exists="append", index=False)


# ========== Query the data from MySQL ========== #

df = pd.read_sql("SELECT * FROM table_name", con=engine)
print(df.head())


# Define which columns are categorical and which are numeric (for later preprocessing)
categorical_cols = [
    "Gender",
    "Job Role",
    "Work-Life Balance",
    "Job Satisfaction",
    "Performance Rating",
    # "Overtime",               Removed during feature engineering
    "Education Level",
    "Marital Status",
    "Company Size",
    "Remote Work",
    "Leadership Opportunities",
    "Innovation Opportunities",
    "Company Reputation",
    "Employee Recognition"
]

# The features that were encoded were Gender, Job Role, Work-Life Balance, Job Satisfaction, Performance Rating, Education Level, Marital Status, Company Size, Remote Work, Leadership Opportunities, Innovation Opportunities, Company Reputation, Employee Recognition

numeric_cols = [
    "Monthly Income",
    "Distance from Home",
    "Career_Progression_Rate",
    "Stagnation",
    "Promotion_Rate"
]


# Print basic info about the data before preprocessing
X_train = df.drop(columns=["Attrition"])
y_train = df["Attrition"].map({"Left": 1, "Stayed": 0})  # Convert to binary labels

X_test = test_df.drop(columns=["Attrition"])
y_test = test_df["Attrition"].map({"Left": 1, "Stayed": 0})  # Convert to binary labels

print("Features before preprocessing:")
print(f"Training features : {X_train.shape}")
print(f"Test features     : {X_test.shape}")
print(f"Attrition rate (train): {y_train.mean():.2%}")
print(f"Attrition rate (test) : {y_test.mean():.2%}")

# Modify the data
fe = FeatureEngineer()
X_train_fe = fe.transform(X_train)
X_test_fe  = fe.transform(X_test)

print("\nAfter feature engineering:")
print(f"Training features : {X_train_fe.shape}")
print(f"Test features     : {X_test_fe.shape}")


# Logistic Regression
# Estimates the probability an employee leaves using a weighted sum of features.
# C=1 is the regularization strength — controls how much the model penalizes large weights.

lr = LogisticRegression(C=1, solver='lbfgs', max_iter=1000, random_state=42)
lr_pipeline = build_lr_pipeline(lr, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
# lr_pipeline.fit(X_train, y_train)

# Cross-validation — trains and tests on 5 different splits to get a reliable score
cv_auc_lr = cross_val_score(lr_pipeline, X_train, y_train, cv=5, scoring='roc_auc')
print(f"CV ROC-AUC: {cv_auc_lr.mean():.4f} ± {cv_auc_lr.std():.4f}")

# Evaluate on the held-out test set
lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)
lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
print(f"Test Accuracy : {accuracy_score(y_test, lr_preds):.4f}")
print(f"Test ROC-AUC  : {roc_auc_score(y_test, lr_probs):.4f}")
print(classification_report(y_test, lr_preds, target_names=['Stayed', 'Left']))

model = lr_pipeline.named_steps["model"]
feature_names = lr_pipeline.named_steps["preprocess"].get_feature_names_out()
selected_features = feature_names[model.coef_[0] != 0]
print("\nSelected features by Logistic Regression:")
print(selected_features)

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, lr_preds, display_labels=['Stayed', 'Left'], cmap='Blues'
)
plt.title('Logistic Regression — Confusion Matrix')
plt.tight_layout()
plt.show()