"""Central configuration: paths, feature lists, model settings and Azure ML environment."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

try:  # optional: load AZURE_* values from a local .env file
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")
except ImportError:
    pass

DATA_PATH = PROJECT_DIR / "assets" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "employee_segmentation_pipeline.pkl"

RANDOM_STATE = 42
K = 3
K_RANGE = range(2, 11)
STABILITY_SEEDS = (1, 2, 3, 4, 5)

# Section 2: constant / identifier columns and binary encodings
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
BINARY_ENCODE_COLS = ["Attrition", "OverTime", "Gender"]

# Section 3: feature selection
CANDIDATE_NUMERIC = [
    "MonthlyIncome", "JobSatisfaction", "EnvironmentSatisfaction",
    "WorkLifeBalance", "RelationshipSatisfaction", "YearsAtCompany",
    "YearsSinceLastPromotion", "DistanceFromHome", "TotalWorkingYears",
    "JobLevel", "NumCompaniesWorked", "Age", "JobInvolvement",
    "PercentSalaryHike",
]
CORR_THRESHOLD = 0.6

NUMERICAL_COLS = [
    "MonthlyIncome", "JobSatisfaction", "EnvironmentSatisfaction",
    "WorkLifeBalance", "RelationshipSatisfaction", "YearsAtCompany",
    "YearsSinceLastPromotion", "DistanceFromHome", "NumCompaniesWorked",
    "JobInvolvement", "PercentSalaryHike",
]
CATEGORICAL_COLS = ["BusinessTravel", "Department", "MaritalStatus"]
BINARY_COLS = ["OverTime", "Gender"]
FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS + BINARY_COLS
assert "Attrition" not in FEATURE_COLS

# Section 8: cluster profiling (held-out columns are NOT used to build clusters)
SEGMENT_NAMES = {
    0: "Experienced Job-Hoppers",
    1: "Early-Career Core",
    2: "Tenured Seniors",
}

# Section 10: Azure ML. Set these as environment variables; never hard-code them.
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "")
AZURE_WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME", "")
AZURE_MODEL_NAME = "employee-segmentation-pipeline"
AZURE_MODEL_DESCRIPTION = (
    "Full SKLearn Pipeline: Preprocessing + KMeans (K=3) for HR employee segmentation. "
    "Excludes Attrition from features; segments: Experienced Job-Hoppers, "
    "Early-Career Core, Tenured Seniors."
)
