"""Run the whole workflow end to end.

    python run_pipeline.py              # train, evaluate, save locally
    python run_pipeline.py --register   # also register in Azure ML (needs AZURE_* env vars)
"""
import argparse

import pandas as pd

from employee_segmentation import azure_registry, config, data, model

NEW_EMPLOYEES = pd.DataFrame([
    {"MonthlyIncome": 3200, "JobSatisfaction": 2, "EnvironmentSatisfaction": 3, "WorkLifeBalance": 3,
     "RelationshipSatisfaction": 3, "YearsAtCompany": 2, "YearsSinceLastPromotion": 1, "DistanceFromHome": 12,
     "NumCompaniesWorked": 6, "JobInvolvement": 3, "PercentSalaryHike": 13,
     "BusinessTravel": "Travel_Rarely", "Department": "Sales", "MaritalStatus": "Single",
     "OverTime": 1, "Gender": 0},
    {"MonthlyIncome": 16000, "JobSatisfaction": 3, "EnvironmentSatisfaction": 3, "WorkLifeBalance": 3,
     "RelationshipSatisfaction": 3, "YearsAtCompany": 18, "YearsSinceLastPromotion": 8, "DistanceFromHome": 5,
     "NumCompaniesWorked": 2, "JobInvolvement": 3, "PercentSalaryHike": 14,
     "BusinessTravel": "Travel_Rarely", "Department": "Research & Development", "MaritalStatus": "Married",
     "OverTime": 0, "Gender": 1},
])


def main(register: bool):
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n== 1. Load & explore ==")
    df = data.load_data()

    print("\n== 2. Clean & encode ==")
    df = data.clean_data(df)

    print("\n== 3. Feature selection ==")
    data.correlated_pairs(df)

    print("\n== 5. Choose K ==")
    k_eval = model.evaluate_k(df[config.FEATURE_COLS])
    model.plot_k_evaluation(k_eval, save_path=config.OUTPUT_DIR / "k_evaluation.png")

    print("\n== 6. Train pipeline ==")
    pipeline, df = model.train_pipeline(df, config.K)

    print("\n== 7. PCA visualisation ==")
    model.plot_pca(pipeline, df, save_path=config.OUTPUT_DIR / "pca_clusters.png")

    print("\n== 8. Cluster analysis ==")
    model.profile_clusters(df)
    model.attrition_chi_square(df)
    model.name_segments(df)

    print("\n== 9. Test on new employees (with save/reload check) ==")
    path = azure_registry.save_pipeline(pipeline)
    reloaded = azure_registry.load_pipeline(path)
    scored = model.predict_segments(reloaded, NEW_EMPLOYEES)
    print(scored[["MonthlyIncome", "YearsAtCompany", "NumCompaniesWorked", "Cluster", "Segment"]])

    if register:
        print("\n== 10. Register in Azure ML ==")
        azure_registry.register_model(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true", help="register the model in Azure ML")
    main(parser.parse_args().register)
