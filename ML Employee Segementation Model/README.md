# Employee Segmentation (KMeans)

> **Disclaimer: this is a learning project, not a production-ready project.**
> It was built for education purposes only, to practise clustering, scikit-learn pipelines and model registration. It must not be used for any production or real HR decision-making. Using it that way can cause problems; see [Limitations](#limitations).

Groups employees into HR segments with KMeans, using workforce characteristics only. **Attrition is never used to build the segments**; it is used afterwards to check whether the segments mean anything.

Dataset: [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (1,470 employees, 35 columns). It is a fictional dataset created by IBM data scientists, so the findings illustrate the method and do not describe a real workforce.

## Pipeline

1. **Load & explore** the data
2. **Clean & encode**: drop constant/identifier columns, encode binary columns as 0/1
3. **Feature selection**: drop redundant seniority features (income, job level, working years and age are highly correlated)
4. **Build pipeline**: `StandardScaler` + `OneHotEncoder` in a `ColumnTransformer`, followed by `KMeans`
5. **Choose K** using inertia, silhouette, Davies-Bouldin and seed stability (ARI) for K = 2..10
6. **Train** the pipeline with K = 3
7. **Visualise** the clusters with PCA
8. **Profile** the segments and test the attrition gap (chi-square)
9. **Score new employees** and verify the pipeline survives save/reload
10. **Register** the pipeline in the Azure ML Model Registry (optional)

## Results

| Segment | Employees | Attrition | Avg income | Avg tenure (yrs) |
|---|---|---|---|---|
| Early-Career Core | 869 | 17.0% | 4,504 | 5.6 |
| Experienced Job-Hoppers | 393 | 19.0% | 7,414 | 4.6 |
| Tenured Seniors | 208 | 9.0% | 13,134 | 17.6 |

Overall attrition is 16.1%. The attrition gap across segments is statistically significant (chi-square = 9.5, 2 dof), driven mostly by the Tenured Seniors.

## Limitations

**This project is for learning only and is not production ready. Do not use it in any production or real-world HR setting.**

Model limitations:

- **Weak cluster structure.** The silhouette score at K = 3 is 0.086 and PCA keeps only 24% of the variance. The segments are a useful grouping along a seniority gradient, not sharply separated groups.
- **K = 3 is a judgement call.** K = 2 scores better on silhouette and Davies-Bouldin; K = 3 was chosen because it is stable and adds an actionable segment.
- **Protected attributes are used.** `Gender` and `MaritalStatus` are clustering features. A real HR use case would need a fairness and legal review, or these should be removed.
- **Synthetic data.** The IBM dataset is fictional, so the findings do not describe a real workforce.
- **Association, not prediction.** The attrition result comes from a single snapshot and does not predict who will leave.
- **Feature preparation is basic.** Skewed variables (income, tenure) are not transformed, ordinal survey scores are treated as continuous, and one-hot and numeric features are mixed in KMeans.

Engineering limitations:

- **No held-out evaluation.** The model is fit and evaluated on the same data; stability is only checked across random seeds.
- **Segment names are hard-coded to cluster IDs.** Retraining can reorder the IDs and silently mislabel the segments.
- **No input validation.** Missing values, out-of-range scores and unseen categories are not handled or reported.
- **Pickle serialisation.** The saved pipeline depends on the scikit-learn version and should never be loaded from an untrusted source.
- **No tests, monitoring or drift detection**, no scoring endpoint, and no retraining plan.

## Project structure

```
assets/               not in the repo; you create it and add the dataset (see Getting started)
  WA_Fn-UseC_-HR-Employee-Attrition.csv
employee_segmentation/
  config.py           paths, feature lists, K, segment names, Azure settings
  data.py             load, clean, correlation check
  model.py            pipeline, choose K, train, PCA, profiling, scoring
  azure_registry.py   save / reload / register the pipeline
run_pipeline.py       runs every step end to end
requirements.txt
.env.example          template for Azure settings
```

## Getting started

### 1. Get the dataset

The dataset is not included in this repository. Create an `assets` folder in the project root and download the data into it:

1. Create the folder: `mkdir assets`
2. Download the IBM HR Analytics dataset from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (a free Kaggle account is required).
3. Place `WA_Fn-UseC_-HR-Employee-Attrition.csv` inside `assets/`, so the path is `assets/WA_Fn-UseC_-HR-Employee-Attrition.csv`.

### 2. Install and run

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python run_pipeline.py
```

Outputs (the saved pipeline and two plots) are written to `outputs/`, which is git-ignored.

### Register in Azure ML (optional)

```bash
copy .env.example .env        # macOS/Linux: cp .env.example .env
# fill in AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME
az login
python run_pipeline.py --register
```

Never commit `.env`.

## License

MIT, see [LICENSE](LICENSE). The dataset is third-party; check its source for terms.
