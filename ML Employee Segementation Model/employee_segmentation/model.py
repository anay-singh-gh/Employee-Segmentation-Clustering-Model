"""Sections 4-9: pipeline, choosing K, training, PCA, cluster profiling, scoring new data."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else x


# ---- Section 4: pipeline ---------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), config.NUMERICAL_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_COLS),
        ("bin", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), config.BINARY_COLS),
    ])


def build_pipeline(k: int = config.K) -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("kmeans", KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)),
    ])


# ---- Section 5: choose K ---------------------------------------------------
def evaluate_k(X: pd.DataFrame) -> pd.DataFrame:
    """Inertia, silhouette, Davies-Bouldin and seed stability (ARI) for each K."""
    X_t = _dense(build_preprocessor().fit_transform(X))
    rows = []
    for k in config.K_RANGE:
        base = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10).fit(X_t)
        other = [KMeans(n_clusters=k, random_state=s, n_init=10).fit_predict(X_t)
                 for s in config.STABILITY_SEEDS]
        rows.append({
            "K": k,
            "Inertia": base.inertia_,
            "Silhouette": silhouette_score(X_t, base.labels_),
            "Davies_Bouldin": davies_bouldin_score(X_t, base.labels_),
            "Stability_ARI": np.mean([adjusted_rand_score(base.labels_, o) for o in other]),
        })
    k_eval = pd.DataFrame(rows).set_index("K")
    print(k_eval.round(3).to_string())
    return k_eval


def plot_k_evaluation(k_eval: pd.DataFrame, save_path=None, show=False):
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    panels = [("Inertia", "Inertia (elbow)"),
              ("Silhouette", "Silhouette (higher = better)"),
              ("Davies_Bouldin", "Davies-Bouldin (lower = better)"),
              ("Stability_ARI", "Seed stability (ARI)")]
    for ax, (col, title) in zip(axes, panels):
        ax.plot(k_eval.index, k_eval[col], marker="o", color="steelblue", linewidth=2)
        ax.axvline(config.K, color="#e74c3c", linestyle="--", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.set_xticks(k_eval.index)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _finish(fig, save_path, show)


# ---- Section 6: train ------------------------------------------------------
def train_pipeline(df: pd.DataFrame, k: int = config.K):
    """Fit the full pipeline, add Cluster labels to df and report the silhouette score."""
    X = df[config.FEATURE_COLS]
    pipeline = build_pipeline(k)
    pipeline.fit(X)
    df["Cluster"] = pipeline.named_steps["kmeans"].labels_

    print(f"Pipeline trained with K={k}")
    print("\nCluster Distribution:")
    print(df["Cluster"].value_counts().sort_index())
    score = silhouette_score(pipeline.named_steps["preprocessor"].transform(X), df["Cluster"])
    print(f"\nSilhouette score: {score:.3f}")
    return pipeline, df


# ---- Section 7: PCA visualisation -----------------------------------------
def plot_pca(pipeline: Pipeline, df: pd.DataFrame, save_path=None, show=False):
    X_t = _dense(pipeline.named_steps["preprocessor"].transform(df[config.FEATURE_COLS]))
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X_t)
    evr = pca.explained_variance_ratio_
    print(f"Variance explained: PC1 = {evr[0]:.1%}, PC2 = {evr[1]:.1%}, total = {evr.sum():.1%}")

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    labels = df["Cluster"].values
    fig = plt.figure(figsize=(9, 5))
    for cluster in range(pipeline.named_steps["kmeans"].n_clusters):
        mask = labels == cluster
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {cluster}",
                    color=colors[cluster], alpha=0.6, s=40)
    plt.title(f"Employee Segments: PCA Projection ({evr.sum():.0%} of variance shown)")
    plt.xlabel(f"PCA Component 1 ({evr[0]:.0%})")
    plt.ylabel(f"PCA Component 2 ({evr[1]:.0%})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _finish(fig, save_path, show)
    return evr


# ---- Section 8: cluster analysis ------------------------------------------
def profile_clusters(df: pd.DataFrame):
    """Profile segments on columns used to build them and on held-out columns."""
    used = df.groupby("Cluster").agg(
        Employees=("Cluster", "size"),
        Avg_Income=("MonthlyIncome", "mean"),
        Avg_Tenure=("YearsAtCompany", "mean"),
        Companies_Worked=("NumCompaniesWorked", "mean"),
        Yrs_Since_Promo=("YearsSinceLastPromotion", "mean"),
        Job_Satisfaction=("JobSatisfaction", "mean"),
        Env_Satisfaction=("EnvironmentSatisfaction", "mean"),
        Work_Life_Balance=("WorkLifeBalance", "mean"),
        Distance_Home=("DistanceFromHome", "mean"),
        OverTime_Rate=("OverTime", "mean"),
    ).round(2)

    held_out = df.groupby("Cluster").agg(
        Attrition_Rate=("Attrition", "mean"),
        Avg_Age=("Age", "mean"),
        Avg_JobLevel=("JobLevel", "mean"),
        Total_Working_Yrs=("TotalWorkingYears", "mean"),
        Performance_Rating=("PerformanceRating", "mean"),
        Training_Last_Year=("TrainingTimesLastYear", "mean"),
    ).round(2)

    print("=" * 70, "\n  USED TO BUILD THE CLUSTERS\n", "=" * 70, sep="")
    print(used.T.to_string())
    print("\n" + "=" * 70 + "\n  NOT USED TO BUILD THE CLUSTERS (independent validation)\n" + "=" * 70)
    print(held_out.T.to_string())
    print(f"\nOverall attrition rate: {df['Attrition'].mean():.1%}")
    return used, held_out


def attrition_chi_square(df: pd.DataFrame):
    """Is the attrition gap between segments bigger than chance? (no extra dependency)"""
    ct = pd.crosstab(df["Cluster"], df["Attrition"]).values
    expected = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / ct.sum()
    chi2 = ((ct - expected) ** 2 / expected).sum()
    dof = (ct.shape[0] - 1) * (ct.shape[1] - 1)
    print(f"Chi-square = {chi2:.1f} with {dof} degrees of freedom "
          f"(critical value at p=0.05 is 5.99 for 2 dof)")
    return chi2, dof


def name_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Attach readable HR names and print the segment summary."""
    df["Segment"] = df["Cluster"].map(config.SEGMENT_NAMES)
    summary = df.groupby("Segment").agg(
        Employees=("Cluster", "size"),
        Attrition_Rate=("Attrition", "mean"),
        Avg_Income=("MonthlyIncome", "mean"),
        Avg_Tenure=("YearsAtCompany", "mean"),
    ).round(2)
    summary["Attrition_Rate"] = (summary["Attrition_Rate"] * 100).round(1).astype(str) + "%"
    print(summary)
    return df


# ---- Section 9: score new employees ---------------------------------------
def predict_segments(pipeline: Pipeline, new_employees: pd.DataFrame) -> pd.DataFrame:
    """Score rows that contain only the feature columns (no Attrition)."""
    out = new_employees.copy()
    out["Cluster"] = pipeline.predict(out[config.FEATURE_COLS])
    out["Segment"] = out["Cluster"].map(config.SEGMENT_NAMES)
    return out


def _finish(fig, save_path, show):
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
