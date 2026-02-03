from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "influencer-engagement-metrics-dataset.csv")
PLOT_DIR = os.path.join(BASE_DIR, "static", "plots")

app = Flask(__name__)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["views_count"] = df["views_count"].fillna(0)
    return df


def compute_clusters(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    features = [
        "likes_count",
        "comments_count",
        "shares_count",
        "views_count",
        "engagement_rate",
        "followers_count",
    ]
    X = df[features].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    k = 3
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = model.fit_predict(Xs)
    return df, features, k


def build_summary(df: pd.DataFrame) -> dict:
    return {
        "total_posts": int(len(df)),
        "total_influencers": int(df["influencer_id"].nunique()),
        "platforms": int(df["platform"].nunique()),
        "avg_engagement": round(float(df["engagement_rate"].mean() * 100), 2),
        "sponsored_rate": round(float(df["sponsored"].mean() * 100), 2),
        "top_platform": df["platform"].value_counts().idxmax(),
        "top_country": df["audience_top_country"].value_counts().idxmax(),
        "last_updated": datetime.utcnow().strftime("%Y-%m-%d"),
    }


def create_plots(df: pd.DataFrame) -> list[str]:
    os.makedirs(PLOT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plots = []

    # Engagement by platform
    plt.figure(figsize=(7.5, 4.5))
    platform_eng = (
        df.groupby("platform")["engagement_rate"].mean().sort_values(ascending=False)
    )
    sns.barplot(x=platform_eng.index, y=platform_eng.values, palette="crest")
    plt.title("Average Engagement Rate by Platform")
    plt.ylabel("Engagement Rate")
    plt.xlabel("Platform")
    plt.tight_layout()
    fname = "engagement_by_platform.png"
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=160)
    plt.close()
    plots.append(fname)

    # Followers vs engagement (colored by cluster)
    plt.figure(figsize=(7.5, 4.5))
    sns.scatterplot(
        data=df,
        x="followers_count",
        y="engagement_rate",
        hue="cluster",
        palette="viridis",
        alpha=0.75,
    )
    plt.title("Followers vs Engagement (Clustered)")
    plt.xlabel("Followers")
    plt.ylabel("Engagement Rate")
    plt.tight_layout()
    fname = "followers_vs_engagement.png"
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=160)
    plt.close()
    plots.append(fname)

    # Post type distribution
    plt.figure(figsize=(7.5, 4.5))
    post_type_counts = df["post_type"].value_counts()
    sns.barplot(x=post_type_counts.index, y=post_type_counts.values, palette="mako")
    plt.title("Post Type Distribution")
    plt.xlabel("Post Type")
    plt.ylabel("Count")
    plt.tight_layout()
    fname = "post_type_distribution.png"
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=160)
    plt.close()
    plots.append(fname)

    return plots


@app.route("/")
def index():
    df = load_data()
    summary = build_summary(df)
    return render_template("index.html", summary=summary)


@app.route("/insights")
def insights():
    df = load_data()
    df, features, k = compute_clusters(df)
    plots = create_plots(df)

    summary = build_summary(df)
    cluster_counts = df["cluster"].value_counts().sort_index().to_dict()

    top_brands = (
        df["brand_mentioned"]
        .dropna()
        .value_counts()
        .head(5)
        .to_dict()
    )

    return render_template(
        "insights.html",
        summary=summary,
        cluster_counts=cluster_counts,
        plots=plots,
        k=k,
        features=features,
        top_brands=top_brands,
    )


if __name__ == "__main__":
    app.run(debug=True)
