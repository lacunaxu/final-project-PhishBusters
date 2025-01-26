import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, auc, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

from clean_data import clean_data, prepare_and_process_nlp_data

# Ensure the output directory exists
output_dir = os.path.abspath("../results/visualizations")
os.makedirs(output_dir, exist_ok=True)

# Load data


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load data from the specified CSV file.

    Parameters:
        csv_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(csv_path)


# Fill missing values


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame with 0.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled.
    """
    return df.fillna(0)


# Drop non-numeric columns


def drop_non_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop non-numeric columns from the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with non-numeric columns removed.
    """
    non_numeric_cols = df.select_dtypes(exclude=["float64", "int64"]).columns
    return df.drop(columns=non_numeric_cols)


# Convert necessary columns to numeric


def convert_to_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Convert specified columns to numeric, forcing errors to NaN.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to convert to numeric.

    Returns:
        pd.DataFrame: The DataFrame with specified columns converted.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# Visualization 1: URL Length Distribution


def url_length_distribution(df: pd.DataFrame):
    """
    Plot and save the distribution of URL lengths.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df["url_length"], bins=30, kde=True, color="blue")
    plt.title("URL Length Distribution")
    plt.xlabel("URL Length")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_dir}/url_length_distribution.png")
    plt.close()


# Visualization 2: Domain Length by Label


def domain_length_by_label(df: pd.DataFrame):
    """
    Plot and save a boxplot of domain length by label.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="label", y="domain_length", data=df)
    plt.title("Domain Length by Label")
    plt.xlabel("Label")
    plt.ylabel("Domain Length")
    plt.savefig(f"{output_dir}/domain_length_by_label.png")
    plt.close()


# Visualization 3: Subdomain Count by Label


def subdomain_count_by_label(df: pd.DataFrame):
    """
    Plot and save a boxplot of subdomain count by label.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="label", y="subdomain_count", data=df)
    plt.title("Subdomain Count by Label")
    plt.xlabel("Label")
    plt.ylabel("Subdomain Count")
    plt.savefig(f"{output_dir}/subdomain_count_by_label.png")
    plt.close()


# Visualization 4: Path Length KDE by Label


def path_length_kde_by_label(df: pd.DataFrame):
    """
    Plot and save a KDE plot of path length by label.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df, x="path_length", hue="label", element="step", kde=True, palette="coolwarm", bins=30)
    plt.title("Path Length Distribution by Label")
    plt.xlabel("Path Length")
    plt.ylabel("Frequency")
    plt.legend(title="Label", labels=["Normal", "Phishing"])
    plt.savefig(f"{output_dir}/path_length_kde_by_label.png")
    plt.close()


# Visualization 5: Feature Correlation Heatmap


def feature_correlation_heatmap(df: pd.DataFrame):
    """
    Plot and save a heatmap of feature correlations.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    plt.figure(figsize=(16, 12))
    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_correlation_heatmap.png")
    plt.close()


# Visualization 6: URL Length vs Domain Length by Label


def url_vs_domain_length_by_label(df: pd.DataFrame):
    """
    Plot and save a scatter plot of URL length vs domain length by label.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="url_length", y="domain_length", hue="label", data=df)
    plt.title("URL Length vs Domain Length by Label")
    plt.xlabel("URL Length")
    plt.ylabel("Domain Length")
    plt.legend()
    plt.savefig(f"{output_dir}/url_vs_domain_length_by_label.png")
    plt.close()


# Visualization 7: Scatter Plot for Feature Pairs


def scatter_plot_feature_pairs(df: pd.DataFrame, feature_pairs: list):
    """
    Plot and save scatter plots for specified feature pairs.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        feature_pairs (list): List of tuples, each containing two feature names to plot.
    """
    for feature_x, feature_y in feature_pairs:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature_x, y=feature_y, hue="label", data=df, palette="coolwarm", alpha=0.6)
        plt.title(f"{feature_x} vs {feature_y} by Label")
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.legend(title="Label", labels=["Normal", "Phishing"])
        plt.savefig(f"{output_dir}/{feature_x}_vs_{feature_y}_by_label.png")
        plt.close()


# Visualization 8: Boxplot Comparisons for Numeric Features


def boxplot_numeric_features(df: pd.DataFrame, numeric_features: list):
    """
    Plot and save boxplots for numeric features grouped by label.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        numeric_features (list): List of numeric feature names to plot.
    """
    for feature in numeric_features:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x="label", y=feature, data=df, palette="coolwarm")
        plt.title(f"{feature} by Label")
        plt.xlabel("Label")
        plt.ylabel(feature)
        plt.savefig(f"{output_dir}/{feature}_by_label.png")
        plt.close()


# Visualization 9: Feature Importance using Random Forest


def feature_importance_random_forest(df: pd.DataFrame):
    """
    Plot and save a bar chart of feature importance using a Random Forest model.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
    """
    features = df.drop(columns=["url", "label"])
    labels = df["label"]

    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features = features.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    feature_names = features.columns
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="viridis")
    plt.title("Feature Importance from Random Forest")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.savefig(f"{output_dir}/feature_importance_random_forest.png")
    plt.close()


# Visualization 10: Cumulative Feature Importance Analysis


def cumulative_feature_importance(df: pd.DataFrame):
    """
    Plot and save a cumulative feature importance chart using a Random Forest model.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
    """
    features = df.drop(columns=["url", "label"])
    labels = df["label"]

    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features = features.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    feature_names = features.columns
    cumulative_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False).cumsum()

    plt.figure(figsize=(12, 8))
    sns.lineplot(x=np.arange(1, len(cumulative_importance) + 1), y=cumulative_importance, marker="o")
    plt.title("Cumulative Feature Importance")
    plt.xlabel("Number of Features")
    plt.ylabel("Cumulative Importance")
    plt.axhline(y=0.95, color="r", linestyle="--", label="95% Threshold")
    plt.axvline(
        x=len(cumulative_importance[cumulative_importance < 0.95]) + 1,
        color="g",
        linestyle="--",
        label="Features for 95%",
    )
    plt.legend()
    plt.savefig(f"{output_dir}/cumulative_feature_importance.png")
    plt.close()


# Visualization 11: Distribution of Risk Scores


def risk_score_distribution(df: pd.DataFrame):
    """
    Plot and save the distribution of risk scores.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df["risk_score"], bins=30, kde=True, color="blue")
    plt.title("Distribution of Risk Scores")
    plt.xlabel("Risk Score")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_dir}/risk_score_distribution.png")
    plt.close()


# Visualization 12: Boxplot of URL Length by Label


def url_length_boxplot_by_label(df: pd.DataFrame):
    """
    Plot and save a boxplot of URL lengths by label.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="label", y="url_length", data=df, palette="Set2")
    plt.title("URL Length Comparison: Phishing (1) vs Legitimate (0)")
    plt.xlabel("Label (0 = Legitimate, 1 = Phishing)")
    plt.ylabel("URL Length")
    plt.savefig(f"{output_dir}/url_length_boxplot_by_label.png")
    plt.close()


# Visualization 13: Feature Count Comparison


def feature_count_comparison(df: pd.DataFrame):
    """
    Plot and save a bar chart comparing feature counts.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
    """
    feature_counts = df[["has_http", "contains_ip_address"]].sum().reset_index()
    feature_counts.columns = ["Feature", "Count"]
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Feature", y="Count", data=feature_counts, palette="viridis")
    plt.title("Feature Count Comparison")
    plt.xlabel("Feature")
    plt.ylabel("Count")
    plt.savefig(f"{output_dir}/feature_count_comparison.png")
    plt.close()


# Visualization 14: Scatter Plot of URL Length vs Risk Score


def url_length_vs_risk_score(df: pd.DataFrame):
    """
    Plot and save a scatter plot of URL length vs risk score, grouped by label.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="url_length", y="risk_score", hue="label", data=df, palette="cool", alpha=0.7, edgecolor="w", linewidth=0.5
    )
    plt.title("URL Length vs Risk Score with Labels")
    plt.xlabel("URL Length")
    plt.ylabel("Risk Score")
    plt.savefig(f"{output_dir}/url_length_vs_risk_score.png")
    plt.close()


# Visualization 15: Pie Chart of Labels


def label_pie_chart(df: pd.DataFrame):
    """
    Plot and save a pie chart showing the proportion of legitimate vs phishing URLs.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    plt.figure(figsize=(8, 8))
    label_counts = df["label"].value_counts()
    plt.pie(
        label_counts,
        labels=["Legitimate (0)", "Phishing (1)"],
        autopct="%1.1f%%",
        startangle=140,
        colors=["green", "red"],
    )
    plt.title("Proportion of Legitimate vs Phishing URLs")
    plt.savefig(f"{output_dir}/label_pie_chart.png")
    plt.close()


# Visualization 16: Hexbin Plot of URL Length vs Risk Score


def hexbin_url_length_vs_risk_score(df: pd.DataFrame):
    """
    Plot and save a hexbin plot of URL length vs risk score.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    plt.figure(figsize=(10, 8))
    plt.hexbin(x=df["url_length"], y=df["risk_score"], gridsize=30, cmap="coolwarm", bins="log")
    plt.title("Hexbin Plot: URL Length vs Risk Score")
    plt.xlabel("URL Length")
    plt.ylabel("Risk Score")
    plt.colorbar(label="Log Density")
    plt.savefig(f"{output_dir}/hexbin_url_length_vs_risk_score.png")
    plt.close()


# Visualization 17: Histogram of Risk Scores by Label


def histogram_risk_scores_by_label(df: pd.DataFrame):
    """
    Plot and save a histogram of risk scores, grouped by label.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    plt.figure(figsize=(10, 6))
    for label in [0, 1]:
        subset = df[df["label"] == label]
        plt.hist(subset["risk_score"], bins=30, alpha=0.5, label=f"Label {label}")
    plt.title("Histogram of Risk Score by Label")
    plt.xlabel("Risk Score")
    plt.ylabel("Frequency")
    plt.legend(title="Label")
    plt.savefig(f"{output_dir}/histogram_risk_scores_by_label.png")
    plt.close()


# Visualization 18: Violin Plot of Domain Length by Label


def domain_length_violin_by_label(df: pd.DataFrame):
    """
    Plot and save a violin plot of domain length, grouped by label.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="label", y="domain_length", data=df, scale="count", inner="quartile", palette="muted")
    plt.title("Violin Plot: Domain Length by Label")
    plt.xlabel("Label (0 = Legitimate, 1 = Phishing)")
    plt.ylabel("Domain Length")
    plt.savefig(f"{output_dir}/domain_length_violin_by_label.png")
    plt.close()


# Visualization 19: Heatmap of Feature Interactions by Label


def feature_interactions_heatmap(df: pd.DataFrame):
    """
    Plot and save a heatmap of average feature values, grouped by label.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    plt.figure(figsize=(12, 8))
    label_grouped = df.select_dtypes(include=["number"]).groupby("label").mean()
    sns.heatmap(label_grouped.T, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True, linewidths=0.5)
    plt.title("Feature Averages by Label")
    plt.xlabel("Label (0 = Legitimate, 1 = Phishing)")
    plt.ylabel("Features")
    plt.savefig(f"{output_dir}/feature_interactions_heatmap.png")
    plt.close()


# Visualization 20: Feature Importance Correlation Bar Plot


def feature_importance_correlation_bar(df: pd.DataFrame):
    """
    Plot and save a bar plot of feature importance based on correlation with the label.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    numerical_features = df.select_dtypes(include=["number"])
    feature_importance = numerical_features.corr()["label"].sort_values(ascending=False).drop("label")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis")
    plt.title("Feature Importance Based on Correlation with Label")
    plt.xlabel("Correlation with Label")
    plt.ylabel("Feature")
    plt.savefig(f"{output_dir}/feature_importance_correlation_bar.png")
    plt.close()


# Visualization 21: Confusion Matrix for Random Forest


def plot_confusion_matrix(df: pd.DataFrame):
    """
    Plot and save a confusion matrix for a Random Forest model.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    features = df.drop(columns=["url", "label"])
    labels = df["label"]

    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features = features.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Phishing"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix for Random Forest")
    plt.savefig(f"{output_dir}/confusion_matrix.png")


# Visualization 22: ROC Curve for Random Forest


def plot_roc_curve(df: pd.DataFrame):
    """
    Plot and save an ROC curve for a Random Forest model.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    features = df.drop(columns=["url", "label"])
    labels = df["label"]

    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features = features.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Random Forest")
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()


# Visualization 23: Feature Density Plot by Label


def plot_feature_density_by_label(df: pd.DataFrame):
    """
    Plot and save density plots for numeric features grouped by label.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    numeric_features = df.select_dtypes(include=["float64", "int64"]).columns
    for feature in numeric_features:
        if feature != "label":
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=df, x=feature, hue="label", common_norm=False, fill=True, alpha=0.5, palette="muted")
            plt.title(f"Density Plot of {feature} by Label")
            plt.xlabel(feature)
            plt.ylabel("Density")
            plt.savefig(f"{output_dir}/feature_density_plot_by_label.png")
            plt.close()


# Visualization 24: Pairplot of Features by Label


def plot_pairplot_features(df: pd.DataFrame):
    """
    Plot and save pairplots for numeric features grouped by label.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    numeric_features = df.select_dtypes(include=["float64", "int64"]).columns
    numeric_features = numeric_features.drop("label", errors="ignore")
    sns.pairplot(df, vars=numeric_features, hue="label", palette="husl", diag_kind="kde")
    plt.suptitle("Pairplot of Features by Label", y=1.02)
    plt.savefig(f"{output_dir}/pairplot_features.png")
    plt.close()


# Visualization 25: Histogram of Top Correlated Features


def plot_top_correlated_features(df: pd.DataFrame):
    """
    Plot and save histograms of the top correlated features with the label.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    """
    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr()
    top_features = corr_matrix["label"].abs().sort_values(ascending=False).index[1:6]  # Exclude 'label'
    df[top_features].hist(bins=30, figsize=(12, 10), layout=(2, 3), color="blue", alpha=0.7)
    plt.suptitle("Histogram of Top Correlated Features", fontsize=16)
    plt.savefig(f"{output_dir}/top_correlated_features_histogram.png")
    plt.close()


# Visualization 26 & 27: Pairplot of Top Features by CV


def pairplot_top_features(df: pd.DataFrame, label_value: int):
    """
    Plot and save pairplots and boxplots for top features with the highest coefficient of variation (CV).

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        label_value (int): The label value (0 or 1) to use in plots.
    """
    data_float = df.copy()
    cv = data_float.std() / data_float.mean()
    top_training_features = int(np.sqrt(170))
    top_features = cv.sort_values(ascending=False).head(top_training_features).index
    data_top = data_float[top_features]
    data_top["label"] = label_value

    sns.pairplot(data_top, hue="label", diag_kind="kde", palette={1: "red", 0: "blue"})
    plt.savefig(f"{output_dir}/pairplot_top_features.png")
    plt.close()

    data_top.boxplot(showfliers=False)
    plt.title("Box Plots of Top Features with Highest CV")
    plt.xticks(rotation=45)
    plt.savefig(f"{output_dir}/boxplot_top_features.png", dpi=300, bbox_inches="tight")
    plt.close()


# Visualization 28: Generate WordClouds


def generate_wordcloud(data: pd.DataFrame, label_value: int, title: str):
    """
    Generate and save a WordCloud for the given label.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing the text data.
        label_value (int): The label value to filter data for WordCloud generation.
        title (str): The title for the WordCloud plot.
    """
    subset = data[data["label"] == label_value]
    all_text = " ".join(subset["text_sent"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()
    output_path = os.path.join(output_dir, f"wordcloud_label_{label_value}.png")
    wordcloud.to_file(output_path)
    print(f"WordCloud successfully saved to: {output_path}")


def generate_wordclouds():
    """
    Generate and save WordClouds for each label (e.g., legitimate and phishing).
    """
    openphish_urls_file = os.path.abspath("../data/raw/openphish_urls.csv")
    urls_file = os.path.abspath("../data/processed/all_urls.csv")
    full_urls = list(pd.read_csv(urls_file)["Values"])
    openphish_urls = list(pd.read_csv(openphish_urls_file)["URL"])

    nlp_data = prepare_and_process_nlp_data(full_urls, openphish_urls)
    label_values = nlp_data["label"].unique()  # e.g., [0, 1]
    for label_value in label_values:
        title = f"WordCloud for Label {label_value}"
        generate_wordcloud(nlp_data, label_value, title)


# Visualization 29-34: Generate ROC Curves


def plot_and_calculate_auc(
    y_true: np.ndarray, y_pred_prob: np.ndarray, title: str = "ROC Curve", output_path: str = None
) -> float:
    """
    Plot and save a ROC curve, and calculate the AUC score.

    Parameters:
        y_true (np.ndarray): True labels.
        y_pred_prob (np.ndarray): Predicted probabilities.
        title (str): Title for the plot.
        output_path (str): Path to save the plot.

    Returns:
        float: The AUC score.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved ROC curve to {output_path}")
    else:
        plt.show()
    plt.close()
    return auc


def generate_ROC_curve():
    """
    Generate and save ROC curves for multiple models using test results.
    """
    results_df = pd.read_csv(os.path.abspath("../data/processed/model_results.csv"), index_col=0)

    knn_result_y_test = results_df["knn_result_y_test"].dropna().to_numpy()
    knn_result_y_test_pred_prob = results_df["knn_result_y_test_pred_prob"].dropna().to_numpy()

    LR_result_y_test = results_df["LR_result_y_test"].dropna().to_numpy()
    LR_y_test_pred_prob = results_df["LR_y_test_pred_prob"].dropna().to_numpy()

    RF_result_y_test = results_df["RF_result_y_test"].dropna().to_numpy()
    RF_y_test_pred_prob = results_df["RF_y_test_pred_prob"].dropna().to_numpy()

    SVM_result_y_test = results_df["SVM_result_y_test"].dropna().to_numpy()
    SVM_y_test_pred_prob = results_df["SVM_y_test_pred_prob"].dropna().to_numpy()

    y_test_nb = results_df["y_test_nb"].dropna().to_numpy()
    y_prob_nb = results_df["y_prob_nb"].dropna().to_numpy()

    y_test_xgb = results_df["y_test_xgb"].dropna().to_numpy()
    y_prob_xgb = results_df["y_prob_xgb"].dropna().to_numpy()

    plot_and_calculate_auc(
        knn_result_y_test,
        knn_result_y_test_pred_prob,
        "ROC Curve (KNN Test Set)",
        os.path.join(output_dir, "roc_knn.png"),
    )
    plot_and_calculate_auc(
        LR_result_y_test,
        LR_y_test_pred_prob,
        "ROC Curve (Logistic Regression Test Set)",
        os.path.join(output_dir, "roc_lr.png"),
    )
    plot_and_calculate_auc(
        RF_result_y_test,
        RF_y_test_pred_prob,
        "ROC Curve (Random Forest Test Set)",
        os.path.join(output_dir, "roc_rf.png"),
    )
    plot_and_calculate_auc(
        SVM_result_y_test, SVM_y_test_pred_prob, "ROC Curve (SVM Test Set)", os.path.join(output_dir, "roc_svm.png")
    )
    plot_and_calculate_auc(
        y_test_nb, y_prob_nb, "ROC Curve (Naïve Bayes Test Set)", os.path.join(output_dir, "roc_nb.png")
    )
    plot_and_calculate_auc(
        y_test_xgb, y_prob_xgb, "ROC Curve (XGBoost Test Set)", os.path.join(output_dir, "roc_xgb.png")
    )


# Main function to execute all visualizations
def main():
    csv_path = os.path.abspath("../data/processed/final_test_features.csv")
    df = load_data(csv_path)
    df = fill_missing_values(df)
    df = convert_to_numeric(
        df, ["url_length", "domain_length", "subdomain_count", "path_length", "risk_score", "malicious_count"]
    )

    # Call visualization functions
    url_length_distribution(df)
    domain_length_by_label(df)
    subdomain_count_by_label(df)
    path_length_kde_by_label(df)
    feature_correlation_heatmap(df)
    url_vs_domain_length_by_label(df)

    scatter_plot_feature_pairs(df, [("url_length", "domain_length"), ("subdomain_count", "path_length")])
    boxplot_numeric_features(df, ["domain_length", "subdomain_count", "risk_score", "malicious_count"])
    feature_importance_random_forest(df)
    cumulative_feature_importance(df)
    risk_score_distribution(df)
    url_length_boxplot_by_label(df)
    feature_count_comparison(df)
    url_length_vs_risk_score(df)
    label_pie_chart(df)
    hexbin_url_length_vs_risk_score(df)
    histogram_risk_scores_by_label(df)
    domain_length_violin_by_label(df)
    feature_interactions_heatmap(df)
    feature_importance_correlation_bar(df)
    plot_confusion_matrix(df)
    plot_roc_curve(df)
    plot_feature_density_by_label(df)
    plot_pairplot_features(df)
    plot_top_correlated_features(df)
    generate_wordclouds()
    ml_data = load_data(csv_path)
    ml_data_float = clean_data(ml_data)
    pairplot_top_features(ml_data_float, ml_data["label"])
    generate_ROC_curve()

    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
