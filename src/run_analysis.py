import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from clean_data import clean_data, prepare_and_process_nlp_data
from utils.models import (LogisticRegression, Multinomial, XGBoost, knn,
                          random_forest, train_svm)

if __name__ == "__main__":

    input_file = os.path.abspath("../data/processed/final_test_features.csv")
    urls_file = os.path.abspath("../data/processed/all_urls.csv")
    openphish_urls_file = os.path.abspath("../data/raw/openphish_urls.csv")
    model_results_file = os.path.abspath("../data/processed/model_results.csv")

    print("Step 1: Loading and cleaning data...")
    data = pd.read_csv(input_file)
    data = clean_data(data)

    print("Step 2: Calculating correlation and selecting top features...")
    cv = data.std() / data.mean()
    top_training_features = int(np.sqrt(170))
    top_features = cv.sort_values(ascending=False).head(top_training_features).index
    data_top = data[top_features].copy()
    data_top["label"] = data["label"]

    print(data_top.describe())

    print("Step 3: Splitting dataset...")
    X = data_top.drop(columns="label")
    y = data_top["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Step 4: Training models...")
    knn_result_y_test, knn_result_y_test_pred_prob = knn(X_train, y_train, X_test, y_test)
    LR_result_y_test, LR_y_test_pred_prob = LogisticRegression(X_train, y_train, X_test, y_test)
    RF_result_y_test, RF_y_test_pred_prob = random_forest(X_train, y_train, X_test, y_test)
    SVM_result_y_test, SVM_y_test_pred_prob = train_svm(X_train, y_train, X_test, y_test)

    print("Step 5: Processing NLP data...")
    full_urls = list(pd.read_csv(urls_file)["Values"])
    openphish_urls = list(pd.read_csv(openphish_urls_file)["URL"])
    nlp_data = prepare_and_process_nlp_data(full_urls, openphish_urls)

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=2000)
    X_nlp = vectorizer.fit_transform(nlp_data["text_sent"]).toarray()
    y_nlp = nlp_data["label"]

    X_train_nlp, X_test_nlp, y_train_nlp, y_test_nlp = train_test_split(X_nlp, y_nlp, test_size=0.3, random_state=42)

    y_test_nb, y_prob_nb = Multinomial(X_train_nlp, y_train_nlp, X_test_nlp, y_test_nlp)
    y_test_xgb, y_prob_xgb = XGBoost(X_train, y_train, X_test, y_test)

    data_dict = {
        "knn_result_y_test": list(knn_result_y_test),
        "knn_result_y_test_pred_prob": list(knn_result_y_test_pred_prob),
        "LR_result_y_test": list(LR_result_y_test),
        "LR_y_test_pred_prob": list(LR_y_test_pred_prob),
        "RF_result_y_test": list(RF_result_y_test),
        "RF_y_test_pred_prob": list(RF_y_test_pred_prob),
        "SVM_result_y_test": list(SVM_result_y_test),
        "SVM_y_test_pred_prob": list(SVM_y_test_pred_prob),
        "y_test_nb": list(y_test_nb),
        "y_prob_nb": list(y_prob_nb),
        "y_test_xgb": list(y_test_xgb),
        "y_prob_xgb": list(y_prob_xgb),
    }

    results_df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in data_dict.items()]))

    results_df.to_csv(model_results_file, index=True)
