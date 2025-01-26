from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def knn(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform K-Nearest Neighbors (KNN) classification to find the best value of k.

    Parameters:
        X_train (np.ndarray): Training feature set.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature set.
        y_test (np.ndarray): Test labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: True test labels and predicted probabilities for the positive class.
    """
    ks = range(1, 31)
    best_k = None
    max_accuracy = 0

    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_k = k

    print(f"The best K value is: {best_k}")

    knn_best = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
    y_test_pred = knn_best.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\nAccuracy on test set: {test_accuracy:.4f}")
    Confusion_matrix_test = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix on Test Set:\n", Confusion_matrix_test)
    y_test_pred_prob = knn_best.predict_proba(X_test)[:, 1]

    return y_test, y_test_pred_prob


def LogisticRegression(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Logistic Regression with cross-validation to select the best regularization parameter (C).

    Parameters:
        X_train (np.ndarray): Training feature set.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature set.
        y_test (np.ndarray): Test labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: True test labels and predicted probabilities for the positive class.
    """
    C_values = np.logspace(-4, 4, 100)
    model_cv = LogisticRegressionCV(Cs=C_values, cv=5, random_state=42)
    model_cv.fit(X_train, y_train)

    print(f"Best C value is: {model_cv.C_[0]}")

    y_test_pred = model_cv.predict(X_test)
    y_test_pred_prob = model_cv.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"\nAccuracy on test set: {test_accuracy:.4f}")

    Confusion_matrix_test = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix on Test Set:\n", Confusion_matrix_test)

    return y_test, y_test_pred_prob


def random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_weight: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a Random Forest classifier and evaluate its performance.

    Parameters:
        X_train (np.ndarray): Training feature set.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature set.
        y_test (np.ndarray): Test labels.
        class_weight (Optional[dict]): Class weights for handling class imbalance. Default is None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: True test labels and predicted probabilities for the positive class.
    """
    rf = RandomForestClassifier(oob_score=True, class_weight=class_weight, random_state=42)
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    y_train_prob = rf.predict_proba(X_train)[:, 1]
    y_test_prob = rf.predict_proba(X_test)[:, 1]

    train_conf_matrix = confusion_matrix(y_train, y_train_pred)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)

    train_auc = roc_auc_score(y_train, y_train_prob)
    test_auc = roc_auc_score(y_test, y_test_prob)

    train_misclassification = 1 - accuracy_score(y_train, y_train_pred)
    test_misclassification = 1 - accuracy_score(y_test, y_test_pred)

    oob_error = 1 - rf.oob_score_ if class_weight is not None else None

    print(f"Training Confusion Matrix is:\n{train_conf_matrix}")
    print(f"\nTesting Confusion Matrix is:\n{test_conf_matrix}")
    print(f"\nTrain Misclassification Rate is: {train_misclassification:.4f}")
    print(f"Test Misclassification Rate is: {test_misclassification:.4f}")
    if oob_error is not None:
        print(f"OOB Error Estimate is: {oob_error:.4f}")
    print(f"\nTrain AUC is: {train_auc:.4f}")
    print(f"Test AUC is: {test_auc:.4f}")

    return y_test, y_test_prob


def train_svm(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a Support Vector Machine (SVM) classifier with hyperparameter tuning using GridSearchCV.

    Parameters:
        X_train (np.ndarray): Training feature set.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature set.
        y_test (np.ndarray): Test labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: True test labels and predicted test labels.
    """
    model = SVC(kernel="rbf", probability=True)
    param_grid = {"C": [0.01, 0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1, 10]}

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test)
    best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None

    Confusion_matrix = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix is: \n{Confusion_matrix}")

    precision = precision_score(y_test, y_test_pred, average="weighted")
    recall = recall_score(y_test, y_test_pred, average="weighted")
    print(f"Overall Precision: {precision:.4f}, Overall Recall: {recall:.4f}")

    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy: {accuracy:.4f}")

    return y_test, y_test_pred


def Multinomial(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a Multinomial Naive Bayes classifier and evaluate its performance.

    Parameters:
        X_train (np.ndarray): Training feature set.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature set.
        y_test (np.ndarray): Test labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: True test labels and predicted probabilities for the positive class.
    """
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    y_prob_nb = nb_model.predict_proba(X_test)
    print("Naïve Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
    return y_test, y_prob_nb[:, 1]


def XGBoost(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train an XGBoost classifier and evaluate its performance.

    Parameters:
        X_train (np.ndarray): Training feature set.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature set.
        y_test (np.ndarray): Test labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: True test labels and predicted probabilities for the positive class.
    """
    xgb_model = XGBClassifier(eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)
    print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
    return y_test, y_prob_xgb[:, 1]
