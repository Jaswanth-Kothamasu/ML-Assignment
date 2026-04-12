# ===================== IMPORTS =====================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

import shap
from lime.lime_tabular import LimeTabularExplainer


# ===================== LOAD DATA =====================
data = pd.read_csv("DCT_mal.csv")
data = data.dropna()

print("Dataset Loaded ✅")
print(data.head())


# ===================== PREPROCESS =====================
target_column = data.columns[-1]

X = data.drop(target_column, axis=1)
y = data[target_column]

# Convert categorical → numeric
X = pd.get_dummies(X)

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Preprocessing Done ✅")


# ===================== SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data Split Done ✅")


# ===================== MODELS =====================
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(),
    "MLP": MLPClassifier(max_iter=500)  # increased iterations
}


# ===================== HYPERPARAMETER TUNING =====================
param_dist = {
    "DecisionTreeClassifier": {
        "max_depth": [3, 5, 10, None],
        "criterion": ["gini", "entropy"]
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None]
    },
    "SVC": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    },
    "MLPClassifier": {
        "hidden_layer_sizes": [(50,), (100,)],
        "learning_rate_init": [0.001, 0.01]
    }
}


def tune_model(model):
    name = model.__class__.__name__

    if name not in param_dist:
        model.fit(X_train, y_train)
        return model

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist[name],
        n_iter=5,
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    return search.best_estimator_


# ===================== TRAIN + EVALUATE =====================
results = []
trained_models = {}

for name, model in models.items():
    print(f"\nProcessing {name}...")

    best_model = tune_model(model)
    trained_models[name] = best_model

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    results.append([
        name,
        accuracy_score(y_train, y_train_pred),
        accuracy_score(y_test, y_test_pred),
        precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
        recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
        f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    ])


# ===================== RESULTS TABLE =====================
results_df = pd.DataFrame(results, columns=[
    "Model", "Train Acc", "Test Acc", "Precision", "Recall", "F1 Score"
])

print("\n\nFINAL RESULTS TABLE 📊")
print(results_df)


# ===================== BEST MODEL =====================
best_model_name = results_df.sort_values(by="Test Acc", ascending=False).iloc[0]["Model"]
print(f"\n🏆 Best Model: {best_model_name}")

best_model = trained_models[best_model_name]


# ===================== SHAP =====================
print("\nRunning SHAP Explainability...")

try:
    if best_model_name in ["Decision Tree", "Random Forest", "AdaBoost"]:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test[:50])
    else:
        explainer = shap.KernelExplainer(best_model.predict_proba, X_train[:100])
        shap_values = explainer.shap_values(X_test[:10])

    shap.summary_plot(shap_values, X_test[:10])

except Exception as e:
    print("SHAP Error:", e)


# ===================== LIME =====================
print("\nRunning LIME Explainability...")

try:
    explainer_lime = LimeTabularExplainer(
        training_data=X_train,
        feature_names=[f"f{i}" for i in range(X_train.shape[1])],
        class_names=list(map(str, np.unique(y))),
        mode="classification"
    )

    exp = explainer_lime.explain_instance(
        X_test[0],
        best_model.predict_proba
    )

    exp.show_in_notebook()

except Exception as e:
    print("LIME Error:", e)