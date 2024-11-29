import numpy as np
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris, make_classification
from sklearn.preprocessing import StandardScaler

# Función para imprimir resultados
def print_results(name, y_true, y_pred):
    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# Cargar datasets
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Dataset 2: Generado sintéticamente
X_dataset2, y_dataset2 = make_classification(
    n_samples=200, n_features=5, n_informative=3, n_classes=3, n_clusters_per_class=1, random_state=42
)

# Dataset 3: Otro dataset sintético ajustado
X_dataset3, y_dataset3 = make_classification(
    n_samples=150, n_features=5, n_informative=3, n_classes=3, n_clusters_per_class=1, random_state=42
)

# Lista de datasets
datasets = [
    ("Iris Plant", X_iris, y_iris),
    ("Synthetic Dataset 2", X_dataset2, y_dataset2),
    ("Synthetic Dataset 3", X_dataset3, y_dataset3)
]

# Clasificador ajustado
mlp = MLPClassifier(
    random_state=42,
    max_iter=1000,  # Aumentamos el número de iteraciones
    learning_rate_init=0.0005,  # Reducimos la tasa de aprendizaje
    solver="adam"  # Optimizador por defecto
)

# Loop por datasets
for name, X, y in datasets:
    print(f"\nDataset: {name}")

    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hold-Out 70/30
    print("\nMethod: Hold-Out 70/30")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    print_results("MLP (Hold-Out)", y_test, y_pred)

    # 10-Fold Cross-Validation
    print("\nMethod: 10-Fold Cross-Validation")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    y_pred = cross_val_predict(mlp, X_scaled, y, cv=kf)
    print_results("MLP (10-Fold)", y, y_pred)

    # Leave-One-Out
    print("\nMethod: Leave-One-Out")
    loo = LeaveOneOut()
    y_pred = cross_val_predict(mlp, X_scaled, y, cv=loo)
    print_results("MLP (Leave-One-Out)", y, y_pred)
