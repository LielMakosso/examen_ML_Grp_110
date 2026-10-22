import os
import json
import datetime
import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Chargement des données
print("Chargement des données...")
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

print(f"Forme de X : {X.shape}")
print(f"Forme de y : {y.shape}")
print(f"Variables : {feature_names}")

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Création du pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

#Validation croisée sur l'ensemble d'entraînement
print("\n--- Validation Croisée (LinearRegression) ---")
cv_mae = -cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_rmse = -cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')

print(f"MAE (CV)  : {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")
print(f"RMSE (CV) : {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
print(f"R² (CV)   : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

#Entraînement final sur tout le train et évaluation sur le test
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = root_mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print("\n--- Évaluation sur le Test ---")
print(f"MAE (Test)  : {test_mae:.2f}")
print(f"RMSE (Test) : {test_rmse:.2f}")
print(f"R² (Test)   : {test_r2:.4f}")

# Comparaison avec Ridge
print("\n--- Recherche du meilleur alpha pour Ridge ---")
ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

param_grid = {'ridge__alpha': np.logspace(-3, 3, 50)}
grid_search = GridSearchCV(ridge_pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

best_alpha = grid_search.best_params_['ridge__alpha']
best_ridge_score = -grid_search.best_score_

print(f"Meilleur alpha trouvé : {best_alpha:.4f}")
print(f"MAE CV (Ridge)        : {best_ridge_score:.2f}")

# Évaluation de Ridge sur le test
best_ridge = grid_search.best_estimator_
y_pred_ridge = best_ridge.predict(X_test)
test_rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)

print(f"RMSE (Test Ridge)     : {test_rmse_ridge:.2f}")
print(f"Le gain de Ridge par rapport à la régression linéaire est de {test_rmse - test_rmse_ridge:.2f} en RMSE.")

#Graphique Prédictions vs Réel
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Prédictions vs Réel (Test)")
plt.grid(True)
plt.show()

#Sauvegarde du modèle et des métadonnées
# Création du dossier "modeles" s'il n'existe pas
os.makedirs("modeles", exist_ok=True)

#Sauvegarde du pipeline complet
joblib.dump(pipeline, "modeles/diabete.joblib")

#Sauvegarde des informations dans un JSON
infos = {
    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "versions": {
        "python": ".".join(map(str, __import__('sys').version_info[:3])),
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__
    },
    "variables_attendues": feature_names,
    "metriques_test": {
        "MAE": round(test_mae, 2),
        "RMSE": round(test_rmse, 2),
        "R2": round(test_r2, 4)
    }
}

with open("modeles/diabete_infos.json", "w", encoding="utf-8") as f:
    json.dump(infos, f, indent=4, ensure_ascii=False)

print("\nModèle et métadonnées sauvegardés dans le dossier 'modeles/'.")