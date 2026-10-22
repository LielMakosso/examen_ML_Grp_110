import json
import joblib
import sklearn
import sys
import pandas as pd
import numpy as np

print("=== SCRIPT DE PRÉDICTION (PRODUCTION) ===\n")

#Rechargement du pipeline et des métadonnées
chemin_modele = "modeles/diabete.joblib"
chemin_infos = "modeles/diabete_infos.json"

try:
    pipeline_charge = joblib.load(chemin_modele)
    with open(chemin_infos, "r", encoding="utf-8") as f:
        infos = json.load(f)
    print("Modèle et métadonnées chargés avec succès.")
except FileNotFoundError:
    print("Erreur : Impossible de trouver les fichiers du modèle.")
    sys.exit(1)

#Vérification des versions (Scikit-learn et Python)
version_sklearn_actuelle = sklearn.__version__
version_python_actuelle = ".".join(map(str, sys.version_info[:3]))

version_sklearn_modele = infos["versions"]["scikit_learn"]
version_python_modele = infos["versions"]["python"]

if version_sklearn_actuelle != version_sklearn_modele:
    print(f"\n AVERTISSEMENT : Version de scikit-learn différente !")
    print(f"Modèle entraîné avec : {version_sklearn_modele}")
    print(f"Version actuelle     : {version_sklearn_actuelle}")
    print("Les prédictions peuvent être instables.\n")
else:
    print(f"Version de scikit-learn compatible ({version_sklearn_actuelle}).")

#Prédiction pour un nouveau patient (dictionnaire)
print("\n--- Prédiction pour 1 patient ---")
patient_1 = {
    "age": 0.038075, 
    "sex": 0.050680,
    "bmi": 0.061696,
    "bp": 0.021872,
    "s1": -0.044223,
    "s2": -0.034820,
    "s3": -0.043400,
    "s4": -0.002592,
    "s5": 0.019907,
    "s6": -0.017646
}

#On crée un DataFrame pour garantir l'ordre des colonnes
df_patient_1 = pd.DataFrame([patient_1])[infos["variables_attendues"]]

#Prédiction
prediction_1 = pipeline_charge.predict(df_patient_1.values)[0]
rmse_test = infos["metriques_test"]["RMSE"]

print(f"Progression estimée : {prediction_1:.0f} (à ± {rmse_test:.0f} près)")

#Prédiction pour 3 patients depuis un fichier CSV
print("\n--- Prédiction pour 3 patients (depuis CSV) ---")

# Création du fichier CSV de test
donnees_csv = [
    [0.038075, 0.050680, 0.061696, 0.021872, -0.044223, -0.034820, -0.043400, -0.002592, 0.019907, -0.017646],
    [-0.001882, -0.044641, -0.051474, -0.026327, -0.008448, -0.019163, 0.074412, -0.039493, -0.068330, -0.092204],
    [0.085299, 0.050680, 0.044451, -0.005670, -0.045599, -0.034194, -0.032355, -0.002592, 0.002863, -0.025930]
]
df_csv = pd.DataFrame(donnees_csv, columns=infos["variables_attendues"])
df_csv.to_csv("nouveaux_patients.csv", index=False)
print("Fichier 'nouveaux_patients.csv' créé.")

# Lecture du CSV et prédiction
df_nouveaux = pd.read_csv("nouveaux_patients.csv")
predictions = pipeline_charge.predict(df_nouveaux.values)

for i, pred in enumerate(predictions):
    print(f"Patient {i+1} : progression estimée à {pred:.0f} (à ± {rmse_test:.0f} près)")