# Examen Bloc A - Machine Learning (Groupe 110)

Ce dépôt contient l'intégralité du travail réalisé dans le cadre de l'examen de Machine Learning. 
Le projet est structuré de manière modulaire pour garantir la reproductibilité et la clarté du code.

## Prérequis et Installation

Ce projet a été développé avec **Python 3.12** et le gestionnaire de paquets **`uv`**. 
Pour reproduire ce travail sur une nouvelle machine (Windows, macOS ou Linux), suivez ces étapes dans l'ordre.

### 1. Recuperation du projet
clonez le dépôt :
```bash
git clone https://github.com/lielmakosso/examen_ML_Grp_110.git
cd examen_ML_Grp_110
```

## 2. Installation de uv
uv est un outil moderne qui va gérer l'environnement virtuel, installer la bonne version de Python, et télécharger les dépendances.
Installez-le avec la commande correspondant à votre système :

## Sous Windows (PowerShell) :
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Sous macOS / Linux :
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Fermez puis rouvrez votre terminal pour que la commande uv soit reconnue.

## 3. Création de l'environnement et installation des dépendances
Cette commande magique va lire le fichier .python-version (pour installer Python 3.12 si besoin), créer le dossier .venv, et télécharger toutes les bibliothèques (NumPy, Scikit-learn, etc.) aux versions exactes ## ## définies dans uv.lock :

```bash
uv sync
```

## Structure du Projet
01_environnement/ : Script environnement.py vérifiant les versions de Python et des bibliothèques installées.

02_regression_lineaire/ : Implémentation "from scratch" de la régression linéaire (MCO, descente de gradient, gradients analytiques) sur des données de loyers.

03_diabete_production/ : Pipeline scikit-learn d'entraînement sur le jeu de données Diabète, sauvegarde du modèle avec joblib, et script d'inférence (exo2_prediction.py) simulant la mise en production. Contient les modèles sérialisés.

04_detection_fraude/ : Classification fortement déséquilibrée (transactions bancaires). Inclut l'optimisation du seuil de décision basée sur un coût métier, et l'évaluation via courbes ROC et Précision-Rappel.

## Exécutions des scripts
Une fois l'installation terminée (uv sync), vous pouvez exécuter les scripts un par un depuis le terminal.
La commande uv run permet d'exécuter le script dans le bon environnement sans avoir à l'activer manuellement.

## Vérifier l'environnement :
```bash
uv run 01_environnement/environnement.py
```

## Lancer la régression linéaire :
```bash
uv run 02_regression_lineaire/exo1.py
```

## Entraîner le modèle Diabète :
```bash
uv run 03_diabete_production/exo2_entrainement.py
```

## Simuler une prédiction Diabète :
```bash
uv run 03_diabete_production/exo2_prediction.py
```

## Lancer l'analyse de détection de fraude :
```bash
uv run 04_detection_fraude/exo3.py
```