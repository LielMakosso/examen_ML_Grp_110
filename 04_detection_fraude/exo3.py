import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict


#Génération et découpage
rng = np.random.default_rng(42)
n = 20_000
tx = pd.DataFrame({
    "montant": rng.lognormal(3.5, 1.0, n).round(2),
    "heure": rng.integers(0, 24, n),
    "distance_km": rng.exponential(10, n).round(1),
    "nb_transactions_24h": rng.poisson(2, n),
    "etranger": (rng.random(n) < 0.05).astype(int),
})

nuit = ((tx["heure"] < 6) | (tx["heure"] >= 23)).astype(int)
logit = (-11.5 + np.log(tx["montant"]) + 2.0 * nuit + 0.08 * tx["distance_km"] + 
         0.5 * tx["nb_transactions_24h"] + 3.0 * tx["etranger"])
tx["fraude"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

print(f"Taux de fraude : {tx['fraude'].mean() * 100:.2f} %")

tx["log_montant"] = np.log(tx["montant"])
tx["nuit"] = nuit

X = tx.drop(columns=["fraude"])
y = tx["fraude"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Forme du train : {X_train.shape}, Forme du test : {X_test.shape}")
print(f"Taux de fraude dans le train : {y_train.mean()*100:.2f} %")
print(f"Taux de fraude dans le test  : {y_test.mean()*100:.2f} %")


#Modèle naïf et Régression Logistique

print("\n--- Le piège du modèle naïf ---")

dummy = DummyClassifier(strategy="constant", constant=0)
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)
accuracy_dummy = accuracy_score(y_test, y_pred_dummy)
print(f"Exactitude du modèle naïf : {accuracy_dummy * 100:.2f} %")

# On retire la variable 'heure' brute car on a créé 'nuit' à partir d'elle
X_train_lr = X_train.drop(columns=["heure"])
X_test_lr = X_test.drop(columns=["heure"])

pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(random_state=42, max_iter=1000))
])

print("\n--- Probabilités via Validation Croisée ---")
y_probas_cv = cross_val_predict(pipe_lr, X_train_lr, y_train, cv=5, method="predict_proba")
y_scores_cv = y_probas_cv[:, 1] 
print("Probabilités d'entraînement générées sans fuite de données.")


# Matrice de confusion, Seuil et Coût métier
print("\n--- Matrice de confusion (seuil 0.5) ---")
from sklearn.metrics import confusion_matrix, classification_report

# Entraînement sur tout le train
pipe_lr.fit(X_train_lr, y_train)

# Prédictions sur le test avec le seuil par défaut (0.5)
y_pred_05 = pipe_lr.predict(X_test_lr)
cm_05 = confusion_matrix(y_test, y_pred_05)
print("Matrice de confusion :")
print(cm_05)

# Optimisation du seuil métier
print("\n--- Optimisation du seuil métier ---")
# On utilise les probabilités générées sur l'entraînement (cross_val_predict) pour choisir le seuil
# On ne touche pas au test pour l'instant !
from sklearn.metrics import recall_score, precision_score

seuils = np.linspace(0.01, 0.99, 99)
cout_total = []

# Coût d'une fraude manquée (FN) = 200€, Coût d'une fausse alerte (FP) = 10€
cout_FN = 200
cout_FP = 10

for seuil in seuils:
    # Prédiction binaire basée sur le seuil
    y_pred_seuil = (y_scores_cv >= seuil).astype(int)
    
    # Calcul des matrices
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred_seuil).ravel()
    
    # Calcul du coût total
    cout = (fn * cout_FN) + (fp * cout_FP)
    cout_total.append(cout)

# Trouver le seuil optimal
idx_optimal = np.argmin(cout_total)
seuil_optimal = seuils[idx_optimal]
cout_min = cout_total[idx_optimal]

print(f"Seuil optimal trouvé : {seuil_optimal:.2f}")
print(f"Coût total minimal estimé (sur train) : {cout_min} €")

# Tracé de la courbe du coût
plt.figure(figsize=(8, 5))
plt.plot(seuils, cout_total, color='purple', lw=2)
plt.axvline(x=0.5, color='red', linestyle='--', label='Seuil par défaut (0.5)')
plt.axvline(x=seuil_optimal, color='green', linestyle='--', label=f'Seuil optimal ({seuil_optimal:.2f})')
plt.title("Coût total en fonction du seuil de décision")
plt.xlabel("Seuil")
plt.ylabel("Coût total (€)")
plt.legend()
plt.grid(True)
plt.show()

#Application unique sur le test
print("\n--- Application du seuil optimal sur le TEST ---")
# On récupère les probabilités sur le test
y_scores_test = pipe_lr.predict_proba(X_test_lr)[:, 1]
# On applique le seuil optimal
y_pred_optimal = (y_scores_test >= seuil_optimal).astype(int)

# Matrice de confusion avec le seuil optimal
tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_test, y_pred_optimal).ravel()
cout_test_optimal = (fn_opt * cout_FN) + (fp_opt * cout_FP)

# Coût sans détection (le modèle naïf)
cout_sans_detection = y_test.sum() * cout_FN 

print(f"Avec le seuil optimal ({seuil_optimal:.2f}) sur le test :")
print(f" - Fraudes ratées (FN) : {fn_opt}")
print(f" - Fausses alertes (FP) : {fp_opt}")
print(f" - Coût total sur le test : {cout_test_optimal} €")
print(f" - Coût si aucune détection : {cout_sans_detection} €")
print(f" - Économie réalisée : {cout_sans_detection - cout_test_optimal} €")

# Comparaison avec Arbres et Forêts
print("\n--- Arbres de décision et Forêts aléatoires ---")
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Arbre de décision
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train_lr, y_train)
y_pred_tree = tree.predict(X_test_lr)
f1_tree = f1_score(y_test, y_pred_tree)

# Forêt aléatoire
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_lr, y_train)
y_pred_rf = rf.predict(X_test_lr)
f1_rf = f1_score(y_test, y_pred_rf)

# Régression logistique pour comparer
y_pred_lr_test = pipe_lr.predict(X_test_lr)
f1_lr = f1_score(y_test, y_pred_lr_test)

print(f"F1-score (Régression Logistique) : {f1_lr:.3f}")
print(f"F1-score (Arbre de décision)     : {f1_tree:.3f}")
print(f"F1-score (Forêt aléatoire)       : {f1_rf:.3f}")


# Courbes ROC et PR

print("\n---Courbes ROC et Précision-Rappel ---")
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Calcul des courbes
fpr, tpr, _ = roc_curve(y_test, y_scores_test)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_test, y_scores_test)
pr_auc = average_precision_score(y_test, y_scores_test)

# Tracé côte à côte
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Courbe ROC
ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('Taux de Faux Positifs (FPR)')
ax1.set_ylabel('Taux de Vrais Positifs (TPR / Rappel)')
ax1.set_title('Courbe ROC')
ax1.legend(loc="lower right")
ax1.grid(True)

# Courbe Précision-Rappel
ax2.plot(recall, precision, color='green', lw=2, label=f'PR (AUC = {pr_auc:.3f})')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Rappel')
ax2.set_ylabel('Précision')
ax2.set_title('Courbe Précision-Rappel')
ax2.legend(loc="upper right")
ax2.grid(True)

plt.tight_layout()
plt.show()

print(f"Score ROC-AUC : {roc_auc:.3f}")
print(f"Score PR-AUC  : {pr_auc:.3f}")