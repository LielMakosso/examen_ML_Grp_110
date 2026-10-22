import numpy as np
import matplotlib.pyplot as plt

# Reconstitution du jeu de données
rng = np.random.default_rng(42)
surface = rng.uniform(20, 150, size=60)
loyer = 12 * surface + 150 + rng.normal(0, 120, size=60)

# Tracage du nuage de points
plt.figure(figsize=(8, 5))
plt.scatter(surface, loyer, color='blue', alpha=0.7, edgecolors='k')
plt.title("Relation entre la surface et le loyer")
plt.xlabel("Surface (m²)")
plt.ylabel("Loyer (€)")
plt.grid(True)
plt.show()


# Fonctions de prédiction et de coût

def predire(x, w, b):
    """
    Calcule les prédictions pour un vecteur de caractéristiques x.
    x : vecteur NumPy des surfaces
    w : pente (poids)
    b : ordonnée à l'origine (biais)
    """
    return w * x + b

def cout_mse(y_vrai, y_predit):
    """
    Calcule l'erreur quadratique moyenne (MSE).
    y_vrai : vecteur des vraies valeurs
    y_predit : vecteur des prédictions
    """
    m = len(y_vrai)
    erreur = y_vrai - y_predit
    return np.sum(erreur ** 2) / m

# test pour vérifier que les fonctions marchent
w_test = 10.0
b_test = 200.0
y_pred_test = predire(surface, w_test, b_test)
mse_test = cout_mse(loyer, y_pred_test)

print(f"Test avec w={w_test}, b={b_test} : MSE = {mse_test:.2f}")


#Solution exacte (Moindres Carrés)

print("\n--- Solution Exacte (NumPy) ---")

X = np.c_[np.ones(len(surface)), surface]  # Matrice de taille (60, 2)

theta_best = np.linalg.inv(X.T @ X) @ X.T @ loyer

b_exact, w_exact = theta_best[0], theta_best[1]
print(f"NumPy (Moindres Carrés) : w = {w_exact:.4f}, b = {b_exact:.4f}")

from sklearn.linear_model import LinearRegression

X_sklearn = surface.reshape(-1, 1) 
model_sklearn = LinearRegression()
model_sklearn.fit(X_sklearn, loyer)

print(f"Scikit-learn  : w = {model_sklearn.coef_[0]:.4f}, b = {model_sklearn.intercept_:.4f}")

# Descente de Gradient
print("\n--- Descente de Gradient ---")

mu = np.mean(surface)
sigma = np.std(surface)
surface_std = (surface - mu) / sigma

w_gd, b_gd = 0.0, 0.0
learning_rate = 0.1
n_iterations = 200
m = len(surface)
cout_historique = []
#Boucle d'entraînement
for i in range(n_iterations):
    y_pred = predire(surface_std, w_gd, b_gd)
    
    erreur = y_pred - loyer
    
    dw = (2/m) * np.sum(erreur * surface_std)
    db = (2/m) * np.sum(erreur)
    
    w_gd = w_gd - learning_rate * dw
    b_gd = b_gd - learning_rate * db
    
    cout_historique.append(cout_mse(loyer, y_pred))

w_original = w_gd / sigma
b_original = b_gd - (w_gd * mu / sigma)

print(f"Paramètres appris (normalisés) : w = {w_gd:.4f}, b = {b_gd:.4f}")
print(f"Paramètres dans l'unité d'origine : w = {w_original:.4f}, b = {b_original:.4f}")

#Tracé de la courbe du coût
plt.figure(figsize=(8, 4))
plt.plot(range(n_iterations), cout_historique, color='red', linewidth=2)
plt.title("Évolution du coût (MSE) au fil des itérations")
plt.xlabel("Itérations")
plt.ylabel("Coût MSE")
plt.grid(True)
plt.show()


# Impact du taux d'apprentissage
print("\n--- Test des taux d'apprentissage ---")

def descente_gradient_alpha(x_std, y, alpha, n_iter=200):
    w, b = 0.0, 0.0
    m_local = len(x_std)
    couts = []
    
    for _ in range(n_iter):
        y_pred = predire(x_std, w, b)
        erreur = y_pred - y
        
        dw = (2/m_local) * np.sum(erreur * x_std)
        db = (2/m_local) * np.sum(erreur)
        
        w = w - alpha * dw
        b = b - alpha * db
        
        cout = cout_mse(y, y_pred)
        couts.append(cout)
        
        if np.isnan(cout) or np.isinf(cout) or cout > 1e10:
            break
            
    return couts

# On teste 3 taux
alphas = [0.001, 0.1, 5.0]
couleurs = ['blue', 'green', 'red']
plt.figure(figsize=(10, 6))

for alpha, couleur in zip(alphas, couleurs):
    couts = descente_gradient_alpha(surface_std, loyer, alpha)
    plt.plot(couts, label=f"Taux = {alpha}", color=couleur, linewidth=2)

plt.title("Impact du taux d'apprentissage sur la convergence")
plt.xlabel("Itérations")
plt.ylabel("Coût (MSE)")
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Vérification du gradient par différences finies
print("\n--- Vérification du Gradient ---")

# On prend des paramètres initiaux au hasard
w_test = 10.0
b_test = 200.0

# Gradient analytique
y_pred = predire(surface_std, w_test, b_test)
erreur = y_pred - loyer
dw_analytique = (2/m) * np.sum(erreur * surface_std)
db_analytique = (2/m) * np.sum(erreur)

# Gradient numérique
epsilon = 1e-5

# Dérivée partielle par rapport à w
cout_w_plus = cout_mse(loyer, predire(surface_std, w_test + epsilon, b_test))
cout_w_moins = cout_mse(loyer, predire(surface_std, w_test - epsilon, b_test))
dw_numerique = (cout_w_plus - cout_w_moins) / (2 * epsilon)

# Dérivée partielle par rapport à b
cout_b_plus = cout_mse(loyer, predire(surface_std, w_test, b_test + epsilon))
cout_b_moins = cout_mse(loyer, predire(surface_std, w_test, b_test - epsilon))
db_numerique = (cout_b_plus - cout_b_moins) / (2 * epsilon)

# Comparaison
ecart_w = abs(dw_analytique - dw_numerique)
ecart_b = abs(db_analytique - db_numerique)

print(f"Gradient dW (analytique) : {dw_analytique:.6f}")
print(f"Gradient dW (numérique)  : {dw_numerique:.6f}")
print(f"Écart dW                 : {ecart_w:.6e}")

print(f"Gradient dB (analytique) : {db_analytique:.6f}")
print(f"Gradient dB (numérique)  : {db_numerique:.6f}")
print(f"Écart dB                 : {ecart_b:.6e}")

print(f"\nÉcart maximal observé : {max(ecart_w, ecart_b):.6e}")


# Calcul des métriques finales
print("\n--- Métriques d'évaluation ---")

# On utilise les prédictions de la solution exacte
y_pred_exact = predire(surface, w_exact, b_exact)

# Calcul des métriques
mae = np.mean(np.abs(loyer - y_pred_exact))
rmse = np.sqrt(cout_mse(loyer, y_pred_exact))

sce = np.sum((loyer - y_pred_exact) ** 2)
sct = np.sum((loyer - np.mean(loyer)) ** 2)
r2 = 1 - (sce / sct)

print(f"MAE  : {mae:.2f} €")
print(f"RMSE : {rmse:.2f} €")
print(f"R²   : {r2:.4f}")


from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
print(f"\nVérification scikit-learn :")
print(f"MAE  : {mean_absolute_error(loyer, y_pred_exact):.2f} €")
print(f"RMSE : {root_mean_squared_error(loyer, y_pred_exact):.2f} €")
print(f"R²   : {r2_score(loyer, y_pred_exact):.4f}")