# 🛒 Prédiction du Chiffre d'Affaires (Sales_Amount) — Déploiement ML avec Gradio

## 📌 Description du projet

Ce projet implémente un pipeline complet de Machine Learning pour résoudre un problème de **régression** : prédire le **chiffre d'affaires (`Sales_Amount`)** d'une commande à partir de ses caractéristiques.

L'interface de prédiction est déployée sur **Hugging Face Spaces** via **Gradio**, permettant à n'importe quel utilisateur de saisir des données et d'obtenir une prédiction instantanée.

---

## 🎯 Problématique

> **Objectif :** Prédire le montant des ventes (`Sales_Amount`) d'une commande à partir de ses caractéristiques.

| Rôle | Variable |
|------|----------|
| Variable cible `y` | `Sales_Amount` |
| Features `X` | `Units_Sold`, `Unit_Price`, `Discount_%`, `Segment`, `Product_Category`, `City` |

---

## 🗂️ Structure du projet

```
├── app.py                  # Interface Gradio (déploiement)
├── notebook.ipynb          # Pipeline ML complet (phases 1 à 5)
├── model.pkl               # Modèle entraîné (exporté avec joblib/pickle)
├── requirements.txt        # Dépendances Python
└── README.md               # Documentation du projet
```

---

## 🔄 Pipeline ML — 6 phases

### Phase 1 — Collecte & Exploration

- Téléchargement et chargement du dataset
- Exploration initiale : nombre de lignes, colonnes, types de variables
- Analyse de la distribution de la variable cible `Sales_Amount`

```python
df.shape          # dimensions du dataset
df.dtypes         # types de variables
df['Sales_Amount'].describe()   # statistiques descriptives de la cible
```

---

### Phase 2 — Visualisation

Au moins 3 visualisations pertinentes produites pour comprendre les données :

1. **Distribution de `Sales_Amount`** — histogramme ou boxplot
2. **Corrélation entre les features numériques** — heatmap
3. **`Sales_Amount` par `Product_Category` ou `Segment`** — barplot

---

### Phase 3 — Nettoyage & Préparation

- Détection et traitement des **valeurs manquantes**
- Suppression des **doublons**
- Suppression des variables non pertinentes
- Détection et traitement des **outliers**
- **Encodage** des variables catégorielles (`Segment`, `Product_Category`, `City`) via `LabelEncoder` ou `OneHotEncoder`

---

### Phase 4 — Normalisation & Séparation

- Séparation des features `X` et de la cible `y`
- Division en ensembles d'**entraînement** et de **test** (`train_test_split`)
- **Normalisation** des features numériques (`StandardScaler` ou `MinMaxScaler`)

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### Phase 5 — Modélisation & Évaluation

Entraînement d'au minimum **deux modèles de régression** :

| Modèle | Description |
|--------|-------------|
| `LinearRegression` | Régression linéaire de base |
| `RandomForestRegressor` | Ensemble d'arbres de décision |

**Métriques d'évaluation :**

| Métrique | Description |
|----------|-------------|
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |
| **R²** | Coefficient de détermination |

```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
```

---

### Phase 6 — Déploiement Gradio

Une interface interactive construite avec **Gradio** dans `app.py` permet à l'utilisateur de :

- Saisir les valeurs : `Units_Sold`, `Unit_Price`, `Discount_%`, `Segment`, `Product_Category`, `City`
- Obtenir en retour le **`Sales_Amount` prédit**

```python
import gradio as gr

def predict(units_sold, unit_price, discount, segment, product_category, city):
    # Prétraitement & prédiction
    ...
    return predicted_sales_amount

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Units Sold"),
        gr.Number(label="Unit Price"),
        gr.Number(label="Discount (%)"),
        gr.Dropdown(label="Segment", choices=[...]),
        gr.Dropdown(label="Product Category", choices=[...]),
        gr.Dropdown(label="City", choices=[...]),
    ],
    outputs=gr.Number(label="Predicted Sales Amount"),
    title="Prédiction du Chiffre d'Affaires",
)

interface.launch()
```

---

## 🚀 Lancement local

```bash
# Cloner le dépôt
git clone https://github.com/mohamadouhayatouabbassi-glitch/D-ploiement-Mod-le-ML-Gradio-Pr-diction-du-CA.git
cd D-ploiement-Mod-le-ML-Gradio-Pr-diction-du-CA

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'interface Gradio
python app.py
```

---

## 🌐 Déploiement sur Hugging Face Spaces

Le projet est déployé sur **Hugging Face Spaces** :

👉 [Accéder à l'application](https://huggingface.co/spaces/mohamadouhayatouabbassi-glitch/D-ploiement-Mod-le-ML-Gradio-Pr-diction-du-CA)

---

## 🛠️ Technologies utilisées

| Outil | Usage |
|-------|-------|
| `pandas` | Manipulation des données |
| `numpy` | Calculs numériques |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | Modélisation ML |
| `gradio` | Interface utilisateur |
| `joblib` | Sérialisation du modèle |
| Hugging Face Spaces | Déploiement cloud |

---

## 👤 Auteur

**Mohamad Ouhayatou Abbassi**  
Projet de déploiement de modèle ML — Régression sur données de ventes