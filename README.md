# Football Match Predictor

Une application Streamlit pour prédire l'issue d'un match de football ainsi que le nombre total de buts, basée sur des modèles de Machine Learning. Les données utilisées sont essentiellement les résultats des matchs de Premier League de la saison 2024-2025. 
## Fonctionnalités

* Prédiction de l'issue du match (victoire domicile, match nul, victoire visiteur).
* Prédiction du nombre total de buts marqués dans le match.
* Affichage des probabilités par issue sous forme de barplot.
* Indication de l’équipe gagnante pour plus de fun.

---

## Capture d’écran

<img width="530" height="531" alt="Annotation 2025-09-12 145922" src="https://github.com/user-attachments/assets/2bb96fab-86a2-419d-9c74-cca56b0d8c6e" />

---

## Technologies utilisées

* Python 3.13
* Streamlit
* Pandas, NumPy
* Scikit-learn (Logistic Regression, Pipeline, ColumnTransformer, etc.)
* XGBoost, AdaBoost, GradientBoosting, RandomForest
* Matplotlib, Seaborn
* Joblib pour la sérialisation des modèles
  
## Modèles ML

* **Classification de l’issue :** Logistic Regression avec `Pipeline` et `OneHotEncoder`.
* **Régression du nombre de buts :** AdaBoostRegressor (choisi après comparaison des performances).

## Améliorations 
- Données : Ajouter plus de matchs et de variables (forme récente, classement, blessures, stats avancées).
- Modèles : Tester d'autres modèles et faire de l’hyperparameter tuning.
- Préprocessing : Encoder intelligemment les variables catégorielles et gérer les valeurs manquantes.
- Interface : Afficher l’historique des matchs, graphiques de probabilités et prédictions batch.
- Déploiement : Cacher les calculs répétitifs, automatiser la mise à jour des modèles et envisager un API.

## 👨‍💻 Auteur
Camille Boris FAGBEDJI
Master en Sciences Agronomiques – Université de Parakou (Bénin)
Spécialisation en **ingénierie des eaux et sols, télédétection et modélisation écohydrologique.**
