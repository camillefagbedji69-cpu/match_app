# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Charger les modèles sauvegardés ---
pipeline_classif = joblib.load('model_match_prediction.pkl')
pipeline_regression = joblib.load('best_regression_model.pkl')

st.title("Prédiction de l'issue d'un match")

# --- Charger la base ---
data = pd.read_csv("Foot_data.csv", sep=';')

# --- Colonnes catégorielles pour le pipeline ---
cols_cat = data.select_dtypes(include=["object", "category"]).columns.tolist()

# --- Extraire les valeurs uniques pour les selectbox ---
equipes = pd.unique(data[['HomeTeam', 'AwayTeam']].values.ravel())
arbitres = data['Referee'].unique()

# --- Sélection utilisateur ---
home_team = st.selectbox("Équipe à domicile", equipes)
away_team = st.selectbox("Équipe visiteuse", equipes)
arbitre = st.selectbox("Arbitre du match", arbitres)

# --- Créer DataFrame pour le modèle ---
input_df = pd.DataFrame({
    'HomeTeam': [home_team],
    'AwayTeam': [away_team],
    'Referee': [arbitre]
})

# Ajouter les colonnes manquantes avec une valeur par défaut
for col in cols_cat:
    if col not in input_df.columns:
        input_df[col] = 'Unknown'

# S'assurer que toutes les colonnes sont du type string
input_df = input_df.astype(str)

# --- Bouton pour prédiction de l'issue ---
if st.button("Prédire le résultat"):
    prediction = pipeline_classif.predict(input_df)
    proba = pipeline_classif.predict_proba(input_df)[0]

    # Décodage si LabelEncoder utilisé
    mapping = {0:'A', 1:'D', 2:'H'}
    pred_label = mapping[prediction[0]]

    st.write(f"**Résultat prédit : {pred_label}**")

    # DataFrame pour le barplot
    proba_df = pd.DataFrame({
        'Résultat': ['A', 'D', 'H'],
        'Probabilité': proba
    })

    st.write("**Probabilités par issue :**")
    st.dataframe(proba_df)

    # --- Barplot ---
    fig, ax = plt.subplots()
    sns.barplot(x='Résultat', y='Probabilité', data=proba_df, palette='viridis', ax=ax)
    ax.set_ylim(0,1)
    for i, v in enumerate(proba_df['Probabilité']):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    st.pyplot(fig)

# --- Bouton pour prédiction du nombre de buts ---
if st.button("Prédire le nombre de buts"):
    buts_predits = pipeline_regression.predict(input_df)
    st.write(f"**Nombre total de buts prédit : {buts_predits[0]:.0f}**")
