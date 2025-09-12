# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Charger les pipelines sauvegardés ---
pipeline_classif = joblib.load('model_match_prediction.pkl')  # pipeline classification
pipeline_regress = joblib.load('best_regression_model.pkl')   # pipeline regression

st.title("Prédiction des matchs de football")

# --- Charger la base pour récupérer les équipes et arbitres ---
data = pd.read_csv("Foot_data.csv", sep=";")

equipes = pd.unique(data[['HomeTeam','AwayTeam']].values.ravel())
arbitres = data['Referee'].unique()

# --- Sélection utilisateur ---
home_team = st.selectbox("Équipe à domicile", equipes)
away_team = st.selectbox("Équipe visiteuse", equipes)
arbitre = st.selectbox("Arbitre du match", arbitres)

# --- Préparer le DataFrame d'entrée (nom exact des colonnes du pipeline) ---
input_df = pd.DataFrame({
    'HomeTeam': [home_team],
    'AwayTeam': [away_team],
    'Referee': [arbitre]
})

# --- Prédiction de l'issue du match ---
if st.button("Prédire le résultat"):
    prediction = pipeline_classif.predict(input_df)
    proba = pipeline_classif.predict_proba(input_df)[0]

    # Mapping fixe pour les labels
    mapping = {0:'A', 1:'D', 2:'H'}
    pred_label = mapping[prediction[0]]

    st.write(f"**Résultat prédit : {pred_label}**")

    # DataFrame pour le barplot
    proba_df = pd.DataFrame({
        'Résultat': ['A','D','H'],
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

# --- Prédiction du nombre total de buts ---
if st.button("Prédire le nombre de buts"):
    but_predit = pipeline_regress.predict(input_df)
    st.write(f"**Nombre total de buts prédits : {but_predit[0]:.0f}**")
