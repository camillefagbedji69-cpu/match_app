# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Charger les modèles sauvegardés ---
model_classif = joblib.load('model_match_prediction.pkl')
model_but = joblib.load('best_regression_model.pkl')

st.title("Prédiction des matchs de football")

# --- Charger la base de données ---
data = pd.read_csv("Foot_data.csv")

# --- Extraire les valeurs uniques pour les selectbox ---
equipes = pd.unique(data[['HomeTeam','AwayTeam']].values.ravel())
arbitres = data['Referee'].unique()

# --- Sélection utilisateur ---
home_team = st.selectbox("Équipe à domicile", equipes)
away_team = st.selectbox("Équipe visiteuse", equipes)
arbitre = st.selectbox("Arbitre du match", arbitres)

# --- Créer DataFrame pour le modèle avec les colonnes exactes du pipeline ---
input_df = pd.DataFrame({
    'HomeTeam': [home_team],
    'AwayTeam': [away_team],
    'Referee': [arbitre]
})

# --- Bouton pour prédiction de l'issue du match ---
if st.button("Prédire le résultat"):
    # Prédiction classification
    prediction = model_classif.predict(input_df)
    proba = model_classif.predict_proba(input_df)[0]  # première ligne

    # Décodage si LabelEncoder utilisé
    mapping = {0:'A', 1:'D', 2:'H'}
    pred_label = mapping[prediction[0]]

    st.write(f"**Résultat prédit : {pred_label}**")

    # --- Barplot des probabilités ---
    proba_df = pd.DataFrame({
        'Résultat': ['A','D','H'],
        'Probabilité': proba
    })

    st.write("**Probabilités par issue :**")
    st.dataframe(proba_df)

    fig, ax = plt.subplots()
    sns.barplot(x='Résultat', y='Probabilité', data=proba_df, palette='viridis', ax=ax)
    ax.set_ylim(0,1)
    for i, v in enumerate(proba_df['Probabilité']):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    st.pyplot(fig)

# --- Bouton pour prédiction du nombre total de buts ---
if st.button("Prédire le nombre de buts"):
    but_predit = model_but.predict(input_df)
    st.write(f"**Nombre total de buts prédits : {but_predit[0]:.1f}**")
