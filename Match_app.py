# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Charger le modèle sauvegardé ---
model = joblib.load('model_match_prediction.pkl')
model_but = joblib.load('best_regression_model.pkl')
st.title("Prédiction de l'issue d'un match")

# --- Charger la base ---
data = pd.read_csv("Foot_data.csv")

# --- Extraire les valeurs uniques pour les selectbox ---
equipes = pd.unique(data[['HomeTeam','AwayTeam']].values.ravel())
arbitres = data['arbitre'].unique()

## sélection 
home_team = st.selectbox("Équipe à domicile", equipes)
away_team = st.selectbox("Équipe visiteuse", equipes)
arbitre = st.selectbox("Arbitre du match", arbitres)

# --- Créer DataFrame pour le modèle ---
input_df = pd.DataFrame({
    'equipe_home': [home_team],
    'equipe_away': [away_team],
    'arbitre': [arbitre]
})

# --- Bouton pour prédiction ---
if st.button("Prédire le résultat"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0]  # obtenir la première ligne

    # Décodage si LabelEncoder utilisé
    mapping = {0:'A', 1:'D', 2:'H'}
    pred_label = mapping[prediction[0]]

    st.write(f"**Résultat prédit : {pred_label}**")

    # --- DataFrame pour le barplot ---
    proba_df = pd.DataFrame({
        'Résultat': ['A','D','H'],
        'Probabilité': proba
    })

    # Afficher le DataFrame
    st.write("**Probabilités par issue :**")
    st.dataframe(proba_df)

    # --- Barplot ---
    fig, ax = plt.subplots()
    sns.barplot(x='Résultat', y='Probabilité', data=proba_df, palette='viridis', ax=ax)
    ax.set_ylim(0,1)
    for i, v in enumerate(proba_df['Probabilité']):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    st.pyplot(fig)

if st.button("Prédire le nombre de buts"):
    but_predit = model_but.predict(input_df)
    st.write(f"**Nombre de buts prédits : {but_predit}**")
