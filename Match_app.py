# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Charger les modèles sauvegardés ---
model_classif = joblib.load('model_match_prediction.pkl')   # modèle classification
model_regress = joblib.load('best_regression_model.pkl')    # modèle régression

st.title("Prédiction de l'issue d'un match et du nombre total de buts")

# --- Charger la base de données ---
data = pd.read_csv("Foot_data.csv", sep=';')

# --- Créer colonne total_goals pour vérifier la cible régression ---
data['total_goals'] = data['FTHG'] + data['FTAG']
data.rename(columns={
    'HomeTeam': 'equipe_home',
    'AwayTeam': 'equipe_away',
    'Referee': 'arbitre'
}, inplace=True)

# ensuite tu peux faire
equipes = pd.unique(data[['equipe_home','equipe_away']].values.ravel())
arbitres = data['arbitre'].unique()

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

# --- Bouton pour prédiction de l'issue ---
if st.button("Prédire le résultat"):
    # Classification
    prediction = model_classif.predict(input_df)
    proba = model_classif.predict_proba(input_df)[0]

    # Décodage si LabelEncoder utilisé
    mapping = {0:'A', 1:'D', 2:'H'}
    pred_label = mapping[prediction[0]]
    st.write(f"**Résultat prédit : {pred_label}**")

    # DataFrame pour barplot
    proba_df = pd.DataFrame({
        'Résultat': ['A','D','H'],
        'Probabilité': proba
    })

    # Afficher le DataFrame
    st.write("**Probabilités par issue :**")
    st.dataframe(proba_df)

    # Barplot des probabilités
    fig, ax = plt.subplots()
    sns.barplot(x='Résultat', y='Probabilité', data=proba_df, palette='viridis', ax=ax)
    ax.set_ylim(0,1)
    for i, v in enumerate(proba_df['Probabilité']):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    st.pyplot(fig)

# --- Bouton pour prédiction du nombre total de buts ---
if st.button("Prédire le nombre de buts"):
    but_predit = model_regress.predict(input_df)[0]  # récupère la valeur
    st.write(f"**Nombre total de buts prédits : {but_predit:.0f}**")

