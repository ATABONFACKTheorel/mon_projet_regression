import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import joblib
import time
import io
import seaborn as sns

# Chargement des données nettoyées
df = pd.read_csv("black_friday_cleaned.csv")

# Configuration globale de la page
st.set_page_config(page_title="Black Friday Modeling", layout="wide")

# Définition du menu de navigation
page = st.sidebar.selectbox(
    "Navigation",
    ("Accueil", "Exploration des données", "Prétraitement", "Modélisation", "Prédictions")
)

# Chemins vers les images (depuis le dossier principal)
img_dir = "images"
image1_path = os.path.join(img_dir, "black_friday_sale.jpg")
image2_path = os.path.join(img_dir, "customer_analysis.jpg")

# PAGE D'ACCUEIL
if page == "Accueil":
    st.title("Black Friday Modeling")
    st.markdown("---")

    # Présentation du projet
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("1. Présentation du Projet")
        st.write("""
        Ce projet de Data Science a pour objectif de prédire le montant des dépenses des clients d’un magasin lors du Black Friday.
        Grâce à des données clients, produits et comportementales, nous avons conçu un modèle prédictif robuste 
        afin d'aider les enseignes à mieux anticiper les ventes.
        """)
    with col2:
        if os.path.exists(image1_path):
            st.image(image1_path, use_container_width=True)
        else:
            st.warning("L'image black_friday_sale.jpg est introuvable dans le dossier 'images'.")

    st.markdown("---")

    # Intérêt du projet
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("2. Pourquoi ce projet ?")
        st.write("""
        Le Black Friday est un événement stratégique pour les commerçants.
        Comprendre les profils des clients les plus dépensiers, identifier les produits clés
        et anticiper les comportements d’achat permet d’optimiser les stratégies de vente et de stock.
        """)
    with col2:
        if os.path.exists(image2_path):
            st.image(image2_path, use_container_width=True)
        else:
            st.warning("L'image customer_analysis.jpg est introuvable dans le dossier 'images'.")

    st.markdown("---")

    # Objectifs
    st.header("3. Objectifs")
    st.write("""
    - Identifier les facteurs clés qui influencent les dépenses des clients.
    - Construire un modèle de prédiction fiable.
    - Déployer ce modèle dans une application accessible aux décideurs via une interface interactive.
    """)

elif page == "Exploration des données":
    st.title("Exploration des Données")

    # Chargement des données
    df = pd.read_csv("train.csv")

    st.subheader("Aperçu général")
    if st.checkbox("Afficher les premières lignes du dataset"):
        st.dataframe(df.head())

    if st.checkbox("Afficher les dimensions du dataset"):
        st.write(f"Nombre de lignes : {df.shape[0]}")
        st.write(f"Nombre de colonnes : {df.shape[1]}")

    if st.checkbox("Afficher les types de variables"):
        st.write(df.dtypes)

    if st.checkbox("Afficher les valeurs manquantes"):
        st.write(df.isnull().sum())

    if st.checkbox("Afficher les doublons"):
        duplicated_rows = df.duplicated().sum()
        st.write(f"Nombre de doublons : {duplicated_rows}")

    if st.checkbox("Afficher la description statistique"):
        st.write(df.describe(include='all'))

    st.subheader("Visualisation des variables catégorielles")

    # Liste des colonnes à visualiser
    plot_cols_sorted = ["Occupation", "Product_Category_1", "Product_Category_2", "Product_Category_3"]
    plot_cols_simple = ["Gender", "Marital_Status"]

    for col in plot_cols_sorted:
        st.markdown(f"{col} (distribution décroissante)")
        counts = df[col].value_counts().sort_values(ascending=False)
        fig = px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            labels={'x': col, 'y': 'Effectif'},
            title=f"Distribution de {col}",
            color=counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

    for col in plot_cols_simple:
        st.markdown(f"{col}")
        counts = df[col].value_counts()
        fig = px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            labels={'x': col, 'y': 'Effectif'},
            title=f"Distribution de {col}",
            color=counts.values,
            color_continuous_scale='Purples'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Sélection uniquement des colonnes numériques
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'int32', 'float32']).columns

    # Calcul de la matrice de corrélation
    corr_matrix = df[numeric_cols].corr()

    
    fig = px.imshow(
    corr_matrix,
    text_auto=True,  
    color_continuous_scale='RdBu_r',  
    aspect='auto',  
    )

    fig.update_layout(
    title="Matrice de corrélation",
    xaxis_title="Variables",
    yaxis_title="Variables",
    width=800,
    height=800
    )

    
    st.plotly_chart(fig)    
elif page == "Prétraitement":
    st.markdown("## Étape 3 : Préparation des Données")
    st.write(
        """
        Cette étape a consisté à nettoyer et transformer les données brutes afin de les rendre exploitables pour la modélisation.
        Elle est cruciale pour garantir la qualité des résultats.
        """
    )

    with st.expander("1. Encodage des variables catégorielles"):
        st.write("Les variables suivantes ont été encodées à l'aide du One Hot Encoding (avec drop_first=True pour éviter la multicolinéarité) :")
        st.markdown("- Gender")
        st.markdown("- Marital_Status")
        st.markdown("- Occupation (encodée sans regroupement)")
        st.markdown("- Product_Category_1, Product_Category_2, Product_Category_3")
        st.code("df = pd.get_dummies(df, columns=['Gender', 'Marital_Status', ...], drop_first=True, dtype=int)")

    with st.expander("2. Traitement de la colonne 'Age'"):
        st.write("Les classes d'âge ont été remplacées par leurs centres, puis standardisées.")
        st.code("""
# Remplacement par centre
age_map = {'0-17': 8, '18-25': 21, '26-35': 30, '36-45': 40, '46-50': 48,
           '51-55': 53, '55+': 60}
df['Age'] = df['Age'].map(age_map)

# Standardisation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Age'] = scaler.fit_transform(df[['Age']])
        """)

    with st.expander("3. Traitement des valeurs manquantes"):
        st.write("Les colonnes Product_Category_2 et Product_Category_3 contenaient des valeurs manquantes (notées -1). Elles ont été conservées lors de la première itération pour permettre au modèle de les interpréter.")
        st.code("# Pas de suppression des -1 pour Product_Category_2 et 3 (itération 1)")

    with st.expander("4. Détection et suppression des valeurs aberrantes"):
        st.write("Les outliers dans les colonnes Product_Category_1, 2 et 3 ont été détectés à l'aide des IQR et supprimés.")
        st.code("""
def remove_outliers(df, column):
    q1 = df[column][df[column] != -1].quantile(0.25)
    q3 = df[column][df[column] != -1].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[column] == -1) | ((df[column] >= lower) & (df[column] <= upper))]

df = remove_outliers(df, 'Product_Category_1')
df = remove_outliers(df, 'Product_Category_2')
df = remove_outliers(df, 'Product_Category_3')
        """)

    with st.expander("5. Suppression des doublons"):
        st.write("Les doublons ont été supprimés du dataset.")
        st.code("""
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
st.write(f"{before - after} doublons supprimés.")
        """)

    with st.expander("6. Séparation des variables explicatives et de la cible"):
        st.write("La variable cible est Purchase, les autres sont utilisées comme prédicteurs.")
        st.code("""
X = df.drop("Purchase", axis=1)
y = df["Purchase"]
        """)

    if st.button("Afficher un aperçu du dataset préparé"):
        st.dataframe(df.head())

    st.success("Les données sont prêtes pour la modélisation !")

elif page == "Modélisation":
    st.title("Modélisation des Dépenses - Black Friday")

    st.markdown("### 1. Sélection du Modèle à Visualiser")
    model_choice = st.selectbox(
        "Choisissez un modèle pour afficher ses performances :",
        ("Régression Linéaire", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors")
    )

    if model_choice == "Régression Linéaire":
        st.markdown("Scores obtenus :")
        st.write("- MSE : 7357426.8338")
        st.write("- R² : 0.7084")
    elif model_choice == "Random Forest":
        st.markdown("Scores obtenus :")
        st.write("- MSE : 7011793.4719")
        st.write("- R² : 0.7221")
    elif model_choice == "Gradient Boosting":
        st.markdown("Scores obtenus :")
        st.write("- MSE : 7,080,000")
        st.write("- R² : 0.7234")
    elif model_choice == "K-Nearest Neighbors":
        st.markdown("Scores obtenus :")
        st.write("- MSE : 8563326.6003")
        st.write("- R² : 0.6606")

    st.markdown("### 2. Comparaison des Modèles")
    st.write("Ci-dessous, un tableau comparatif des modèles évalués :")

    scores_df = pd.DataFrame({
        "Modèle": ["Régression Linéaire", "Random Forest", "Gradient Boosting", "KNN"],
        "MSE": [7357426.8338, 7011793.4719, 7080000, 8563326.6003],
        "R²": [0.7084, 0.7221, 0.7234, 0.6606]
    })

    st.dataframe(scores_df)

    st.markdown("### 3. Hyperparamètres optimaux")
    st.write("Les modèles non linéaires ont été optimisés via GridSearchCV.")
    st.code("""
# Exemple pour Random Forest :
{'max_depth': 15, 'min_samples_split': 5, 'n_estimators': 200}

# Exemple pour Gradient Boosting :
{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}

# Exemple pour KNN :
{'n_neighbors': 7, 'weights': 'distance'}
    """)

    st.markdown("### 4. Visualisation - Prédictions vs Valeurs Réelles (Régression Linéaire)")
    st.write("Ce graphique compare les valeurs prédites aux vraies valeurs (jeu de test) pour la régression linéaire.")

    try:
        df_pred = pd.read_csv("pred_vs_true_linreg.csv")
        fig = px.scatter(df_pred, x="y_test", y="y_pred_lr", opacity=0.5,
                         labels={"y_test": "Valeurs Réelles", "y_pred_linreg": "Prédictions"},
                         title="Régression Linéaire : Valeurs Réelles vs Prédictions")
        fig.add_shape(type="line", x0=df_pred["y_test"].min(), y0=df_pred["y_test"].min(),
                      x1=df_pred["y_test"].max(), y1=df_pred["y_test"].max(),
                      line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.error("Le fichier pred_vs_true_linreg.csv est introuvable. Assurez-vous qu’il est bien dans le dossier principal.")

elif page == "Prédictions":
    st.title("Prédiction des dépenses - Black Friday")

    # 1. Choix du modèle
    st.markdown("## 1. Sélectionner le modèle")
    model_files = {
        "Régression Linéaire": "pipeline_linear_regression.pkl",
        "Random Forest": "pipeline_random_forest.pkl",
        "Gradient Boosting": "pipeline_gradient_boosting.pkl",
        "KNN": "pipeline_knn.pkl"
    }
    model_choice = st.selectbox("Choisissez un modèle", list(model_files.keys()))
    model_path = model_files[model_choice]
    model = joblib.load(model_path)

    # 2. Prédiction manuelle
    st.markdown("## 2. Prédiction sur un client")
    with st.expander("Entrer les données manuellement"):

        # Sélections de l'utilisateur
        gender = st.selectbox("Genre", ["F", "M"])
        age = st.selectbox("Tranche d'âge", ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'])
        city = st.selectbox("Catégorie de ville", ['A', 'B', 'C'])
        stay = st.selectbox("Années dans la ville actuelle", ['0', '1', '2', '3', '4+'])
        occupation = st.selectbox("Occupation", list(range(21)))
        marital_status = st.selectbox("Statut marital", [0, 1])
        pc1 = st.selectbox("Catégorie Produit 1", list(range(1, 21)))
        pc2 = st.selectbox("Catégorie Produit 2", [-1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        pc3 = st.selectbox("Catégorie Produit 3", [-1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

        # Mapping des valeurs
        gender = 0 if gender == 'F' else 1
        age_mapping = {'0-17': 16, '18-25': 21, '26-35': 30, '36-45': 40,
                       '46-50': 48, '51-55': 53, '55+': 60}
        age = age_mapping[age]

        stay_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4+': 4}
        stay = stay_mapping[stay]

        # Construction d'un DataFrame
        user_input = pd.DataFrame({
            "Gender": [gender],
            "Age": [age],
            "City_Category": [city],
            "Stay_In_Current_City_Years": [stay],
            "Occupation": [occupation],
            "Marital_Status": [marital_status],
            "Product_Category_1": [pc1],
            "Product_Category_2": [pc2],
            "Product_Category_3": [pc3]
        })

        if st.button("Prédire"):
            with st.spinner("Prédiction en cours..."):
                time.sleep(1)
                prediction = model.predict(user_input)[0]
            st.success(f"Dépense prédite : {round(prediction, 2)}")

    # 3. Prédiction par fichier CSV
    st.markdown("## 3. Prédiction à partir d'un fichier CSV")
                     # Génération d'un fichier CSV exemple en mémoire
    def generate_sample_csv():
        data = {
            "Gender": np.random.choice(["F", "M"], size=100),
            "Age": np.random.choice(['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'], size=100),
            "City_Category": np.random.choice(['A', 'B', 'C'], size=100),
            "Stay_In_Current_City_Years": np.random.choice(['0', '1', '2', '3', '4+'], size=100),
            "Occupation": np.random.randint(0, 21, size=100),
            "Marital_Status": np.random.choice([0, 1], size=100),
            "Product_Category_1": np.random.choice(list(range(1, 21)), size=100),
            "Product_Category_2": np.random.choice([-1] + list(range(2, 19)), size=100),
            "Product_Category_3": np.random.choice([-1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], size=100)
        }
        df = pd.DataFrame(data)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue()

    # Proposer le fichier exemple à télécharger
    st.download_button(
        label="Télécharger un exemple de fichier CSV",
        data=generate_sample_csv(),
        file_name="exemple_100_clients.csv",
        mime="text/csv"
    )
    uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
# Affichage d'une animation de chargement
            with st.spinner("Préparation des données et génération des prédictions..."):
                time.sleep(1.5)

                # Mapping pour l'âge
                age_map = {
                    '0-17': 8,
                    '18-25': 21.5,
                    '26-35': 30.5,
                    '36-45': 40.5,
                    '46-50': 48,
                    '51-55': 53,
                    '55+': 60
                }

                df_input["Age"] = df_input["Age"].map(age_map)

                # Remplacer les valeurs manquantes par -1 pour les colonnes produit
                for col in ["Product_Category_2", "Product_Category_3"]:
                    df_input[col] = df_input[col].fillna(-1)

                # Forcer les types si besoin (par sécurité)
                df_input["Marital_Status"] = df_input["Marital_Status"].astype(int)
                df_input["Occupation"] = df_input["Occupation"].astype(int)

                # Prédiction
                preds = model.predict(df_input)
                df_input["Predicted_Purchase"] = preds

            st.success("Prédiction terminée avec succès !")
            st.dataframe(df_input)

            # Télécharger le fichier
            csv = df_input.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger les résultats en CSV",
                data=csv,
                file_name="predictions_clients.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Une erreur est survenue lors de la lecture du fichier : {e}")            
