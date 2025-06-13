Projet : Application Streamlit de Prédiction des Dépenses Black Friday

Ce projet présente une application interactive développée avec Streamlit, conçue pour prédire le montant des dépenses des clients lors du Black Friday. Il intègre un pipeline complet de Data Science, de l'exploration des données à la modélisation prédictive, en passant par le prétraitement des données.

Fonctionnalités de l'ApplicationL'application Streamlit est structurée en plusieurs sections :
1.Accueil : Présentation générale du projet, de ses objectifs et de son intérêt stratégique pour les commerçants.
2.Exploration des Données : Permet de visualiser un aperçu des données brutes, d'afficher les informations clés (types de variables, valeurs manquantes, doublons) et d'explorer les distributions des variables catégorielles et les corrélations entre variables numériques.
3.Prétraitement : Décrit les étapes de nettoyage et de transformation des données (encodage des variables catégorielles, traitement de la colonne 'Age', gestion des valeurs manquantes, détection et suppression des outliers, suppression des doublons, séparation des variables explicatives et de la cible).
4.Modélisation : Présente les performances de différents modèles de régression (Régression Linéaire, Random Forest, Gradient Boosting, K-Nearest Neighbors) évalués sur le jeu de données. Un tableau comparatif des métriques (MSE, R²) est affiché, ainsi qu'une visualisation des prédictions vs. valeurs réelles pour la régression linéaire.
5.Prédictions : Offre la possibilité de réaliser des prédictions de dépenses de deux manières :
•Manuellement : L'utilisateur peut entrer les caractéristiques d'un client et obtenir une prédiction instantanée.
•Via un fichier CSV : L'utilisateur peut télécharger un fichier CSV contenant les données de plusieurs clients et obtenir un fichier CSV en retour avec les prédictions associées.Structure du ProjetLe projet est organisé comme suit :Copierblack_friday_modeling/
├── app.py
├── black_friday_cleaned.csv
├── train.csv
├── pred_vs_true_linreg.csv
├── pipeline_linear_regression.pkl
├── pipeline_random_forest.pkl
├── pipeline_gradient_boosting.pkl
├── pipeline_knn.pkl
└── images/
    ├── black_friday_sale.jpg
    └── customer_analysis.jpg

    •app.py : Le script principal de l'application Streamlit.
    •black_friday_cleaned.csv : Le jeu de données après les étapes de prétraitement.
    •train.csv : Le jeu de données original utilisé pour l'exploration.
    •pred_vs_true_linreg.csv : Fichier contenant les prédictions et les vraies valeurs pour la régression linéaire, utilisé pour la visualisation.
    •pipeline_*.pkl : Fichiers sérialisés (joblib) contenant les modèles de régression entraînés (Linear Regression, Random Forest, Gradient Boosting, KNN).
    •images/ : Dossier contenant les images utilisées dans l'interface de l'application.Comment Exécuter l'ApplicationPour exécuter cette application Streamlit localement, suivez les étapes ci-dessous :

    1.Cloner le dépôt GitHub (si ce n'est pas déjà fait) :Copier git clone <URL_DE_VOTRE_DEPOT>
cd black_friday_modeling

2.Créer un environnement virtuel :Copier python -m venv venv
source venv/bin/activate  # Sur Windows : .\venv\Scripts\activate

3.Installer les dépendances :pip install -r requirements.txt

4.Lancer l'application Streamlit :streamlit run app.pyL'application s'ouvrira automatiquement dans votre navigateur web par défaut.DépendancesLes principales bibliothèques Python utilisées dans ce projet sont listées dans le fichier requirements.txt.Auteur[Votre Nom/Pseudo GitHub]