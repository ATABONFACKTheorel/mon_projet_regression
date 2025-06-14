Projet : Prédiction des Achats du Black Friday

Ce dépôt contient un notebook Jupyter qui analyse les données du Black Friday pour prédire le montant des achats. Le projet inclut une analyse exploratoire des données (EDA), un prétraitement des données et l'application de plusieurs modèles de régression pour la prédiction.

Table des Matières
•Description du Projet
•Jeu de Données
•Méthodologie
•Résultats
•Comment Exécuter le Notebook

Description du Projet
L'objectif principal de ce projet est de construire un modèle prédictif capable d'estimer le montant des achats (Purchase) effectués par les clients lors du Black Friday. Pour ce faire, nous avons exploré un ensemble de données contenant diverses informations sur les clients (âge, sexe, statut marital, occupation, catégorie de ville, années passées dans la ville actuelle) et les produits (catégories de produits).Le notebook black_friday_regression_model.ipynb détaille les étapes suivantes :1.Analyse Exploratoire des Données (EDA) : Comprendre la structure des données, identifier les distributions des variables, détecter les valeurs manquantes et les outliers, et visualiser les relations entre les variables.2.Prétraitement des Données : Nettoyer et transformer les données brutes en un format adapté à la modélisation. Cela inclut la gestion des valeurs manquantes, l'encodage des variables catégorielles et la normalisation des variables numériques.3.Modélisation : Entraîner et évaluer plusieurs modèles de régression, notamment la Régression Linéaire, K-Nearest Neighbors (KNN), Gradient Boosting et Random Forest, pour prédire le montant des achats.

Jeu de Données
Le jeu de données utilisé pour ce projet est train.csv. Il contient les colonnes suivantes:
•User_ID : Identifiant unique de l'utilisateur.
•Product_ID : Identifiant unique du produit.
•Gender : Sexe de l'utilisateur (M/F).
•Age : Catégorie d'âge de l'utilisateur.
•Occupation : Occupation de l'utilisateur.
•City_Category : Catégorie de la ville (A, B, C).
•Stay_In_Current_City_Years : Nombre d'années passées dans la ville actuelle.•Marital_Status : Statut marital de l'utilisateur (0 pour célibataire, 1 pour marié).•Product_Category_1 : Catégorie de produit principale.
•Product_Category_2 : Catégorie de produit secondaire (peut contenir des valeurs manquantes).
•Product_Category_3 : Catégorie de produit tertiaire (peut contenir des valeurs manquantes).•Purchase : Montant de l'achat (variable cible).

Méthodologie
Le processus d'analyse et de modélisation a suivi les étapes clés suivantes :
1.Importation des Bibliothèques : Chargement des packages Python nécessaires (pandas, numpy, matplotlib, seaborn, scikit-learn).

2.Chargement des Données : Lecture du fichier train.csv dans un DataFrame pandas.

3.Analyse Exploratoire des Données (EDA) :
•Vérification des dimensions du dataset (.shape).
•Examen des types de données et des valeurs non nulles (.info()).
•Identification et quantification des valeurs manquantes (.isna().sum()).
•Analyse des distributions des variables catégorielles (Gender, Age, City_Category, Stay_In_Current_City_Years, Occupation, Marital_Status, Product_Category_1, Product_Category_2, Product_Category_3) à l'aide de countplot.
•Visualisation des variables numériques (Purchase et autres catégories de produits après transformation) à l'aide de boxplots pour détecter les outliers.
•Calcul et visualisation de la matrice de corrélation entre les variables numériques pour comprendre leurs relations.

4.Prétraitement des Données :
•Gestion des valeurs manquantes : Les valeurs manquantes dans Product_Category_2 et Product_Category_3 ont été remplacées par -1 pour les traiter comme une catégorie distincte, puis converties en type entier.
•Transformation de Age : Les catégories d'âge (ex: '0-17', '18-25') ont été mappées à des valeurs numériques représentant le centre de chaque intervalle.
•Encodage One-Hot : Les variables catégorielles restantes (Stay_In_Current_City_Years, Marital_Status, Gender, City_Category, Occupation, Product_Category_1, Product_Category_2, Product_Category_3) ont été encodées en utilisant l'encodage One-Hot (pd.get_dummies) avec drop_first=True pour éviter la multicolinéarité.
•Normalisation de Age : La colonne Age a été normalisée à l'aide de StandardScaler.•Suppression des colonnes originales : Les colonnes originales qui ont été transformées ou encodées, ainsi que les identifiants (User_ID, Product_ID), ont été supprimées du DataFrame.

5.Modélisation :
•Séparation des données : Le dataset a été divisé en ensembles d'entraînement (80%) et de test (20%) en utilisant train_test_split.
•Entraînement et Évaluation des Modèles : Les modèles suivants ont été entraînés et évalués sur les métriques MSE (Mean Squared Error) et R² (Coefficient de Détermination) :
•Régression Linéaire (LinearRegression)
•K-Nearest Neighbors Regressor (KNeighborsRegressor)
•Gradient Boosting Regressor (GradientBoostingRegressor)
•Random Forest Regressor (RandomForestRegressor)RésultatsLes performances de chaque modèle sont affichées dans le notebook. Le modèle le plus performant en termes de MSE et R² est identifié, offrant une bonne base pour la prédiction des montants d'achat. (Les résultats spécifiques seront visibles dans le notebook après exécution).

Comment Exécuter le Notebook
Pour exécuter ce notebook, suivez les étapes ci-dessous :
1.Cloner le dépôt GitHub :Copier git clone <https://github.com/ATABONFACKTheorel/mon_projet_regression.git>
cd <mon_projet_regression>
2.Installer les dépendances :
pip install -r requirements.txt

3.Télécharger le jeu de données :
Le fichier train.csv doit être placé dans le même répertoire que le notebook.

4.Ouvrir et exécuter le notebook :
Vous pouvez ouvrir le notebook avec Jupyter Notebook ou JupyterLab :jupyter notebook black_friday_regression_model.ipynb Ou avec Google Colab en l'important directement.Exécutez toutes les cellules du notebook séquentiellement pour reproduire l'analyse et les résultats.

## Application Déployée

Découvrez l'application interactive Streamlit de ce projet, qui permet de [ visualiser les prédictions, tester le modèle avec de nouvelles données].

**Accéder à l'application ici :** [Lien vers l'application Streamlit](https://monprojetregression-meghkousv9ahe5qsbeipgt.streamlit.app/)
