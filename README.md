Projet P7
- Modéliser la prédiction du score de défaut de paiement et tracker l'évaluation des performances via des expériences MLFlow 
- Réaliser et déployer le modèle retenu via une API dans le cloud (FastAPI, Heroku)
- Réaliser et déployer 1 application dashboard (Python, Streamlit) destinée au conseiller bancaire pour l'analyse de la relation client et l'interprétation de la prédiction du score.
- Analyse du data drift (dérive dans les données)

ARBORESCENCE DES DOSSIERS

"modelisation"
- 3 note_books : EDA, Modeling, Data_Drift
- tableaux HTML d'analyse du data drift
- liste des packages python des notebooks : requirements.txt

"api"
- script de l'api : api.py (Python, FastAPI)
- modèle LGBM retenu au format binaire (pikle) : lgbm_model_final.p
- fichiers de configuration pour le déploiement dans Heroku : Procfile, runtime.txt
- packages python : requirements.txt, utilisé pour l'intégration continue et le déploiement 
        - sous dossier "api/.github/workflows" : python-app.yml pour automatisation des tests unitaires 
        - sous dossier "api/tests" : scripts des tests unitaires

"dashboard"
- script du dashboard : appli.py (Python, Streamlit)
- data source : df_appli_test.csv
- 50 features importance des modèles LGBM et LOGISTIC : df_lgbm_feat_importance.csv, df_log_feat_importance.csv
- fichiers binaires pour l'interprétaion SHAP locale via modèle LGBM et LOGISTIC : lgbm_explainer.p, lgbm_shap_values.p, log_shap_values.p
- images de l'interprétation SHAP globale : lgbm_summary_plot.png, log_summary_plot.png
- fichiers de configuration pour le déploiement du dashboard dans Heroku : Procfile, runtime.txt, setup.sh, requirements.txt
