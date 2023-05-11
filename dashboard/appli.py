#######################
# Import des packages #
#######################
# Built-in packages : format JSON pour échange de data via le web + requête URL 
# API de calcul du score de probabilité d'octroi selon le modèle choisi (Regression log / Light GBM)
import json
import requests
from PIL import Image

# Data manipulation
import numpy as np
import pandas as pd

# Data viz
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

# Import d'objets SHAPE à déserialiser pour l'analyse shape locale
import pickle # sérialisation / déserialisation
from shap import LinearExplainer
from shap import waterfall_plot, summary_plot
from shap.explainers import Tree as shap_Tree
from shap import plots as shap_plots
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

# OLD --> A SUPPRIMER (existe dans l'API)
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

#############
# Fonctions #
#############
# Données du TEST set pour l'application (échantillon de 100 individus avec 1 distribution de 50% par classe)
@st.cache_data
def fct_get_data():
    df_appli = pd.read_csv('df_appli_test.csv')
    df_appli.drop('Unnamed: 0', axis=1, inplace=True)
    return df_appli

# Extract des objets shape selon le modèle choisi pour analyse de l'impacte des features locales sur la target
@st.cache_data
def fct_get_shap_value():
    log_shap_values = pickle.load(open('log_shap_values.p', 'rb'))
    lgbm_explainer = pickle.load(open('lgbm_explainer.p', 'rb'))
    lgbm_shap_values = pickle.load(open('lgbm_shap_values.p', 'rb'))
    return log_shap_values, lgbm_explainer, lgbm_shap_values

# Liste des 50 Features importance dans l'ordre décroissant selon les modèles Log et LGBM 
@st.cache_data
def fct_get_feat_importance():
    df_log_feat_importance = pd.read_csv('df_log_feat_importance.csv')
    df_log_feat_importance.drop('Unnamed: 0', axis=1, inplace=True)
    df_lgbm_feat_importance = pd.read_csv('df_lgbm_feat_importance.csv')
    df_lgbm_feat_importance.drop('Unnamed: 0', axis=1, inplace=True)
    log_feat_importance = df_log_feat_importance['feat_importance'].values.tolist() # ordonnée
    lgbm_feat_importance = df_lgbm_feat_importance['feat_importance'].values.tolist() # ordonnée
    feats = df_lgbm_feat_importance['feat'].values.tolist() # dans l'ordre de la matrice X_train/X_test
    return log_feat_importance, lgbm_feat_importance, feats

# Sélection du modèle par l'utilisateur et des features importances correspondantes 
def fct_select_model():
    model_selected = st.sidebar.radio(
        'Modèle de prédiction du score',
        ('Reg Logistique', 'Light GBM'),
    )
    st.sidebar.write('Modèle sélectionné : ', model_selected)
    return model_selected

# Selection d'une demande de crédit client par l'utilisateur
def fct_select_customer():
    sk_id_curr_selected = st.sidebar.selectbox('N° de demande client (crédit)', 
                                               df_appli['SK_ID_CURR'].values)
    st.sidebar.write('N° de demande sélectionné : ',
                     sk_id_curr_selected)
    return sk_id_curr_selected

# Selection d'1 feature parmi les features importance 
def fct_select_feat(label, feat_importances):
    x = st.sidebar.selectbox(label, feat_importances)
    return x

# Selection utilisateur du nbre max de features importantes à afficher (max 50) pour les analyses d'impact local/global 
# et parmi celles-ci sélection de 2 features pour le graphe de positionnement client 
def fct_select_feats_importance(model_selected, max_display=50):
    if model_selected == 'Reg Logistique':
        feat_importances = log_feat_importance[0:max_display]
    else:
        feat_importances = lgbm_feat_importance[0:max_display]
    feat1 = fct_select_feat('Feature 1', feat_importances)
    feat2 = fct_select_feat('Feature 2', feat_importances)
    return feat1, feat2

# Recherche des features X d'une demande client et les transforme en dictionnaire 
# pour les envoyer à l'API pour prédiction de la proba du score d'octroi
def fct_extract_feat(sk_id_curr):
    feats = df_appli.copy()
    feats = df_appli[df_appli['SK_ID_CURR']==sk_id_curr]
    target = feats['TARGET'].values 
    target = target[0]
    feats.drop(['TARGET', 'log_score_pred', 'lgbm_score_pred'], axis=1, inplace=True)
    # Conversion du dataframe des features en dictionnaire (JSON)
    feats = feats.to_dict('list')
    for k, v in feats.items():
        feats[k] = feats[k][0] # 1er élément de la liste
    return feats, target

###########
# Graphes #
###########
# jauge avec la valeur du score de prédiction retournée par l'API 
def fct_jauge_score(score):
    fig_jauge_score = go.Figure(go.Indicator(mode='gauge+number+delta',
                                             value=score,
                                             domain={'x' : [0,1], 'y' : [0,1]},
                                             title={'text' : "Score de prédiction", 'font': {'size': 20}},
                                             delta={'reference' : 0.5,
                                                    'increasing': {'color': 'RebeccaPurple'}},
                                             gauge={'axis' : {'range' : [None, 1]},
                                                    'bar': {'color': 'lightgray'},
                                                    'steps' : [
                                                        {'range' : [0, 0.5], 'color' : 'green'},
                                                        {'range' : [0.5, 1], 'color' : 'red'}]
                                                    #,'threshold' : {'line': {'color': "red", 'width': 4},
                                                    #               'thickness': 0.75,
                                                    #               'value': 0.50}    
                                             }))
    fig_jauge_score.update_layout(showlegend=False,
                                  height=300,
                                  margin={'l':30, 'r':30, 't':40, 'b':40})
    return fig_jauge_score

# Graphe de distribution pour 1 feature importance (feat) par classe réelle (TARGET) selon le modèle choisi
def fct_dist_feat(feat):
    #var_score_pred = 'log_score_pred' if model_selected == 'Reg Logistique' else 'lgbm_score_pred'
    #feat_0 = df_appli[df_appli[var_score_pred] < 0.5][feat].values.tolist()
    #feat_1 = df_appli[df_appli[var_score_pred] >= 0.5][feat].values.tolist()
    
    # liste des data des classe 0 (octroi) et 1 (refus)
    feat_0 = df_appli[df_appli['TARGET'] == 0][feat].values.tolist()
    feat_1 = df_appli[df_appli['TARGET'] == 1][feat].values.tolist()
    feat_0_1 = [feat_0, feat_1]  
    classes = ['Ocroi', 'Refus']
    
    # Distribution de la featue feat par classes
    fig_feat = ff.create_distplot(feat_0_1, group_labels=classes, 
                                  colors=['green', 'red'],
                                  show_hist=False, show_rug=False, 
                                  curve_type='normal')
    fig_feat.update_layout(height=300, width=500,
                           margin={'l':20, 'r':20, 't':0, 'b':0},
                           legend=dict(yanchor='top', y=0.99,
                                       xanchor='right', x=0.99))
    
    # Ajout du positionnement client : droite verticale au point X (val_feat)
    #val_score_pred = df_appli.loc[df_appli['SK_ID_CURR']==sk_id_curr_selected, var_score_pred].values[0]
    #line_color = 'red' if val_score_pred >= 0.5 else 'green'  
    val_feat = df_appli.loc[df_appli['SK_ID_CURR']==sk_id_curr_selected, feat].values[0]
    val_target = df_appli.loc[df_appli['SK_ID_CURR']==sk_id_curr_selected, 'TARGET'].values[0]
    line_color = 'red' if val_target == 1 else 'green'
    fig_feat.add_vline(x=val_feat,  line_dash='dash', line_color=line_color, annotation_text='client')
    return fig_feat

# Graphe de position de l'individu sélectionné dans 1 nuage de points à partir de 2 features sélectionnées
def fct_position_individu(model_selected):
    df = df_appli.copy()
    df['size'] = 0.1 # taille par défaut d'affichage de tous les individus
    df.loc[df['SK_ID_CURR']==sk_id_curr_selected, 'size'] = 1 # taille de l'individu sélectionné
    #df['TARGET'] = df['TARGET'].astype(str) # pour définir les couleurs discrètes
    #colors = ['red' if target==1 else 'green' for target in df['TARGET'].values.tolist() ]
    fig_position = px.scatter(df, x=feat1_selected, y=feat2_selected, 
                              #color_discrete_sequence=['green', 'red'], 
                              #color='TARGET',
                              color='log_score_pred' if model_selected == 'Reg Logistique' else 'lgbm_score_pred',  
                              labels = ['Ocroi', 'Refus'],
                              size='size')
    fig_position.update_layout(height=300, width=500,
                               margin={'l':20, 'r':20, 't':0, 'b':0},
                               legend=dict(yanchor='top', y=0.99,
                                           xanchor='right', x=0.99))
    return fig_position

# Graphe de l'impacte des features locales sur le score d'octroi de crédit
def fct_shap_local(sk_id_curr_selected, model_selected, max_display=50):
    # N° de ligne de la demande sélectionnée
    row = df_appli[df_appli['SK_ID_CURR']==sk_id_curr_selected].index.values[0]
    fig = plt.figure(figsize=(12,5))
    if model_selected == 'Reg Logistique':
        waterfall_plot(log_shap_values[row], max_display=max_display, show=False)
        #plt.savefig('shap_explainer_local.png')
        #image = Image.open('shap_explainer_local.png')
        #return image
    else:
        shap_plots._waterfall.waterfall_legacy(lgbm_explainer.expected_value[0],
                                               lgbm_shap_values[row].values[:,0],
                                               feature_names=feats,
                                               max_display=max_display,
                                               show=False)
    return fig

# Graphe de l'impacte moyen des features globales sur le score d'octroi de crédit
def fct_shap_global(model_selected):
    #fig = plt.figure(figsize=(12,5))
    #summary_plot(log_shap_values, 
                 #feature_names = feats,
                 #plot_type = 'bar',
                 #color='green',
                 #max_display=max_display,
                 #show=True)
    file = 'log_summary_plot.png' if model_selected == 'Reg Logistique' else 'lgbm_summary_plot.png' 
    image = Image.open(file)
    return image
    #return fig

#########################################################
# Appel l'API de prédiction du score d'octroi de crédit #
#########################################################
def fct_API_score_calculate(model_selected, dict_feats):
    if st.sidebar.button('Score calculate', help='Appel 1 API pour calculer le score de prédiction'):
        # inputs au format json
        inputs = {'classifier_name' : model_selected,
                  'features' : dict_feats
        }
        #result = requests.post(url="http://127.0.0.1:8000/score_calculate", data=json.dumps(inputs))
        result = requests.post(url="https://ffinky-test-api.herokuapp.com/score_calculate", data=json.dumps(inputs))
        status_code = 'OK' if (result.status_code==200) else 'KO !'
        st.sidebar.write(f'Réponse API : {status_code}')
        return float(result.text)

########
# MAIN #
########
st.set_page_config(page_title='Dashboard Analyse Relation Client', 
                   layout='wide'
)
st.title('Dashboard de prédiction du score d\'octroi de crédit et d\'analyse de la relation client')

# Data
data_load_state = st.text('Chargement des data pour l\'application (100 échantillons : 50 par classe)...')
df_appli = fct_get_data()
nb_refus = df_appli[df_appli['TARGET']==1].shape[0]
nb_octrois = df_appli[df_appli['TARGET']==0].shape[0]
data_load_state.text(f"Chargement des data réussi : {nb_octrois} octrois / {nb_refus} refus !")

# Init des 50 premieres features importance
log_feat_importance, lgbm_feat_importance, feats = fct_get_feat_importance()
log_shap_values, lgbm_explainer, lgbm_shap_values = fct_get_shap_value()

if st.checkbox('Affichage des data'):
    st.subheader('Application data')
    st.dataframe(df_appli)
    #st.write(df_appli)

######################################################
# Layout (sidebar) utilisé comme filtre du dashboard #
######################################################
# Partie 1 : filtres pour calcul du score 
st.sidebar.markdown('## Filtres du score')
sk_id_curr_selected = fct_select_customer()
dict_feats, target = fct_extract_feat(sk_id_curr_selected)
model_selected = fct_select_model()
score_pred = fct_API_score_calculate(model_selected, dict_feats)

# Partie 2 : fitres sur les features importances
st.sidebar.markdown('## Filtres des features importance')
nb_feats_selected = st.sidebar.slider('Nombre de features importance', min_value=5, max_value=50, value=5)
feat1_selected, feat2_selected = fct_select_feats_importance(model_selected, max_display=nb_feats_selected)

########################################################################################
# Layout (contenu) : 1 colonne pour afficher la Jauge du score de prédiction (appel API)
########################################################################################
# Score de prédiction selon 1 objectif visé : [0, 0.5[ = octroi de crédit, >=0.5 = refus
if score_pred != None:
    status_credit_pred = 'octroi crédit' if score_pred < 0.5 else 'refus crédit'
    status_credit_true = '(réel : octroi)' if target == 0 else '(réel : refus)'
    st.subheader(f'Réponse API de prédiction = {status_credit_pred} {status_credit_true}')
    fig_jauge_score = fct_jauge_score(score_pred)
    st.plotly_chart(fig_jauge_score, use_container_width=True)

#################################################################################################################################
# Layout (contenu) : 2 colonnes pour afficher l'analyse d'impact des features locales/globales sur le score de proba octroi/refus
#################################################################################################################################
col_left, col_right = st.columns(2)
with col_left:
    # Graphe SHAP : impacte des features locales
    st.subheader('Impacte des features locales')
    fig_shap_local = fct_shap_local(sk_id_curr_selected, model_selected, max_display=nb_feats_selected)
    st.pyplot(fig_shap_local)

with col_right:
    # Graphe SHAP : impacte des features globales
    st.subheader('Impacte des features globales')
    #fig_shap_global= fct_shap_global(model_selected)
    #st.pyplot(fig_shap_global)
    image_shape_global = fct_shap_global(model_selected)
    st.image(image_shape_global)

####################################################################################################################################
# Layout (contenu) : 3 colonnes pour afficher la distribution des 2 features sélectionnées et le positionnement de la demande client
####################################################################################################################################
col1, col2, col3 = st.columns(3)
with col1:
    # Graphe de distribution de la feature importance (feature 1) par classes octroi/refus
    st.write(f'Distribution : {feat1_selected}')
    fig_feat1 = fct_dist_feat(feat1_selected)
    st.plotly_chart(fig_feat1, use_container_width=True)
with col2:
    # Graphe de distribution de la feature importance (feature 2) par classes octroi/refus
    st.write(f'Distribution : {feat2_selected}')
    fig_feat2 = fct_dist_feat(feat2_selected)
    st.plotly_chart(fig_feat2, use_container_width=True)
with col3:
    # Graphe de position de la demande client (taille supérieure) dans le nuage des points (feat1, feat2)
    st.write(f'Position de la demande client : {sk_id_curr_selected}')
    fig_position = fct_position_individu(model_selected)
    st.plotly_chart(fig_position, use_container_width=True)