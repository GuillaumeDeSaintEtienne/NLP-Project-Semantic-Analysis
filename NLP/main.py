import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go
import os

# --- Étape 1: Chargement du modèle et des données ---
print("Loading SBERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

try:
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df_competencies = pd.read_csv(os.path.join(base_path, "data/competencies.csv"))
    df_jobs = pd.read_csv(os.path.join(base_path, "data/jobs.csv"))
    df_jobs["RequiredCompetencies"] = df_jobs["RequiredCompetencies"].apply(lambda x: x.split(";"))
    competency_texts = df_competencies['Competency'].tolist()
    competency_embeddings = model.encode(competency_texts, convert_to_tensor=True)
    df_competencies['Embeddings'] = list(competency_embeddings)
except FileNotFoundError:
    print("Error: Make sure 'competencies.csv' and 'jobs.csv' are in a 'data' folder at the project root.")
    df_competencies = pd.DataFrame()
    df_jobs = pd.DataFrame()

# --- Étape 2: Définition de la fonction principale de traitement ---
def nlp(level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,
        tools, languages, frameworks, data_types, preferred_domains,
        experience_text):
    if df_jobs.empty or df_competencies.empty:
        return None

    # --- Étape 3: Combinaison des entrées utilisateur ---
    user_profile_text = (
        f"My experience includes: {experience_text}. "
        f"I have used these tools and software: {tools}. "
        f"I know the following programming languages: {languages}. "
        f"I am familiar with these frameworks and libraries: {frameworks}. "
        f"I have worked with these types of data: {data_types}. "
        f"I am interested in these domains: {preferred_domains}."
    )

    # --- Étape 4: Calcul de la similarité sémantique ---
    user_embedding = model.encode(user_profile_text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, competency_embeddings)
    df_competencies['UserSimilarity'] = cosine_scores[0].cpu().numpy()

    # --- Étape 5: Calcul des scores pour les jobs & blocs ---

    # Assurez-vous que les clés ici correspondent exactement aux 'BlockName' dans votre competencies.csv
    slider_map = {
        'Data Analysis': level_data_analysis,
        'Machine Learning': level_ml,
        'NLP': level_nlp,
        'Data Engineering': level_data_eng,
        'Cloud & MLOps': level_cloud 
    }


    def calculate_weight(rating, sensitivity_factor=0.4):
    # Centre la note autour de 0 (note 3 = 0, note 5 = 2, note 1 = -2)
        centered_rating = rating - 3
        # Applique une fonction exponentielle pour un poids non-linéaire
        return np.exp(centered_rating * sensitivity_factor)

    df_competencies['SliderWeight'] = df_competencies['BlockName'].map(slider_map).apply(lambda x: calculate_weight(x) if pd.notnull(x) else 1.0)
    df_competencies['WeightedSimilarity'] = df_competencies['UserSimilarity'] * df_competencies['SliderWeight']
    df_competencies['WeightedSimilarity'] = np.clip(df_competencies['WeightedSimilarity'], 0, 1)


    job_scores = {}
    for index, row in df_jobs.iterrows():
        required_competencies_list = row['RequiredCompetencies']
        relevant_scores = df_competencies[df_competencies['CompetencyID'].isin(required_competencies_list)]['WeightedSimilarity']
        average_score = relevant_scores.mean() if not relevant_scores.empty else 0
        job_scores[row['JobTitle']] = average_score
    
    block_scores = df_competencies.groupby('BlockName')['WeightedSimilarity'].mean().apply(lambda x: round(x * 100, 2)).to_dict()

    # Le reste du code pour trouver le top 3 des jobs est identique mais utilisera les `job_scores` mis à jour
    sorted_jobs = sorted(job_scores.items(), key=lambda item: item[1], reverse=True)
    
    top_3_jobs = []
    for job_title, score in sorted_jobs[:3]:
        job_info = df_jobs[df_jobs['JobTitle'] == job_title].iloc[0]
        required_competencies_ids = job_info['RequiredCompetencies']
        job_competencies_df = df_competencies[df_competencies['CompetencyID'].isin(required_competencies_ids)]
        top_matching_skills = job_competencies_df.sort_values(by='WeightedSimilarity', ascending=False).head(3)
        
        top_3_jobs.append({
            "title": job_title,
            "score": round(score * 100, 2),
            "matching_skills": top_matching_skills['Competency'].tolist()
        })
        
    # --- Étape 6: Génération de la visualisation ---
    labels = list(job_scores.keys())
    values = [round(v * 100, 2) for v in job_scores.values()]
    fig = go.Figure(data=go.Scatterpolar(r=values, theta=labels, fill='toself', name='Profile Match'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Overall Job Profile Match (Weighted)"
    )

    # --- Étape 7: Retourner tous les résultats dans un dictionnaire ---
    return {
        "fig": fig,
        "top_jobs": top_3_jobs,
        "block_scores": block_scores
    }