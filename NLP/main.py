import pandas as pd 
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

MODEL_ID = "all-MiniLM-L6-v2"

data_path = Path.cwd() / "data"
 

def loadData():

    df_competencies = pd.read_csv(data_path / r"competencies.csv", sep=",")
    df_jobs = pd.read_csv(data_path / r"jobs.csv", sep=",")

    df_jobs["RequiredCompetencies"] = df_jobs["RequiredCompetencies"].apply(lambda x: x.split(";"))

    return df_competencies,df_jobs


def graph_result(df_jobs, block_scores):
    fig = go.Figure()

    categories = ['Technical Skills', 'Analytical Thinking', 'Communication', 'Creativity', 'Domain Knowledge']

    # Example job skill profiles
    data_analyst = [8, 9, 7, 6, 8]
    data_scientist = [9, 9, 6, 7, 9]
    ml_engineer = [10, 8, 6, 5, 8]
    business_analyst = [7, 8, 9, 6, 8]

    # Add radar traces
    fig.add_trace(go.Scatterpolar(r=data_analyst, theta=categories, fill='toself', name='Data Analyst'))
    fig.add_trace(go.Scatterpolar(r=data_scientist, theta=categories, fill='toself', name='Data Scientist'))
    fig.add_trace(go.Scatterpolar(r=ml_engineer, theta=categories, fill='toself', name='ML Engineer'))
    fig.add_trace(go.Scatterpolar(r=business_analyst, theta=categories, fill='toself', name='Business Analyst'))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        title="📊 Job Profile Match Comparison"
    )
    return fig


def transformInDf(level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,
                  tools, languages, frameworks, data_types, preferred_domains, experience_text):
    data = {
        "level_data_analysis": level_data_analysis,
        "level_ml": level_ml,
        "level_nlp": level_nlp,
        "level_data_eng": level_data_eng,
        "level_cloud": level_cloud,
        "tools": tools,
        "languages": languages,
        "frameworks": frameworks,
        "data_types": data_types,
        "preferred_domains": preferred_domains,
        "experience_text": experience_text
    }
    df = pd.DataFrame([data])
    return df

def preprocessing(df):
    for col in df.columns:
        if "level" in col : 
            df[col] = df[col].astype(int)
        else : 
            df[col] = df[col].astype(str)

    return df

def processing_user_input(df, model):
    user_embeddings = []
    for col in df.columns:
        if df[col].dtype != "int":
            user_embeddings.append(model.encode(df[col].astype(str).tolist(), convert_to_tensor=True)) 
    return user_embeddings



def score(df, df_competencies, model):
    user_embeddings = processing_user_input(df, model)
    block_scores = {}

    for block in df_competencies["BlockName"].unique():
        # Get all competencies for this block and make sure they are strings
        competencies = df_competencies["Competency"][df_competencies["BlockName"] == block]
        block_embeddings = model.encode([str(c) for c in competencies if pd.notna(c)], convert_to_tensor=True)

        all_max_sims = []

        # Loop over each column's embeddings
        for col_emb in user_embeddings:
            similarities = util.cos_sim(col_emb, block_embeddings)
            max_similarities = [float(sim.max()) for sim in similarities]
            all_max_sims.extend(max_similarities)

        block_scores[block] = np.mean(all_max_sims)

    return block_scores



def nlp(level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,tools, languages, frameworks, data_types, preferred_domains,experience_text):
    df=transformInDf(level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,tools, languages, frameworks, data_types, preferred_domains,experience_text)
    df_competencies, df_jobs = loadData()
    df=preprocessing(df)

    block_scores = score(df, df_competencies, SentenceTransformer(MODEL_ID))
    

    fig = graph_result(df_jobs, block_scores)

    return fig



