import pandas as pd 
import numpy as np 
import re 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
import plotly.graph_objects as go
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')
nltk.download('punkt_tab')  
nltk.download('stopwords')
nltk.download('wordnet')

def graph_result(jobScores):
    fig = go.Figure()

    bestJobs=jobScores[:5]
    bestJobsTitle = [job for job, score in bestJobs]
    bestJobsScoresValues = [score for job, score in bestJobs]

    # Add radar traces
    fig.add_trace(go.Scatterpolar(r=bestJobsScoresValues, theta=bestJobsTitle, fill='toself', name='Best jobs'))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="📊 Job Profile Match Comparison"
    )
    return fig


def loadData():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    data_path = os.path.join(base_path, "data")

    comp_file = os.path.join(data_path, "competencies.csv")
    jobs_file = os.path.join(data_path, "jobs.csv")

    df_competencies = pd.read_csv(comp_file, sep=",")
    df_jobs = pd.read_csv(jobs_file, sep=",")

    df_jobs["RequiredCompetencies"] = df_jobs["RequiredCompetencies"].apply(lambda x: x.split(";"))

    return df_competencies,df_jobs


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

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english') and len(t) > 2]
    return " ".join(tokens)

def preprocessing(df):
    mappingLevel = {
        1: "Débutant",
        2: "Novice",
        3: "Intermédiaire",
        4: "Avancé",
        5: "Expert"
    }
    mappingCompetence={"level_data_analysis":"python",
                        "level_ml":"Machine Learning",
                        "level_nlp":"NLP", 
                        "level_data_eng":"Tokenization",
                        "level_cloud":"Cloud Computing"
    }
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].apply(lambda numLevel: f"Je suis un {mappingLevel.get(numLevel)} en {mappingCompetence.get(col)}")

        elif pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].fillna("").apply(clean_text)
    return df


def nlp(level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,tools, languages, frameworks, data_types, preferred_domains,experience_text):
    df_question=transformInDf(level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,tools, languages, frameworks, data_types, preferred_domains,experience_text)
    df_question=preprocessing(df_question)
    


    df_competencies, df_jobs = loadData()


    # 3. Charger le modèle SBERT
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('all-mpnet-base-v2')


    # Encoder le texte utilisateur et les compétences
    listQuestion=df_question.iloc[0]
    userEmbedding = model.encode(listQuestion, convert_to_tensor=True)
    print(userEmbedding)
    compEmbeddings = model.encode(df_competencies["Competency"].tolist(), convert_to_tensor=True)

    blockEmbeddings = model.encode(df_competencies["BlockName"].unique().tolist(), convert_to_tensor=True)

    # 4. Similarités cosinus
    compCosineMatrix = util.cos_sim(userEmbedding, compEmbeddings).cpu().numpy()
    compCosineScores = compCosineMatrix.max(axis=0)
    df_competencies["similarity"] = compCosineScores

    print(compCosineScores)
    blockCosineMatrix = util.cos_sim(userEmbedding, blockEmbeddings).cpu().numpy()
    blockCosineScores = blockCosineMatrix.max(axis=0)

    # 5. Pondération par les niveaux
    """  
    df_competencies["weighted_score"] = df_competencies.apply(
        lambda row: row["similarity"] * (1 ),
        axis=1
    )"""
    df_competencies["weighted_score"] = df_competencies.apply(
        lambda row: row["similarity"] * (1 + 0.5*blockCosineScores[row['BlockID']-1]),
        axis=1
    )

    # 6. Score moyen par bloc
    scoresByBlock = df_competencies.groupby("BlockName")["weighted_score"].mean().to_dict()
    print(scoresByBlock)

    # 7. Score par métier
    jobScores = []
    for _, job in df_jobs.iterrows():
        compScores = df_competencies[df_competencies["CompetencyID"].isin(job["RequiredCompetencies"])]["weighted_score"]
        jobScore = compScores.mean() if not compScores.empty else 0
        jobScores.append((job["JobTitle"], jobScore))

    jobScores.sort(key=lambda x: x[1], reverse=True)

    # 8. Radar chart
    #fig = graph_result(scores_by_block)

    print("\n=== 🧠 RECOMMANDATION DE MÉTIERS (SBERT) ===")
    for job, score in jobScores[:5]:
        print(f"- {job}: {round(score, 3)}")

    #return fig, job_scores    
    fig = graph_result(jobScores)

    return fig



def nlp2(level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,tools, languages, frameworks, data_types, preferred_domains,experience_text):
    df_question=transformInDf(level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,tools, languages, frameworks, data_types, preferred_domains,experience_text)
    df_question=preprocessing(df_question)
    
    user_text = " ".join(df_question.iloc[0, 5:]).strip()


    df_competencies, df_jobs = loadData()


    # 3. Charger le modèle SBERT
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encoder le texte utilisateur et les compétences
    user_embedding = model.encode(user_text, convert_to_tensor=True)
    print(user_embedding)
    comp_embeddings = model.encode(df_competencies["Competency"].tolist(), convert_to_tensor=True)

    # 4. Similarités cosinus
    cosine_scores = util.cos_sim(user_embedding, comp_embeddings)[0].cpu().numpy()
    df_competencies["similarity"] = cosine_scores

    print(cosine_scores)

    # 5. Pondération par les niveaux
    level_dict = {
        1: level_data_analysis,
        2: level_ml,
        3: level_nlp,
        4: level_data_eng,
        5: level_cloud
    }
    df_competencies["weighted_score"] = df_competencies.apply(
        lambda row: row["similarity"] * (1 + 0.1 * level_dict.get(row["BlockID"], 0)),
        axis=1
    )

    # 6. Score moyen par bloc
    scores_by_block = df_competencies.groupby("BlockName")["weighted_score"].mean().to_dict()
    print(scores_by_block)

    # 7. Score par métier
    job_scores = []
    for _, job in df_jobs.iterrows():
        comp_scores = df_competencies[df_competencies["CompetencyID"].isin(job["RequiredCompetencies"])]["weighted_score"]
        job_score = comp_scores.mean() if not comp_scores.empty else 0
        job_scores.append((job["JobTitle"], job_score))

    job_scores = sorted(job_scores, key=lambda x: x[1], reverse=True)

    # 8. Radar chart
    #fig = graph_result(scores_by_block)

    print("\n=== 🧠 RECOMMANDATION DE MÉTIERS (SBERT) ===")
    for job, score in job_scores[:5]:
        print(f"- {job}: {round(score, 3)}")

    #return fig, job_scores    
    fig = graph_result()

    return fig
