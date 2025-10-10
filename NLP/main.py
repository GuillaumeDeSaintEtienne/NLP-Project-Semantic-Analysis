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

# --- Étape 1: Chargement du modèle et des données ---
print("Loading SBERT model...")
model = SentenceTransformer('all-mpnet-base-v2')
print("Model loaded.")





def graph_result(jobScores):
    fig = go.Figure()

    bestJobsTitle = [job for job, score, topSkills in jobScores]
    bestJobsScoresValues = [round(score * 100, 2) for job, score, topSkills in jobScores]

    maxScore=max(bestJobsScoresValues)
    minScore=min(bestJobsScoresValues)

    fig = go.Figure(data=go.Scatterpolar(r=bestJobsScoresValues, theta=bestJobsTitle, fill='toself', name='Profile Match'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[minScore-4, maxScore+1])),
        title="Overall Job Profile Match (Weighted)"
    )
    
    return fig


def loadData():
    try:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        data_path = os.path.join(base_path, "data")

        comp_file = os.path.join(data_path, "competencies.csv")
        jobs_file = os.path.join(data_path, "jobs.csv")

        df_competencies = pd.read_csv(comp_file, sep=",")
        df_jobs = pd.read_csv(jobs_file, sep=",")

        df_jobs["RequiredCompetencies"] = df_jobs["RequiredCompetencies"].apply(lambda x: x.split(";"))
    except FileNotFoundError:
        print("Error: Make sure 'competencies.csv' and 'jobs.csv' are in a 'data' folder at the project root.")
        df_competencies = pd.DataFrame()
        df_jobs = pd.DataFrame()

    return df_competencies,df_jobs


def transformInDf(level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,
                        tools, languages, frameworks, data_types, preferred_domains,
                        experience_text, challenges, learning_goals):
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
        "experience_text": experience_text,
        "challenges" : challenges,
        "learning_goals" : learning_goals
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
        1: "Beginner",
        2: "Novice",
        3: "Intermediate",
        4: "Advanced",
        5: "Expert"
    }
    mappingCompetence={"level_data_analysis":"python",
                        "level_ml":"Machine Learning",
                        "level_nlp":"NLP", 
                        "level_data_eng":"Tokenization",
                        "level_cloud":"Cloud Computing"
    }
    mappingExperience = {
        "experience_text":"My experience includes:  ",
        "tools":"I have used these tools and software:  ",
        "languages":"I know the following programming languages: ",
        "frameworks":"I am familiar with these frameworks and libraries: ",
        "data_types":"I have worked with these types of data: ",
        "preferred_domains":"I am interested in these domains: "
    }

    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].apply(lambda numLevel: f"I am a {mappingLevel.get(numLevel)} in {mappingCompetence.get(col)}")

        elif pd.api.types.is_string_dtype(df[col]):
            if col in mappingExperience.keys():
                df[col] = df[col].apply(lambda x: f"{mappingExperience[col]} {x}" if pd.notnull(x) and x != "" else "")
            df[col] = df[col].fillna("").apply(clean_text)
    return df


def nlp(level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,
            tools, languages, frameworks, data_types, preferred_domains,
            experience_text, challenges, learning_goals
        ):
    
    df_question=transformInDf(level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,
        tools, languages, frameworks, data_types, preferred_domains,
        experience_text, challenges, learning_goals)
    
    df_question=preprocessing(df_question)
    
    df_competencies, df_jobs = loadData()

    #model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('all-mpnet-base-v2')

    listQuestion=df_question.iloc[0]
    userEmbedding = model.encode(listQuestion, convert_to_tensor=True)
    print(userEmbedding)
    compEmbeddings = model.encode(df_competencies["Competency"].tolist(), convert_to_tensor=True)

    blockEmbeddings = model.encode(df_competencies["BlockName"].unique().tolist(), convert_to_tensor=True)

    compCosineMatrix = util.cos_sim(userEmbedding, compEmbeddings).cpu().numpy()
    compCosineScores = compCosineMatrix.max(axis=0)
    df_competencies["similarity"] = compCosineScores

    print(compCosineScores)
    blockCosineMatrix = util.cos_sim(userEmbedding, blockEmbeddings).cpu().numpy()
    blockCosineScores = blockCosineMatrix.max(axis=0)

    df_competencies["weightedScore"] = df_competencies.apply(
        lambda row: row["similarity"] * (1 + 0.1*blockCosineScores[row['BlockID']-1]),
        axis=1
    )

    scoresByBlock = df_competencies.groupby("BlockName")["weightedScore"].mean().to_dict()
    print(scoresByBlock)

    jobScores = []
 
    for _, job in df_jobs.iterrows():
        jobComps = df_competencies[df_competencies["CompetencyID"].isin(job["RequiredCompetencies"])]
        topCompScores = jobComps.sort_values(by='weightedScore', ascending=False).head(3)

        jobScore = jobComps["weightedScore"].mean() if not jobComps.empty else 0
        jobScores.append((
            job["JobTitle"],
            jobScore,
            topCompScores["Competency"].tolist()
        ))
    jobScores.sort(key=lambda x: x[1], reverse=True)
      
        

    print("\n=== 🧠 RECOMMANDATION DE MÉTIERS (SBERT) ===")
    top3Jobs=[]
    for job, score, topSkills in jobScores[:3]:
        top3Jobs.append({
            "title": job,
            "score": round(score * 100, 2),
            "matching_skills": topSkills
            })
        print(f"- {job}: {round(score, 3)}")

    fig = graph_result(jobScores)

    return {
        "fig": fig,
        "top_jobs": top3Jobs,
        "block_scores": scoresByBlock
    }

