import pandas as pd 
import re 
import plotly.graph_objects as go
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sentence_transformers import SentenceTransformer, util
import math

nltk.download('punkt')
nltk.download('punkt_tab')  
nltk.download('stopwords')
nltk.download('wordnet')

print("Loading SBERT model...")
#MODEL_ID = 'all-MiniLM-L6-v2'
MODEL_ID = 'all-mpnet-base-v2'
print("Model loaded.")
data_path = Path.cwd() / "data"





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
        df_competencies = pd.read_csv(data_path / r"competencies.csv", sep=",")
        df_jobs = pd.read_csv(data_path / r"jobs.csv", sep=",")

        df_jobs["RequiredCompetencies"] = df_jobs["RequiredCompetencies"].apply(lambda x: x.split(";"))
        
    except FileNotFoundError:
        print("Error: Make sure 'competencies.csv' and 'jobs.csv' are in a 'data' folder at the project root.")
        df_competencies = pd.DataFrame()
        df_jobs = pd.DataFrame()

    return df_competencies,df_jobs


def transformInDf(level_python, level_ai, level_visu, level_sql, level_token_embedding,
                        tools, languages, frameworks, data_types, preferred_domains,
                        experience_text, challenges, learning_goals):
    data = {
        "level_python": level_python,
        "level_ai": level_ai,
        "level_visu": level_visu,
        "level_sql": level_sql,
        "level_token_embedding": level_token_embedding,
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
    mappingCompetence={"level_python":"python",
                        "level_ai":"Artificial Intelligence",
                        "level_visu":"Visualization", 
                        "level_sql":"SQL",
                        "level_token_embedding":"Tokenization and embeddings"
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


def nlp(level_python, level_ai, level_visu, level_sql, level_token_embedding,
            tools, languages, frameworks, data_types, preferred_domains,
            experience_text, challenges, learning_goals
        ):
    
    df_question=transformInDf(level_python, level_ai, level_visu, level_sql, level_token_embedding,
        tools, languages, frameworks, data_types, preferred_domains,
        experience_text, challenges, learning_goals)
    
    df_question=preprocessing(df_question)
    
    df_competencies, df_jobs = loadData()

    total_jobs = len(df_jobs)
    #Explode the list of competencies for each job to count occurrences
    competency_counts = df_jobs.explode('RequiredCompetencies')['RequiredCompetencies'].value_counts().to_dict()

    #Calculate IDF score for each competency and add it as a new column
    def calculate_idf(competency_id):
        count = competency_counts.get(competency_id, 0)
        return math.log(total_jobs / (count + 1))

    df_competencies['idf_score'] = df_competencies['CompetencyID'].apply(calculate_idf)

    model = SentenceTransformer(MODEL_ID)

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
        #Get the full competency details (including the new idf_score)
        jobComps = df_competencies[df_competencies["CompetencyID"].isin(job["RequiredCompetencies"])]
        
        if not jobComps.empty:
            #Multiply the user's match score by the competency's rarity score
            weighted_job_score = (jobComps['weightedScore'] * jobComps['idf_score']).sum()
            
            #We normalize by the sum of IDF scores to avoid bias towards jobs with more skills
            sum_of_idf = jobComps['idf_score'].sum()
            
            #The final score is a weighted average
            jobScore = weighted_job_score / sum_of_idf if sum_of_idf > 0 else 0
            
            #Get the top skills based on the user's score for them, not their rarity
            topCompScores = jobComps.sort_values(by='weightedScore', ascending=False).head(3)
        else:
            jobScore = 0
            topCompScores = pd.DataFrame(columns=["Competency"]) #Ensure it's an empty list if no comps
        
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

