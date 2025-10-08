import pandas as pd 
import numpy as np 
import re 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def graph_result():
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
    return None



def nlp(level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,tools, languages, frameworks, data_types, preferred_domains,experience_text):
    df=transformInDf(level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,tools, languages, frameworks, data_types, preferred_domains,experience_text)
    df=preprocessing(df)
    fig = graph_result()

    return fig