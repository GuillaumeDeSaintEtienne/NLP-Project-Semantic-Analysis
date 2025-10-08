import streamlit as st
import plotly.graph_objects as go
import pandas as pd

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from NLP import main

def start():
    st.set_page_config(page_title="Job Finder", page_icon="💼", layout="centered")
    st.title("Find Your Ideal Job")
    st.write("Answer a few questions and find out the best job opportunities based on your profile.")

    with st.form("job_form"):
        st.header("👤 Your Profile")

        st.subheader("📊 Évaluez vos compétences (1 = Débutant, 5 = Expert)")
        level_data_analysis = st.slider("1️⃣ Niveau en **Data Analysis**", 1, 5, 3)
        level_ml = st.slider("2️⃣ Niveau en **Machine Learning / IA**", 1, 5, 2)
        level_nlp = st.slider("3️⃣ Niveau en **NLP / Computer Vision**", 1, 5, 1)
        level_data_eng = st.slider("4️⃣ Niveau en **Data Engineering**", 1, 5, 2)
        level_cloud = st.slider("5️⃣ Niveau en **Cloud / MLOps**", 1, 5, 1)
        
        st.subheader("💡 Domaines & Outils")
        tools = st.text_input("6️⃣ Outils / logiciels")
        languages = st.text_input("7️⃣ Langages de programmation")
        frameworks = st.text_input("8️⃣ Frameworks / librairies IA")
        data_types = st.text_input("9️⃣ Types de données manipulés")
        preferred_domains = st.text_input("🔟 Domaines / types de projets")
        
        st.subheader("📝 Votre expérience")
        experience_text = st.text_area("1️⃣1️⃣ Décrivez votre expérience")

        # This just returns True when clicked
        submitted = st.form_submit_button("🔍 Find My Job")

    if submitted:
        fig = main.nlp(
            level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,
            tools, languages, frameworks, data_types, preferred_domains,
            experience_text
        )
        st.plotly_chart(fig, use_container_width=True)




start()
