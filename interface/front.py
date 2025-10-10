import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
import os

# --- Corrected Path Handling ---
# Get the absolute path of the current file's directory (interface/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory (the project root)
project_root = os.path.dirname(current_dir)
# Add the project root to the system's path
sys.path.append(project_root)

# Now, Python can find the NLP module which is in a sibling directory
from NLP import main

def start():
    st.set_page_config(page_title="Job Finder", page_icon="💼", layout="wide")
    st.title("Find Your Ideal Job 🔎")
    st.write("Answer a few questions and find out the best job opportunities based on your profile.")

    with st.form("job_form"):
        st.header("👤 Your Profile")

        # Using columns for a cleaner layout
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Rate Your Skills (1-Beginner, 5-Expert)")
            level_data_analysis = st.slider("How much do you love python ?", 1, 5, 3)
            level_ml = st.slider("Do you like working with AI ?", 1, 5, 2)
            level_nlp = st.slider("Can you make art with data ?", 1, 5, 1)
            level_data_eng = st.slider("How confident are you concerning your knoledge in SQL ?", 1, 5, 2)
            level_cloud = st.slider("How familiar are you with tokkenization and embeddings ?", 1, 5, 1)

        with col2:
            st.subheader("💡 Domains & Tools")
            tools = st.text_input("Tools / software (e.g., Power BI, Excel)")
            languages = st.text_input("Programming languages (e.g., Python, R, SQL)")
            frameworks = st.text_input("AI frameworks / libraries (e.g., Scikit-learn, Pandas)")
            data_types = st.text_input("Types of data handled (e.g., tabular, text, images)")
            preferred_domains = st.text_input("Preferred domains (e.g., finance, healthcare)")
        
        st.subheader("📝 Describe Your Experience")
        experience_text = st.text_area("Provide a summary of your projects and professional experience.", height=150)
        challenges = st.text_area("What was the biggest challenge you faced in your projects, and how did you overcome it?", height=150)
        learning_goals = st.text_area("What skills or domains are you looking to improve or learn next?", height=100)

        submitted = st.form_submit_button("🔍 Find My Job")

    if submitted:
        if all([level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,
                tools, languages, frameworks, data_types, preferred_domains,
                experience_text, challenges, learning_goals
                ]):
        
            if not all([tools, languages, frameworks, experience_text]):
                st.warning("⚠️ Please fill in all the text fields for an accurate analysis!")
            else:
                with st.spinner('Analyzing your profile...'):
                    results = main.nlp(
                        level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud,
                        tools, languages, frameworks, data_types, preferred_domains,
                        experience_text, challenges, learning_goals
                    )

                if results:
                    st.header("📈 Your Personalized Results Dashboard")
                    st.subheader("🏆 Your Top 3 Job Recommendations")
                    cols = st.columns(3)
                    for i, job in enumerate(results["top_jobs"]):
                        with cols[i]:
                            st.metric(label=job['title'], value=f"{job['score']}% Match")
                            with st.expander("Why this recommendation?"):
                                st.write("This role is a good fit because of your skills in:")
                                for skill in job['matching_skills']:
                                    st.markdown(f"- **{skill}**")
                    
                    st.markdown("---")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🎯 Overall Profile Match")
                        st.plotly_chart(results["fig"], use_container_width=True)
                    with col2:
                        st.subheader("💡 Competency Block Coverage")
                        st.write("This shows how well your profile covers different skill areas.")
                        block_names = list(results["block_scores"].keys())
                        block_values = list(results["block_scores"].values())
                        bar_fig = go.Figure([go.Bar(x=block_values, y=block_names, orientation='h', text=block_values, textposition='auto', marker_color='#4169E1')])
                        bar_fig.update_layout(title="Coverage per Skill Category (%)", xaxis_title="Coverage Score", yaxis_title="Competency Block")
                        st.plotly_chart(bar_fig, use_container_width=True)
                else:
                    st.error("❌ Could not process your profile. Please check if the data files are available.")
        else : 
            submitted = False
            st.warning("Please answer all the questions !!")

        
start()