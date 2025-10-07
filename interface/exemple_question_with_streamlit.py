import streamlit as st
import plotly as plt
import pandas as pd

st.set_page_config(page_title="Job Finder", page_icon="💼", layout="centered")

st.title("Find Your Ideal Job")
st.write("Answer a few questions and find out the best job opportunities based on your profile.")

with st.form("job_form"):
    st.header("👤 Your Profile")

    st.subheader("📊 Évaluez vos compétences (1 = Débutant, 5 = Expert)")
    level_data_analysis = st.slider("1️⃣ Niveau en **Data Analysis** (Python, SQL, statistiques, visualisation)", 1, 5, 3)
    level_ml = st.slider("2️⃣ Niveau en **Machine Learning / IA** (modélisation, scikit-learn, deep learning…)", 1, 5, 2)
    level_nlp = st.slider("3️⃣ Niveau en **NLP / Computer Vision**", 1, 5, 1)
    level_data_eng = st.slider("4️⃣ Niveau en **Data Engineering** (ETL, pipelines, Spark…)", 1, 5, 2)
    level_cloud = st.slider("5️⃣ Niveau en **Cloud / MLOps** (AWS, Docker, CI/CD…)", 1, 5, 1)
    
    st.subheader("💡 Domaines & Outils")
    tools = st.text_input("6️⃣ Quels **outils / logiciels** maîtrisez-vous ? (ex: Tableau, Power BI, VS Code...)")
    languages = st.text_input("7️⃣ Quels **langages de programmation** connaissez-vous ? (ex: Python, SQL, R, Java...)")
    frameworks = st.text_input("8️⃣ Quels **frameworks / librairies IA** avez-vous utilisés ? (ex: TensorFlow, PyTorch, HuggingFace...)")
    data_types = st.text_input("9️⃣ Quels **types de données** avez-vous manipulés ? (ex: images, texte, séries temporelles, données tabulaires...)")
    preferred_domains = st.text_input("🔟 Quels **domaines / types de projets** vous intéressent le plus ? (ex: NLP, Data Engineering, Computer Vision...)")

    st.subheader("📝 Votre expérience")
    experience_text = st.text_area("1️⃣1️⃣ Décrivez brièvement votre **parcours ou une expérience marquante** liée à la Data ou à l'IA.")

    submitted = st.form_submit_button("🔍 Find My Job")

if submitted:
    df = pd.DataFrame(columns=["level_data_analysis", "level_ml", "level_nlp", "level_data_eng", "level_cloud", "tools", "languages", "frameworks", "data_types", "preferred_domains","experience_text", "submitted"])
    df.loc[df.shape[0]+1] = [level_data_analysis, level_ml, level_nlp, level_data_eng, level_cloud, tools, languages, frameworks, data_types, preferred_domains,experience_text, submitted]
    
    st.success(f"Thanks ! I'm analyzing your responses... 💭")
    
    # --- Placeholder output ---
    st.info("✅ Based on your profile, you might be a great fit for:")
    st.write("- **Data Analyst** at a tech startup (Remote)")
    st.write("- **Business Intelligence Specialist** in your region")
    
    # --- Radar chart ---
    fig = plt.graph_objects.Figure()

    # Define the categories (axes)
    categories = ['Technical Skills', 'Analytical Thinking', 'Communication', 'Creativity', 'Domain Knowledge']

    # Define example scores for each job
    data_analyst = [8, 9, 7, 6, 8]
    data_scientist = [9, 9, 6, 7, 9]
    ml_engineer = [10, 8, 6, 5, 8]
    business_analyst = [7, 8, 9, 6, 8]

    # Add radar traces
    fig.add_trace(plt.graph_objects.Scatterpolar(
        r=data_analyst,
        theta=categories,
        fill='toself',
        name='Data Analyst'
    ))

    fig.add_trace(plt.graph_objects.Scatterpolar(
        r=data_scientist,
        theta=categories,
        fill='toself',
        name='Data Scientist'
    ))

    fig.add_trace(plt.graph_objects.Scatterpolar(
        r=ml_engineer,
        theta=categories,
        fill='toself',
        name='ML Engineer'
    ))

    fig.add_trace(plt.graph_objects.Scatterpolar(
        r=business_analyst,
        theta=categories,
        fill='toself',
        name='Business Analyst'
    ))

    # Layout customization
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10])
        ),
        showlegend=True,
        title="📊 Job Profile Match Comparison"
    )

    # ✅ Display the radar chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)
