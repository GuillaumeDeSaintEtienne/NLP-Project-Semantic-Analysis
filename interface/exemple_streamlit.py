import streamlit as st
import plotly as plt
import pandas as pd

st.set_page_config(page_title="Job Finder", page_icon="💼", layout="centered")

st.title("Find Your Ideal Job")
st.write("Answer a few questions and find out the best job opportunities based on your profile.")

with st.form("job_form"):
    st.header("👤 Your Profile")
    name = st.text_input("1️⃣ What's your name?")
    age = st.number_input("2️⃣ How old are you?", min_value=16, max_value=80, step=1)
    education = st.selectbox("3️⃣ What's your highest level of education?", 
                             ["High school", "Bachelor's degree", "Master's degree", "PhD", "Other"])
    field = st.text_input("4️⃣ What field did you study or specialize in?")
    experience = st.number_input("5️⃣ How many years of professional experience do you have?", min_value=0, max_value=40)
    
    st.header("💡 Your Preferences")
    skills = st.text_area("6️⃣ List a few of your key skills (comma-separated):")
    interests = st.text_area("7️⃣ What kind of work interests you the most?")
    work_env = st.radio("8️⃣ Do you prefer working remotely, on-site, or hybrid?", 
                        ["Remote", "On-site", "Hybrid"])
    salary = st.slider("9️⃣ What is your desired monthly salary range (in USD)?", 1000, 10000, (3000, 6000))
    location = st.text_input("🔟 Where would you like to work? (city, country)")

    submitted = st.form_submit_button("🔍 Find My Job")

if submitted:
    df = pd.DataFrame(columns=["name", "age", "education", "field", "experience", "skills", "interests", "work_env", "salary", "location", "submitted"])
    df.loc[df.shape[0]+1] = [name, age, education, field, experience, skills, interests, work_env, salary, location, submitted]
    
    st.success(f"Thanks {name}! I'm analyzing your responses... 💭")
    
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
