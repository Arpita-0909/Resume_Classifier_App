import streamlit as st
import joblib
import re
import tempfile
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity

# Load model files
model = joblib.load("resume_classifier.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("encoder.pkl")

# -----------------------------
# TEXT CLEANING FUNCTION
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# -----------------------------
# SKILL EXTRACTION FUNCTION
# -----------------------------
def extract_skills(text, skills_list):
    text = text.lower()
    found_skills = set()

    for skill in skills_list:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found_skills.add(skill)

    return found_skills
# -----------------------------
# SKILL LIST (IMPROVED)
# -----------------------------
skills_list = [
    "python","sql","machine learning","ml",
    "deep learning","dl","nlp","natural language processing",
    "pandas","numpy","data analysis",
    "excel","power bi","powerbi","tableau",
    "aws","spark","docker","hadoop",
    "tensorflow","pytorch","statistics",
    "html","css","javascript","react","node.js"
]

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("AI Resume Analyzer")
st.write("Upload your resume to predict job roles")

# Upload Resume
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# Job Description Input
st.subheader("Paste Job Description")
job_description = st.text_area("Enter Job Description Here")

# -----------------------------
# PROCESS FILE
# -----------------------------
if uploaded_file is not None:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    # Extract text
    resume_text = extract_text(path)

    st.subheader("Extracted Resume Preview")
    st.write(resume_text[:500])

    # -----------------------------
    # EXTRACT RESUME SKILLS
    # -----------------------------
    resume_skills = extract_skills(resume_text, skills_list)

    st.subheader("Detected Skills")
    st.write(list(resume_skills))

    # -----------------------------
    # TEXT CLEANING
    # -----------------------------
    cleaned_resume = clean_text(resume_text)
    resume_vector = tfidf.transform([cleaned_resume])

    # -----------------------------
   

    # -----------------------------
    # JOB DESCRIPTION MATCH
    # -----------------------------
    if job_description:

        cleaned_jd = clean_text(job_description)
        jd_vector = tfidf.transform([cleaned_jd])

        similarity = cosine_similarity(resume_vector, jd_vector)[0][0]
        match_score = similarity * 100

        st.subheader("Resume vs Job Description Match")
        st.write(f"Match Score: {match_score:.2f}%")
        st.progress(int(match_score))

        # -----------------------------
        # EXTRACT JD SKILLS
        # -----------------------------
        jd_skills = extract_skills(job_description, skills_list)

        # -----------------------------
       

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        st.subheader("Skill Analysis")

        st.write(" Your Skills:", ", ".join(resume_skills) if resume_skills else "None")
        st.write(" Required Skills:", ", ".join(jd_skills) if jd_skills else "None")

       