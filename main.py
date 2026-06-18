import re
import pickle
import io
import os

import pdfplumber
from docx import Document
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Load model artifacts ──────────────────────────────────────────
BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE, "tfidf.pkl"), "rb") as f:
    tfidf = pickle.load(f)
with open(os.path.join(BASE, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(title="ResumeIQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Constants ─────────────────────────────────────────────────────
ALL_SKILLS = [
    "python", "sql", "machine learning", "artificial intelligence",
    "data science", "dbms", "database design", "full stack",
    "html", "css", "javascript", "react", "flask", "node",
    "git", "power bi", "tableau", "excel", "data structures",
    "algorithms", "mysql", "mongodb", "aws", "docker",
    "tensorflow", "keras", "numpy", "pandas", "matplotlib",
    "scikit learn", "deep learning", "nlp", "computer vision",
    "java", "c++", "typescript", "angular", "vue", "django",
    "fastapi", "kubernetes", "linux", "bash", "spark", "hadoop",
]

REQUIRED_SKILLS = [
    "python", "sql", "machine learning", "artificial intelligence",
    "data science", "git", "data structures", "algorithms",
    "aws", "docker", "tensorflow", "power bi",
]

COMPANIES = [
    {"name": "Google",      "role": "Data Science",      "skills": ["python", "machine learning", "sql", "data science"],      "color": "#4285F4", "abbr": "G"},
    {"name": "Microsoft",   "role": "Data Science",      "skills": ["python", "sql", "power bi", "data science"],              "color": "#00A4EF", "abbr": "M"},
    {"name": "Amazon",      "role": "Data Science",      "skills": ["python", "machine learning", "data science", "aws"],       "color": "#FF9900", "abbr": "A"},
    {"name": "Accenture",   "role": "Data Science",      "skills": ["python", "sql", "data science", "tableau"],               "color": "#A100FF", "abbr": "Ac"},
    {"name": "Infosys",     "role": "Data Science",      "skills": ["python", "sql", "data science"],                          "color": "#007CC3", "abbr": "I"},
    {"name": "Zoho",        "role": "Python Developer",  "skills": ["python", "sql", "git"],                                   "color": "#E42527", "abbr": "Z"},
    {"name": "Freshworks",  "role": "Python Developer",  "skills": ["python", "javascript", "react"],                         "color": "#25C16F", "abbr": "F"},
    {"name": "TCS",         "role": "Data Science",      "skills": ["python", "sql", "data science", "machine learning"],      "color": "#E31937", "abbr": "T"},
    {"name": "Wipro",       "role": "Data Science",      "skills": ["python", "sql", "machine learning"],                      "color": "#341C6B", "abbr": "W"},
    {"name": "IBM",         "role": "Data Science",      "skills": ["python", "machine learning", "deep learning", "sql"],     "color": "#1F70C1", "abbr": "IB"},
    {"name": "Razorpay",    "role": "Python Developer",  "skills": ["python", "sql", "flask", "git"],                         "color": "#2D81F7", "abbr": "R"},
    {"name": "PhonePe",     "role": "Data Science",      "skills": ["python", "sql", "machine learning", "data science"],      "color": "#5F259F", "abbr": "P"},
]

CAREER_PATHS = {
    "Data Science": [
        {"title": "Data Scientist",          "emoji": "🧠", "demand": "Very High"},
        {"title": "ML Engineer",             "emoji": "⚙️", "demand": "Very High"},
        {"title": "AI Researcher",           "emoji": "🔬", "demand": "High"},
        {"title": "Data Analyst",            "emoji": "📊", "demand": "High"},
    ],
    "Python Developer": [
        {"title": "Backend Developer",       "emoji": "🖥️", "demand": "Very High"},
        {"title": "API Engineer",            "emoji": "🔗", "demand": "High"},
        {"title": "Data Engineer",           "emoji": "🗄️", "demand": "High"},
        {"title": "Full Stack Developer",    "emoji": "🌐", "demand": "High"},
    ],
    "Web Designing": [
        {"title": "Frontend Developer",      "emoji": "🎨", "demand": "High"},
        {"title": "UI/UX Designer",          "emoji": "✏️", "demand": "High"},
        {"title": "Full Stack Developer",    "emoji": "🌐", "demand": "Very High"},
        {"title": "React Developer",         "emoji": "⚛️", "demand": "High"},
    ],
    "Java Developer": [
        {"title": "Backend Java Developer",  "emoji": "☕", "demand": "High"},
        {"title": "Android Developer",       "emoji": "📱", "demand": "High"},
        {"title": "Full Stack Developer",    "emoji": "🌐", "demand": "High"},
        {"title": "Software Engineer",       "emoji": "💻", "demand": "Very High"},
    ],
    "DevOps Engineer": [
        {"title": "DevOps Engineer",         "emoji": "🔧", "demand": "Very High"},
        {"title": "Cloud Architect",         "emoji": "☁️", "demand": "Very High"},
        {"title": "Site Reliability Eng.",   "emoji": "📡", "demand": "High"},
        {"title": "Platform Engineer",       "emoji": "🏗️", "demand": "High"},
    ],
    "default": [
        {"title": "Software Engineer",       "emoji": "💻", "demand": "Very High"},
        {"title": "Tech Consultant",         "emoji": "🤝", "demand": "High"},
        {"title": "Project Manager",         "emoji": "📋", "demand": "High"},
        {"title": "Business Analyst",        "emoji": "📈", "demand": "High"},
    ],
}

# ── Helpers ───────────────────────────────────────────────────────
def clean_resume(text: str) -> str:
    text = re.sub(r"http\S+", " ", str(text))
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def extract_text_from_pdf(data: bytes) -> str:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)


def extract_text_from_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_skills(cleaned_text: str):
    found = [s for s in ALL_SKILLS if s in cleaned_text]
    missing = [s for s in REQUIRED_SKILLS if s not in found]
    return found, missing


def score_resume(found_skills: list) -> float:
    matched = sum(1 for s in REQUIRED_SKILLS if s in found_skills)
    return round((matched / len(REQUIRED_SKILLS)) * 100, 2)


def match_companies(found_skills: list, predicted_role: str):
    results = []
    for c in COMPANIES:
        matched = sum(1 for s in found_skills if s in c["skills"])
        pct = round((matched / len(c["skills"])) * 100, 2)
        results.append({
            "name": c["name"],
            "role": c["role"],
            "score": pct,
            "color": c["color"],
            "abbr": c["abbr"],
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]


def get_career_paths(predicted_role: str):
    return CAREER_PATHS.get(predicted_role, CAREER_PATHS["default"])


# ── Routes ────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ResumeIQ API is running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext not in ("pdf", "docx", "doc"):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Extract text
    try:
        if ext == "pdf":
            raw_text = extract_text_from_pdf(data)
        else:
            raw_text = extract_text_from_docx(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read file: {str(e)}")

    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    # ML pipeline
    cleaned = clean_resume(raw_text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)
    predicted_role = encoder.inverse_transform(prediction)[0]

    # Skills & score
    found_skills, missing_skills = extract_skills(cleaned)
    resume_score = score_resume(found_skills)
    companies = match_companies(found_skills, predicted_role)
    career_paths = get_career_paths(predicted_role)

    return {
        "predicted_role": predicted_role,
        "resume_score": resume_score,
        "skills_found": found_skills,
        "skills_missing": missing_skills,
        "companies": companies,
        "career_paths": career_paths,
        "accuracy": 99.48,
    }
