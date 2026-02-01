import os
import gc
import joblib
import datetime
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# --- 1. INITIALIZATION & PAGE CONFIG ---
st.set_page_config(page_title="Weatherify AI", page_icon="🌤️", layout="wide")
load_dotenv()

DB_PATH = "./weather_db"
API_KEY = st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")

def cleanup_memory():
    gc.collect()

# --- 2. DATA & MODEL LOADING (CACHED) ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('weather_model.joblib')
        scaler = joblib.load('scaler.joblib')
        coords_df = pd.read_csv("GlobalLandTemperaturesByCity.csv", 
                               usecols=['City', 'Latitude', 'Longitude']).drop_duplicates('City')
        coords_map = coords_df.set_index('City').T.to_dict()
        return model, scaler, coords_map
    except Exception as e:
        st.error(f"Initialization Error: {e}. Check if .joblib and .csv files exist.")
        return None, None, None

MODEL, SCALER, COORDS_MAP = load_resources()

# --- 3. THE INTELLIGENCE LAYER (RAG) ---
from langchain_community.document_loaders import WikipediaLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

@st.cache_resource
def initialize_rag():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        loader = WikipediaLoader(query="Climatology and Thermodynamics", load_max_docs=1)
        docs = loader.load()
        vector_db = Chroma.from_documents(docs, embeddings, persist_directory=DB_PATH)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=API_KEY) # Updated to 1.5-flash
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are 'Weatherify', an AI weather analyst. Use context to explain thermodynamics. Context: {context}"),
        ("human", "{input}")
    ])
    return create_retrieval_chain(vector_db.as_retriever(), create_stuff_documents_chain(llm, prompt))

RAG_CHAIN = initialize_rag()

# --- 4. ANALYTICS LOGIC ---
def predict_weather(city_name, date_val, time_val, user_query):
    try:
        city = city_name.strip().title()
        if city not in COORDS_MAP:
            return None, f"### 🏙️ Record Missing\n'{city}' is not in our database."

        dt = datetime.datetime.combine(date_val, time_val)
        info = COORDS_MAP[city]
        
        def cln(v): return float(v[:-1]) if v[-1] in ['N', 'E'] else -float(v[:-1])
        lat, lon = cln(info['Latitude']), cln(info['Longitude'])
        
        feat = SCALER.transform(np.array([[dt.month, lat, lon]]))
        temp = MODEL.predict(feat)[0] + (4 * np.cos((dt.hour - 14) * np.pi / 12))
        
        report = f"### 📊 Analysis for {city}\n"
        report += f"The **Random Forest Regressor** predicts a baseline of {temp:.1f}°C."

        if user_query:
            with st.spinner("AI Analyst is thinking..."):
                ai_res = RAG_CHAIN.invoke({"input": f"Analyze {temp:.1f}C for {city} in {dt.strftime('%B')}. {user_query}"})
                report += f"\n\n---\n### 🔍 Weatherify AI Insight\n{ai_res['answer']}"
        
        return f"{temp:.1f}°C", report
    except Exception as e:
        return "Error", f"### ❌ System Alert\n{str(e)}"

# --- 5. STREAMLIT UI DESIGN ---
st.markdown("""
    <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
        <h1 style='margin: 0; color: #00BFFF;'>🌤️ Weatherify</h1>
        <p style='margin: 0; font-style: italic; opacity: 0.7;'>Hybrid Intelligence Predictive System</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📍 Configuration")
    loc_in = st.text_input("Target City", placeholder="e.g. London")
    date_in = st.date_input("Forecast Date", datetime.date.today())
    time_in = st.time_input("Forecast Time", datetime.time(12, 0))
    query_in = st.text_area("Weatherify Inquiry", placeholder="Ask about trends or anomalies...")
    run_btn = st.button("🚀 Run Analytics", use_container_width=True)
    
    st.divider()
    with st.expander("📖 About The Project"):
        st.markdown("""
            **Developer:** Sonu Sourav | **© 2026**
            - **Models:** Scikit-Learn + Gemini 3
            - **Data:** ChromaDB + Wikipedia RAG
        """)

# Main Display Area
col1, col2 = st.columns([1, 2])

if run_btn:
    if loc_in:
        temp_res, report_res = predict_weather(loc_in, date_in, time_in, query_in)
        with col1:
            st.metric("Predicted Temperature", temp_res)
        with col2:
            st.markdown(report_res)
    else:
        st.warning("Please enter a city name.")
else:
    col2.info("### Weatherify Standby\nEnter details in the sidebar to initialize analysis.")

cleanup_memory()
