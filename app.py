import os
import gc
import time
import joblib
import datetime
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# --- 1. INITIALIZATION & UI CONFIG ---
st.set_page_config(page_title="Weatherify AI", page_icon="🌤️", layout="wide")
load_dotenv()

# --- SIDEBAR: API KEY & CONFIGURATION ---
with st.sidebar:
    st.markdown("### 🔑 Authentication")
    # Waterfall check for API Key: .env -> Secrets -> Manual Input
    default_key = os.getenv("GOOGLE_API_KEY") or ""
    try:
        if not default_key:
            default_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass

    API_KEY = st.text_input("Google API Key", value=default_key, type="password")
    
    if not API_KEY:
        st.warning("Please enter your Google API Key to enable AI features.")

    st.divider()
    st.header("📍 Target Parameters")
    city_input = st.text_input("City Name", placeholder="e.g. London")
    
    col_d, col_t = st.columns(2)
    date_input = col_d.date_input("Date", datetime.date.today())
    time_input = col_t.time_input("Time", datetime.time(12, 0))
    
    query_input = st.text_area("Weatherify Inquiry", placeholder="Ask about climate trends...")
    run_btn = st.button("🚀 Run Weatherify Analytics", use_container_width=True, type="primary")

DB_PATH = "./weather_db"

def cleanup_memory():
    gc.collect()

# --- 2. DATA & MODEL LOADING ---
@st.cache_resource
def load_scientific_resources():
    try:
        model = joblib.load('weather_model.joblib')
        scaler = joblib.load('scaler.joblib')
        # Optimized CSV loading
        coords_df = pd.read_csv("GlobalLandTemperaturesByCity.csv", 
                               usecols=['City', 'Latitude', 'Longitude']).drop_duplicates('City')
        coords_map = coords_df.set_index('City').T.to_dict()
        return model, scaler, coords_map
    except Exception as e:
        st.error(f"❌ Initialization Error: {e}")
        return None, None, None

MODEL, SCALER, COORDS_MAP = load_scientific_resources()

# --- 3. THE INTELLIGENCE LAYER (LCEL - Bypass Broken Modules) ---
from langchain_community.document_loaders import WikipediaLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

@st.cache_resource
def initialize_rag_engine(_api_key):
    if not _api_key:
        return None
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if local DB exists, otherwise fetch from Wiki
    if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        with st.spinner("🌐 Building knowledge base..."):
            loader = WikipediaLoader(query="Climatology and Thermodynamics", load_max_docs=1)
            docs = loader.load()
            vector_db = Chroma.from_documents(docs, embeddings, persist_directory=DB_PATH)
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=_api_key)
    
    # Define LCEL Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are 'Weatherify', an AI weather analyst. Use the provided context to answer the inquiry.
    
    Context: {context}
    
    Question: {question}
    
    Answer as a scientific expert:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # The LCEL Chain: Input -> Context Retrieval -> Prompt -> LLM -> String
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- 4. ANALYTICS ENGINE ---
def run_weatherify(city_name, date_val, time_val, user_query):
    try:
        city = city_name.strip().title()
        if city not in COORDS_MAP:
            return None, f"### 🏙️ Record Missing\n'{city}' is not in our historical database."

        dt = datetime.datetime.combine(date_val, time_val)
        info = COORDS_MAP[city]
        
        # Parse Lat/Lon
        def cln(v): return float(v[:-1]) if v[-1] in ['N', 'E'] else -float(v[:-1])
        lat, lon = cln(info['Latitude']), cln(info['Longitude'])
        
        # Predict
        feat = SCALER.transform(np.array([[dt.month, lat, lon]]))
        temp = MODEL.predict(feat)[0] + (4 * np.cos((dt.hour - 14) * np.pi / 12))
        
        report = f"### 📊 Analysis for {city}\n"
        report += f"The **Random Forest Regressor** predicts a baseline of **{temp:.1f}°C**."

        # RAG Logic
        if user_query and API_KEY:
            rag_chain = initialize_rag_engine(API_KEY)
            with st.spinner("🔍 Weatherify AI is synthesizing insight..."):
                query_str = f"Analyze a predicted temperature of {temp:.1f}C for {city} in {dt.strftime('%B')}. {user_query}"
                ai_answer = rag_chain.invoke(query_str)
                report += f"\n\n---\n### 🔍 Weatherify AI Insight\n{ai_answer}"
        elif user_query and not API_KEY:
            report += "\n\n⚠️ *AI Insight skipped: No API Key provided in sidebar.*"
        
        return f"{temp:.1f}°C", report
    except Exception as e:
        return "Error", f"### ❌ System Alert\n{str(e)}"

# --- 5. UI DESIGN ---
st.markdown("""
    <div style="display: flex; align-items: center; gap: 20px; padding: 15px; background: rgba(0, 198, 255, 0.1); border-radius: 15px; margin-bottom: 25px;">
        <div style="font-size: 50px;">🌤️</div>
        <div>
            <h1 style='margin: 0; color: #00BFFF;'>Weatherify</h1>
            <p style='margin: 0; font-style: italic; opacity: 0.7;'>Hybrid Intelligence Predictive System</p>
        </div>
    </div>
""", unsafe_allow_html=True)

main_col1, main_col2 = st.columns([1, 2])

if run_btn:
    if city_input:
        temp_out, report_out = run_weatherify(city_input, date_input, time_input, query_input)
        with main_col1:
            st.metric(label="Predicted Temperature", value=temp_out)
        with main_col2:
            st.markdown(report_out)
    else:
        st.warning("Please specify a city in the sidebar.")
else:
    main_col2.info("### 📡 System Standby\nEnter location and schedule in the sidebar to initialize analytics.")

with st.sidebar.expander("📖 About Architecture"):
    st.markdown("""
        **Developer:** Sonu Sourav | **© 2026**
        * **Core:** Random Forest (Scikit-Learn)
        * **AI:** Gemini 1.5 Flash + LCEL RAG
        * **Vector DB:** ChromaDB (Wiki-Context)
    """)

if __name__ == "__main__":
    cleanup_memory()
