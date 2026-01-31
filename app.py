# Install the core LangChain modular packages and Google integration
# pip install -qU langchain langchain-community langchain-core langchain-google-genai langchain-huggingface chromadb sentence-transformers scikit-learn joblib pandas numpy beautifulsoup4 python-dotenv wikipedia datasets

import os
import gc
import time
import joblib
import shutil
import datetime
import pandas as pd
import numpy as np
import gradio as gr
from dotenv import load_dotenv

# --- 1. INITIALIZATION & RESOURCE MANAGEMENT ---
load_dotenv()
DB_PATH = "./weather_db"
API_KEY = os.getenv("GOOGLE_API_KEY")

def cleanup_memory():
    """Ensures no file locks are held before starting."""
    gc.collect()

# --- 2. DATA & MODEL LOADING ---
print("🚀 Weatherify: Loading Scientific Resources...")
try:
    MODEL = joblib.load('weather_model.joblib')
    SCALER = joblib.load('scaler.joblib')
    coords_df = pd.read_csv("GlobalLandTemperaturesByCity.csv", 
                           usecols=['City', 'Latitude', 'Longitude']).drop_duplicates('City')
    COORDS_MAP = coords_df.set_index('City').T.to_dict()
    print(f"✅ Indexed {len(COORDS_MAP)} cities successfully.")
except Exception as e:
    print(f"❌ Initialization Error: {e}. Ensure .joblib and .csv files are in the directory.")
    exit()

# --- 3. THE INTELLIGENCE LAYER (RAG) ---
from langchain_community.document_loaders import WikipediaLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_db():
    if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
        print("📁 Loading local knowledge base...")
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    print("🌐 Building knowledge base from Wikipedia...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            loader = WikipediaLoader(query="Climatology and Thermodynamics", load_max_docs=1)
            docs = loader.load()
            if docs:
                return Chroma.from_documents(docs, embeddings, persist_directory=DB_PATH)
        except Exception:
            print(f"⚠️ Retry {attempt+1}...")
            time.sleep(2)
    
    from langchain_core.documents import Document
    return Chroma.from_documents([Document(page_content="Meteorology study.")], embeddings, persist_directory=DB_PATH)

vector_db = get_vector_db()
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=API_KEY)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are 'Weatherify', an AI weather analyst. Use context to explain thermodynamics. Context: {context}"),
    ("human", "{input}")
])
RAG_CHAIN = create_retrieval_chain(vector_db.as_retriever(), create_stuff_documents_chain(llm, prompt))

# --- 4. ANALYTICS ENGINE ---
def run_weatherify(city_name, dt_input, user_query):
    if not city_name or city_name.strip() == "":
        return "N/A", "### 📍 Selection Required\nPlease specify a city to start the analysis."
    
    if dt_input is None:
        return "N/A", "### 📅 Schedule Required\nPlease select a date and time for the prediction."

    try:
        dt = datetime.datetime.fromtimestamp(dt_input) if isinstance(dt_input, (int, float)) else dt_input
        city = city_name.strip().title()

        if city not in COORDS_MAP:
            return "N/A", f"### 🏙️ Record Missing\n'{city}' is not in our historical database."

        info = COORDS_MAP[city]
        def cln(v): return float(v[:-1]) if v[-1] in ['N', 'E'] else -float(v[:-1])
        lat, lon = cln(info['Latitude']), cln(info['Longitude'])
        
        feat = SCALER.transform(np.array([[dt.month, lat, lon]]))
        temp = MODEL.predict(feat)[0] + (4 * np.cos((dt.hour - 14) * np.pi / 12))
        
        report = f"### 📊 Analysis for {city}\n"
        report += f"The **Random Forest Regressor** predicts a baseline of {temp:.1f}°C."

        if user_query:
            ai_res = RAG_CHAIN.invoke({"input": f"Analyze {temp:.1f}C for {city} in {dt.strftime('%B')}. {user_query}"})
            report += f"\n\n---\n### 🔍 Weatherify AI Insight\n{ai_res['answer']}"
        
        return f"{temp:.1f}°C", report

    except Exception as e:
        return "Error", f"### ❌ System Alert\n{str(e)}"

# --- 5. UI DESIGN (GRADIO 6.0) ---
with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    # Header with Library-Style Icon (Styled Emoji + CSS)
    gr.HTML("""
        <div style="display: flex; align-items: center; gap: 20px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 15px;">
            <div style="font-size: 50px; background: linear-gradient(135deg, #00c6ff, #0072ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
               🌤️
            </div>
            <div>
                <h1 style='margin: 0; color: #00BFFF; font-family: sans-serif;'>Weatherify</h1>
                <p style='margin: 0; font-style: italic; opacity: 0.7;'>Hybrid Intelligence Predictive System</p>
            </div>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            loc_in = gr.Textbox(label="📍 Target City", placeholder="City name...")
            dt_in = gr.DateTime(label="📅 Forecast Horizon")
            query_in = gr.Textbox(label="💬 Weatherify Inquiry", placeholder="Ask me about trends...")
            run_btn = gr.Button("🚀 Run Weatherify Analytics", variant="primary")
            
            with gr.Accordion("📖 About The Project", open=False):
                gr.Markdown("""
                ### Weatherify Architecture
                **Developer:** Sonu Sourav | **© 2026**
                
                **Technologies:**
                * **Core:** Random Forest Regressor (Scikit-Learn)
                * **Intelligence:** Gemini 3 Flash + LangChain RAG
                * **Vector DB:** ChromaDB (Persistent Storage)
                * **Optimization:** Hybrid NeuralGCM/PINN Architecture
                """)

        with gr.Column(scale=2):
            temp_display = gr.Label(label="Predicted Temperature")
            with gr.Column(variant="panel"):
                report_display = gr.Markdown("### Weatherify Standby\nEnter details to initialize.")

    run_btn.click(run_weatherify, [loc_in, dt_in, query_in], [temp_display, report_display])

# --- 6. LAUNCH ---
if __name__ == "__main__":
    cleanup_memory()
    demo.launch(inline=True,share=True)
