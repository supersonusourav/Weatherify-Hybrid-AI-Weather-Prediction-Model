---
title: Weatherify Hybrid AI
emoji: 🌤️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
license: mit
---

# 🌤️ Weatherify: Hybrid AI Weather Prediction System

**Weatherify** is an advanced meteorological tool that merges classical machine learning with Generative AI (RAG) to provide high-precision temperature forecasts and scientific climatological insights.

---

## 🎯 Objective: Why a Hybrid Model?
Standard weather models often suffer from a "context gap." Classical models provide numerical data without explanation, while LLMs provide context but are prone to hallucinating numbers.

**Weatherify solves this by:**
* **Numerical Precision:** Utilizing a **Random Forest Regressor** to predict baseline temperatures based on geographical coordinates and timestamps.
* **Physics-Informed Correction:** Applying sinusoidal modeling to account for diurnal (day/night) temperature fluctuations.
* **Scientific Reasoning:** Integrating **RAG (Retrieval-Augmented Generation)** to fetch real-time climatology context from Wikipedia, allowing the AI to explain the *thermodynamic factors* behind the predicted temperature.

---

## 🧪 Methodology
The system operates through a three-layer architecture:

### 1. The Predictive Engine (Scikit-Learn)
* **Model:** Random Forest Regressor.
* **Feature Engineering:** Months, Latitudes, and Longitudes are normalized via `StandardScaler`.
* **Heuristic Layer:** A diurnal correction formula $temp + (4 \times \cos((hour - 14) \times \pi / 12))$ is applied to simulate solar peak cycles.

### 2. The Knowledge Layer (ChromaDB + LangChain)
* **Vector Store:** **ChromaDB** stores vectorized climatological documents.
* **Embeddings:** `HuggingFaceEmbeddings` (all-MiniLM-L6-v2) for semantic search.
* **Data Source:** Dynamic Wikipedia scraping for the latest meteorological research.

### 3. The Cognitive Layer (Google Gemini)
* **LLM:** **Gemini 3 Flash** acts as the analyst.
* **Chain:** A Retrieval Chain synthesizes the numerical output from the ML model with the text context from the Vector DB to produce a comprehensive report.

---

## ⚙️ Setup and Run

### Prerequisites
* Python 3.9+
* A Google AI Studio API Key (for Gemini)

### 1. Clone the Repository
```
git clone [https://github.com/supersonusourav/Weatherify-Hybrid-AI-Weather-Prediction-Model.git](https://github.com/supersonusourav/Weatherify-Hybrid-AI-Weather-Prediction-Model.git)
cd Weatherify-Hybrid-AI-Weather-Prediction-Model
```
---
### 2. Install Dependencies
```
pip install -r requirements.txt
```
---
### 3. Configure Environment
Create a .env file in the root directory:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
---
### 4. Run the Application
```
python app.py
```
---
### 🚀 Future Scope

Multi-Variate Expansion: Incorporate humidity, barometric pressure, and wind speed into the Random Forest training set.
Satellite Integration: Use NASA/NOAA API hooks to feed real-time cloud cover data into the RAG pipeline.
Edge Deployment: Quantize the Random Forest model for execution on IoT devices for remote weather stations.
Predictive Mapping: Integrate Folium or Mapbox to visualize temperature gradients across regions dynamically.

---
### 🛠️ Tech Stack

Language: Python
ML: Scikit-Learn, Joblib, NumPy, Pandas
GenAI: LangChain, Google Gemini 3 Flash
Database: ChromaDB (Vector Search)
UI: Gradio 6.0 / Streamlit
Data Engineering: Git LFS (Large File Storage)

---
## Developed with ❤️ by Sonu Sourav © 2026 Weatherify Intelligence Systems
