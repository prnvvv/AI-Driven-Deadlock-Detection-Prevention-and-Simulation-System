import streamlit as st
from typing import Dict, List, Any
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import plotly.express as px

load_dotenv("GROQ_API_KEY.env")
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    collection_name="deadlock_analysis",
    embedding_function=embeddings,
    persist_directory="./deadlock_db"
)

history = []

ANALYSIS_PROMPT = PromptTemplate(
    input_variables=[
        "Process_ID", "Resource_ID", "Allocated", "Requested", 
        "Available", "Priority", "Wait_Time", "Total_System_Resources", 
        "Hold_and_Wait", "System_Load", "Circular_Wait_Probability", "Deadlock_State"
    ],
    template='''
You are an expert in operating systems and deadlock management. Analyze the system state based on the given parameters:
Process ID: {Process_ID}
Resource ID: {Resource_ID}
Allocated Resources: {Allocated}
Requested Resources: {Requested}
Available Resources: {Available}
Process Priority: {Priority}
Wait Time: {Wait_Time} ms
Total System Resources: {Total_System_Resources}
Hold and Wait Probability: {Hold_and_Wait}
System Load: {System_Load}
Circular Wait Probability: {Circular_Wait_Probability}
Deadlock State Prediction: {Deadlock_State}
Provide an analysis and recommendations based on this information.'''
)

model = RandomForestClassifier()
X_train = np.array([
    [50, 30, 20, 5, 100, 200, 0.5, 0.7, 0.3],
    [70, 50, 10, 8, 200, 150, 0.8, 0.9, 0.7]
])
y_train = np.array([0, 1])
model.fit(X_train, y_train)

def predict_deadlock_state(features: List[float]) -> int:
    return int(model.predict([features])[0])

def analyze_system_state(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        timestamp = datetime.now().isoformat()
        prompt_text = ANALYSIS_PROMPT.format(**data)
        result = llm.invoke(prompt_text)
        analysis_record = {
            "timestamp": timestamp,
            "input": data,
            "analysis": result.content
        }
        history.append(analysis_record)
        return analysis_record
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return {"error": str(e)}

def get_historical_analysis(n_records: int = 5) -> List[Dict]:
    return history[-n_records:]

def get_similar_cases(current_state: Dict) -> List[Any]:
    return db.similarity_search(json.dumps(current_state), k=5)

def main():
    st.title("Deadlock Analysis System")
    st.sidebar.header("System State Input")
    Process_ID = st.sidebar.number_input("Process ID", min_value=1, step=1, value=1)
    Resource_ID = st.sidebar.number_input("Resource ID", min_value=1, step=1, value=2)
    Allocated = st.sidebar.number_input("Allocated Resources", min_value=0.0, step=1.0, value=50.0)
    Requested = st.sidebar.number_input("Requested Resources", min_value=0.0, step=1.0, value=30.0)
    Available = st.sidebar.number_input("Available Resources", min_value=0.0, step=1.0, value=20.0)
    Priority = st.sidebar.number_input("Process Priority", min_value=0.0, step=1.0, value=5.0)
    Wait_Time = st.sidebar.number_input("Wait Time (ms)", min_value=0.0, step=1.0, value=100.0)
    Total_System_Resources = st.sidebar.number_input("Total System Resources", min_value=0.0, step=1.0, value=200.0)
    Hold_and_Wait = st.sidebar.number_input("Hold and Wait Probability", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
    System_Load = st.sidebar.number_input("System Load", min_value=0.0, max_value=1.0, step=0.1, value=0.7)
    Circular_Wait_Probability = st.sidebar.number_input("Circular Wait Probability", min_value=0.0, max_value=1.0, step=0.1, value=0.3)
    features = [Allocated, Requested, Available, Priority, Wait_Time, Total_System_Resources, Hold_and_Wait, System_Load, Circular_Wait_Probability]
    if st.button("Predict Deadlock State and Analyze"):
        with st.spinner("Predicting and Analyzing..."):
            Deadlock_State = predict_deadlock_state(features)
            st.write(f"Predicted Deadlock State: {'Deadlock' if Deadlock_State == 1 else 'Safe State'}")
            system_state = {
                "Process_ID": Process_ID,
                "Resource_ID": Resource_ID,
                "Allocated": Allocated,
                "Requested": Requested,
                "Available": Available,
                "Priority": Priority,
                "Wait_Time": Wait_Time,
                "Total_System_Resources": Total_System_Resources,
                "Hold_and_Wait": Hold_and_Wait,
                "System_Load": System_Load,
                "Circular_Wait_Probability": Circular_Wait_Probability,
                "Deadlock_State": Deadlock_State
            }
            analysis_result = analyze_system_state(system_state)
            st.subheader("Analysis Result")
            st.json(analysis_result)
            st.subheader("Historical Analysis")
            historical_analysis = pd.DataFrame(get_historical_analysis())
            if not historical_analysis.empty:
                st.dataframe(historical_analysis)
                st.plotly_chart(px.line(historical_analysis, x="timestamp", y="analysis", title="Historical Analysis Over Time"))
            else:
                st.write("No historical analysis available.")
            st.subheader("Similar Cases")
            similar_cases = get_similar_cases(system_state)
            st.write(similar_cases)
            input_data = pd.DataFrame([system_state])
            st.subheader("Input Data Summary")
            st.dataframe(input_data)

if __name__ == "__main__":
    main()
