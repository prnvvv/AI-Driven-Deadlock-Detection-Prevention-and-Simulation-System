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
You are an expert in operating systems and deadlock management, with deep knowledge of resource allocation algorithms and concurrency control. Your task is to provide meaningful and actionable insights to help prevent and resolve deadlocks in computing systems.  
The input provided includes the following parameters and a machine learning prediction on the system's state:
**Input Parameters:**  
- Process ID: {Process_ID}  
- Resource ID: {Resource_ID}  
- Allocated Resources: {Allocated}  
- Requested Resources: {Requested}  
- Available Resources: {Available}  
- Process Priority: {Priority}  
- Wait Time: {Wait_Time} ms  
- Total System Resources: {Total_System_Resources}  
- Hold and Wait Probability: {Hold_and_Wait}  
- System Load: {System_Load}  
- Circular Wait Probability: {Circular_Wait_Probability}  
- Deadlock State Prediction: {Deadlock_State} (1 indicates a potential deadlock, 0 indicates a safe state)
**Your Instructions:**  
- If the **Deadlock_State** is `1`, assume a **potential deadlock** and provide the following:  
  - A **detailed analysis** of why the system is likely in a deadlock state, referencing specific input parameters such as **high Circular_Wait_Probability**, **excessive resource allocation**, or **system load**.  
  - Suggest at least **three prevention or resolution strategies** specific to this scenario. Consider algorithms like the **Banker's Algorithm**, resource preemption, and priority adjustments.  
  - Offer **alternative resource allocation strategies** or system tuning suggestions to avoid future deadlocks.  
  - Provide **insights on system efficiency** and any misconfigurations observed.
- If the **Deadlock_State** is `0`, assume the system is in a **safe state**, and provide the following:  
  - A **confirmation of system stability**, highlighting positive factors such as low system load or effective resource allocation.  
  - Suggest **best practices for maintaining this state**, considering load balancing or priority tuning strategies.  
  - Offer **predictive recommendations** based on input parameters (e.g., if resource requests increase or circular wait probability rises).
- Always provide scenario-based learning insights that help users understand how each factor (like Priority, Available Resources, or Circular_Wait_Probability) influences system stability and how these can be optimized.
'''
)

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

# Streamlit UI
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
    Deadlock_State = st.sidebar.radio("Deadlock State", options=[0, 1], index=1)

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

    if st.button("Analyze System State"):
        with st.spinner("Analyzing..."):
            analysis_result = analyze_system_state(system_state)
            st.subheader("Analysis Result")
            st.json(analysis_result)

            st.subheader("Similar Cases")
            similar_cases = get_similar_cases(system_state)
            st.json(similar_cases)

            st.subheader("Historical Analysis")
            historical_analysis = get_historical_analysis()
            st.json(historical_analysis)

if __name__ == "__main__":
    main()
