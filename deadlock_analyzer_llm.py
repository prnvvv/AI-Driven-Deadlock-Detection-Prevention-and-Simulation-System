from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.memory import VectorStoreRetrieverMemory
from datetime import datetime
import json
from typing import Dict, List, Any

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0.2,
    groq_api_key="GROQ_API_KEY" 
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    collection_name="deadlock_analysis",
    embedding_function=embedding_model,
    persist_directory="./deadlock_db"
)

retriever = db.as_retriever(search_kwargs={"k": 5})
memory = VectorStoreRetrieverMemory(retriever=retriever)

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

chain = LLMChain(
    llm=llm,
    prompt=ANALYSIS_PROMPT,
    memory=memory,
    verbose=True
)

history = []

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
        print(f"Error during analysis: {str(e)}")
        return {"error": str(e)}

def get_historical_analysis(n_records: int = 5) -> List[Dict]:
    return history[-n_records:]

def get_similar_cases(current_state: Dict) -> List[Any]:
    return db.similarity_search(json.dumps(current_state), k=5)
