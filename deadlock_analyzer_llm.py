from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.memory import VectorStoreRetrieverMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import pandas as pd
import numpy as np
from datetime import datetime

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.2,
    groq_api_key="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"
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    collection_name="deadlock_analysis",
    embedding_function=embedding_model,
    persist_directory="./deadlock_db"
)

retriever = db.as_retriever(search_kwargs={"k": 5})
memory = VectorStoreRetrieverMemory(retriever=retriever)

response_schemas = [
    ResponseSchema(name="risk_level", description="Overall risk assessment of the system state"),
    ResponseSchema(name="analysis", description="Detailed analysis of the current system state"),
    ResponseSchema(name="prevention_strategies", description="List of prevention strategies"),
    ResponseSchema(name="optimization_suggestions", description="System optimization recommendations")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

prompt = PromptTemplate(
    input_variables=[
        "Process_ID", "Resource_ID", "Allocated", "Requested", 
        "Available", "Priority", "Wait_Time", "Total_System_Resources", 
        "Hold_and_Wait", "System_Load", "Circular_Wait_Probability", 
        "Deadlock_State", "history", "format_instructions"
    ],
    template="""
You are DeadlockGPT, an advanced AI system specializing in operating system resource management and deadlock analysis. You have extensive knowledge of distributed systems, concurrent programming, and resource allocation algorithms.

System Context:
Process_ID: {Process_ID}
Resource_ID: {Resource_ID}
Resource Metrics:
- Allocated: {Allocated}
- Requested: {Requested}
- Available: {Available}
- Total: {Total_System_Resources}

Process Characteristics:
- Priority Level: {Priority}
- Wait Time: {Wait_Time}ms
- Hold and Wait Probability: {Hold_and_Wait}
- System Load: {System_Load}
- Circular Wait Probability: {Circular_Wait_Probability}

ML Model Prediction:
Deadlock State: {Deadlock_State}

Historical Context:
{history}

Based on the above parameters, provide:

1. RISK ASSESSMENT:
   - Critical evaluation of system state
   - Resource utilization analysis
   - Bottleneck identification

2. DETAILED ANALYSIS:
   If Deadlock_State == 1:
   - Root cause analysis of potential deadlock
   - Resource allocation patterns
   - Process interaction assessment
   - Impact on system performance
   If Deadlock_State == 0:
   - Factors contributing to stability
   - Resource distribution efficiency
   - Process scheduling effectiveness

3. PREVENTION STRATEGIES:
   - Immediate actions required
   - Long-term prevention measures
   - Resource allocation optimization
   - Process priority adjustments
   - Wait time management recommendations

4. SYSTEM OPTIMIZATION:
   - Resource utilization improvements
   - Process scheduling enhancements
   - System load balancing suggestions
   - Performance optimization recommendations
   - Monitoring and alerting setup

Additional considerations:
- Resource starvation risks
- Priority inversion possibilities
- Livelock prevention
- System scalability
- Fault tolerance measures

{format_instructions}
"""
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

class DeadlockAnalyzer:
    def __init__(self, model_path="best_classification_model.pkl"):
        self.db = db
        self.chain = chain
        self.history = []
        
    def analyze_system_state(self, data):
        timestamp = datetime.now().isoformat()
        data_with_time = {**data, "timestamp": timestamp}
        vector_data = [str(v) for v in data_with_time.values()]
        self.db.add_texts([str(vector_data)])
        
        historical_context = self.db.similarity_search(str(vector_data))
        data["history"] = "\n".join([doc.page_content for doc in historical_context])
        data["format_instructions"] = output_parser.get_format_instructions()
        
        result = self.chain.run(data)
        parsed_output = output_parser.parse(result)
        
        self.history.append({
            "timestamp": timestamp,
            "input": data,
            "analysis": parsed_output
        })
        
        return parsed_output
    
    def get_historical_analysis(self, n_records=5):
        return self.history[-n_records:]
    
    def get_similar_cases(self, current_state):
        return self.db.similarity_search(str(current_state), k=5)

analyzer = DeadlockAnalyzer()

system_state = {
    "Process_ID": 1.0,
    "Resource_ID": 2.0,
    "Allocated": 50.0,
    "Requested": 30.0,
    "Available": 20.0,
    "Priority": 5.0,
    "Wait_Time": 100.0,
    "Total_System_Resources": 200.0,
    "Hold_and_Wait": 0.5,
    "System_Load": 0.7,
    "Circular_Wait_Probability": 0.3,
    "Deadlock_State": 1
}

analysis_result = analyzer.analyze_system_state(system_state)
similar_cases = analyzer.get_similar_cases(system_state)
historical_analysis = analyzer.get_historical_analysis()