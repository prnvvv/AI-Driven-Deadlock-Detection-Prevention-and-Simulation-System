import streamlit as st
import pandas as pd
import numpy as np
import joblib
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import time
from plotly.subplots import make_subplots
import threading
import queue

# Configure Streamlit page
st.set_page_config(
    page_title="Deadlock Nexus",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Neon Yellow/Green Theme CSS
st.markdown("""
<style>
    :root {
        --primary-color: #ccff00;
        --secondary-color: #00ff00;
        --background-color: #0a0a0a;
        --text-color: #ffffff;
    }

    .stApp {
        background: linear-gradient(45deg, var(--background-color), #1a1a1a);
        color: var(--text-color);
    }
    
    .css-1d391kg {
        background: linear-gradient(180deg, var(--background-color), #1a1a1a);
    }
    
    h1, h2, h3 {
        color: var(--primary-color) !important;
        text-shadow: 0 0 10px rgba(204, 255, 0, 0.7);
    }
    
    .stTextInput input, .stNumberInput input {
        background-color: rgba(10, 10, 10, 0.7) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--primary-color) !important;
        border-radius: 5px !important;
    }
    
    .stButton button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color)) !important;
        color: var(--background-color) !important;
        font-weight: bold !important;
        border: none !important;
        box-shadow: 0 0 10px rgba(204, 255, 0, 0.5);
    }
    
    .css-1r6slb0 {
        background: rgba(10, 10, 10, 0.7);
        border: 1px solid var(--primary-color);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(204, 255, 0, 0.2);
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(10, 10, 10, 0.7);
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--primary-color);
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(204, 255, 0, 0.2);
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = []
if 'update_count' not in st.session_state:
    st.session_state.update_count = 0
if 'current_params' not in st.session_state:
    st.session_state.current_params = None
if 'max_simulation_points' not in st.session_state:
    st.session_state.max_simulation_points = 50

# Initialize LLM components
@st.cache_resource
def initialize_llm():
    try:
        llm = ChatGroq(
            model="deepseek-r1-distill-llama-70b",
            temperature=0.2,
            groq_api_key="GROQ_API_KEY"
            
        )
        memory = ConversationBufferMemory()
        return llm, memory
    except Exception as e:
        st.error(f"Error initializing LLM components: {str(e)}")
        return None, None

# Load ML model
@st.cache_resource
def load_ml_model():
    try:
        return joblib.load('best_classification_model.pkl')
    except Exception as e:
        st.error(f"Error loading ML model: {str(e)}")
        return None

# Preprocessing Function
def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    ss = StandardScaler()
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = ss.fit_transform(df[numeric_cols])
    return df

# Create time series plots with deadlock indicators
def create_time_series_subplot(history_data):
    if not history_data:
        return None
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Resource Allocation Over Time', 
                       'Wait Time Trend',
                       'System Parameters'),
        vertical_spacing=0.12
    )
    
    df = pd.DataFrame(history_data)
    
    # Add traces with improved styling
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], 
            y=df['Allocated'],
            name='Allocated', 
            line=dict(color='#ccff00', width=2),
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], 
            y=df['Requested'],
            name='Requested', 
            line=dict(color='#00ff00', width=2),
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], 
            y=df['Available'],
            name='Available', 
            line=dict(color='#ffff00', width=2),
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], 
            y=df['Wait_Time'],
            name='Wait Time', 
            line=dict(color='#ccff00', width=2),
            mode='lines+markers'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], 
            y=df['System_Load'],
            name='System Load', 
            line=dict(color='#00ff00', width=2),
            mode='lines+markers'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccff00'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='#ccff00',
            borderwidth=1
        )
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(204, 255, 0, 0.1)',
        showline=True,
        linewidth=2,
        linecolor='#ccff00'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(204, 255, 0, 0.1)',
        showline=True,
        linewidth=2,
        linecolor='#ccff00'
    )
    
    return fig

# Deadlock detection function
def check_deadlock(allocated, requested, available):
    """Simple deadlock detection logic"""
    if requested > available and allocated > 0:
        return True
    return False

# Simulation data generation with deadlock detection
def generate_simulation_data(base_params):
    t = st.session_state.update_count
    noise = np.sin(t/10) * 0.1
    
    new_state = {
        'timestamp': datetime.now(),
        'Allocated': base_params['Allocated'] * (1 + noise + np.random.normal(0, 0.05)),
        'Requested': base_params['Requested'] * (1 + noise + np.random.normal(0, 0.05)),
        'Available': base_params['Available'] * (1 - noise + np.random.normal(0, 0.05)),
        'System_Load': base_params['System_Load'] * (1 + noise),
        'Wait_Time': base_params['Wait_Time'] * (1 + np.random.normal(0, 0.1))
    }
    
    # Check for deadlock
    deadlock_detected = check_deadlock(
        new_state['Allocated'],
        new_state['Requested'],
        new_state['Available']
    )
    
    new_state['deadlock'] = deadlock_detected
    return new_state

# Analysis prompt template
ANALYSIS_PROMPT = """
You are an expert in operating systems and deadlock management. Analyze the following system state:

System Parameters:
Process ID: {process_id}
Resource ID: {resource_id}
Allocated Resources: {allocated}
Requested Resources: {requested}
Available Resources: {available}
Process Priority: {priority}
Wait Time: {wait_time} ms
Total System Resources: {total_resources}
Hold and Wait Probability: {hold_wait}
System Load: {system_load}
Circular Wait Probability: {circular_wait}
Deadlock State Prediction: {deadlock_state}

Provide a comprehensive analysis of the system's deadlock potential and actionable recommendations.
"""

# Initialize components
llm, memory = initialize_llm()
ml_model = load_ml_model()

# Main app layout
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1>🌐 Deadlock Nexus: Advanced System Analysis</h1>
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["System Analysis", "Simulation", "History Visualization"])

# System Analysis Tab
with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        process_id = st.number_input("Process ID", min_value=1, value=1, step=1)
        resource_id = st.number_input("Resource ID", min_value=1, value=1, step=1)
        allocated = st.number_input("Allocated Resources", min_value=0, value=50, step=1)
        wait_time = st.number_input("Wait Time (ms)", min_value=0, value=100, step=1)

    with col2:
        requested = st.number_input("Requested Resources", min_value=0, value=30, step=1)
        available = st.number_input("Available Resources", min_value=0, value=20, step=1)
        priority = st.number_input("Process Priority", min_value=1, value=5, step=1)
        total_resources = st.number_input("Total System Resources", min_value=0, value=200, step=1)

    with col3:
        hold_wait = st.slider("Hold and Wait Probability", 0.0, 1.0, 0.5)
        system_load = st.slider("System Load", 0.0, 1.0, 0.7)
        circular_wait = st.slider("Circular Wait Probability", 0.0, 1.0, 0.3)

    # Store current parameters
    st.session_state.current_params = {
        "Process_ID": process_id,
        "Resource_ID": resource_id,
        "Allocated": allocated,
        "Requested": requested,
        "Available": available,
        "Priority": priority,
        "Wait_Time": wait_time,
        "Total_System_Resources": total_resources,
        "Hold_and_Wait": hold_wait,
        "System_Load": system_load,
        "Circular_Wait_Probability": circular_wait
    }

    # Analysis button
    if st.button("🔍 Analyze System State", key="analyze_button"):
        if not ml_model or not llm:
            st.error("Error: ML model or LLM not properly initialized")
        else:
            with st.spinner("Analyzing system state..."):
                try:
                    # Prepare input data
                    input_data = st.session_state.current_params
                    
                    # Preprocess and predict
                    X_pred = preprocess_input(input_data)
                    deadlock_prediction = ml_model.predict(X_pred)[0]
                    
                    # Get LLM analysis
                    prompt = ANALYSIS_PROMPT.format(
                        process_id=process_id,
                        resource_id=resource_id,
                        allocated=allocated,
                        requested=requested,
                        available=available,
                        priority=priority,
                        wait_time=wait_time,
                        total_resources=total_resources,
                        hold_wait=hold_wait,
                        system_load=system_load,
                        circular_wait=circular_wait,
                        deadlock_state=int(deadlock_prediction)
                    )
                    
                    analysis_result = llm.invoke(prompt).content
                    
                    # Add to history
                    history_entry = {
                        "timestamp": datetime.now(),
                        "input": input_data,
                        "analysis": analysis_result,
                        "deadlock_state": deadlock_prediction
                    }
                    st.session_state.analysis_history.append(history_entry)
                    
                    # Display results
                    st.markdown("### Analysis Results")
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        if deadlock_prediction == 1:
                            st.error("⚠ Potential Deadlock Detected!")
                        else:
                            st.success("✅ System is in a Safe State")
                        
                        # System load gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=system_load * 100,
                            title={'text': "System Load"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#ccff00"},
                                'steps': [
                                    {'range': [0, 50], 'color': "rgba(204, 255, 0, 0.2)"},
                                    {'range': [50, 75], 'color': "rgba(204, 255, 0, 0.5)"},
                                    {'range': [75, 100], 'color': "rgba(204, 255, 0, 0.6)"}
                                ]
                            }
                        ))
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#ccff00"})
                        st.plotly_chart(fig)
                    
                    with result_col2:
                        st.info(analysis_result)
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

# Simulation Tab
with tab2:
    st.subheader("Real-time System Simulation")
    
    # Simulation controls
    control_col1, control_col2, control_col3 = st.columns([1, 2, 1])
    
    with control_col1:
        max_points = st.number_input(
            "Maximum Simulation Points",
            min_value=10,
            max_value=200,
            value=st.session_state.max_simulation_points,
            step=10
        )
        st.session_state.max_simulation_points = max_points
        
        simulation_speed = st.slider(
            "Simulation Speed",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Lower value = faster simulation"
        )
    
    with control_col2:
        if not st.session_state.simulation_running:
            if st.button("▶ Start Simulation"):
                if st.session_state.current_params:
                    st.session_state.simulation_running = True
                    st.session_state.simulation_data = []
                    st.session_state.update_count = 0
                else:
                    st.warning("Please set system parameters in the Analysis tab first")
        else:
            if st.button("⏹ Stop Simulation"):
                st.session_state.simulation_running = False
    
    # Simulation visualization
    if st.session_state.simulation_running and st.session_state.current_params:
        plot_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Check if simulation should continue
        if st.session_state.update_count >= st.session_state.max_simulation_points:
            st.session_state.simulation_running = False
            st.warning(f"Simulation completed after {st.session_state.max_simulation_points} points")
        else:
            # Generate new data point
            new_data = generate_simulation_data(st.session_state.current_params)
            st.session_state.simulation_data.append(new_data)
            st.session_state.update_count += 1
            
            # Keep only the most recent points
            if len(st.session_state.simulation_data) > st.session_state.max_simulation_points:
                st.session_state.simulation_data = st.session_state.simulation_data[-st.session_state.max_simulation_points:]
            
            # Update visualization
            fig = create_time_series_subplot(st.session_state.simulation_data)
            if fig:
                plot_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Show current metrics
            with metrics_placeholder.container():
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Allocated Resources", f"{new_data['Allocated']:.2f}")
                with metric_col2:
                    st.metric("Requested Resources", f"{new_data['Requested']:.2f}")
                with metric_col3:
                    st.metric("Available Resources", f"{new_data['Available']:.2f}")
                with metric_col4:
                    st.metric("System Load", f"{new_data['System_Load']:.2%}")
                
                if new_data.get('deadlock', False):
                    st.error("⚠ Deadlock Detected!")
                    
            # Control simulation speed
            time.sleep(simulation_speed)
            st.rerun()

# History Visualization Tab
with tab3:
    if st.session_state.analysis_history:
        st.subheader("Historical Analysis")
        
        # Time series visualization
        history_data = [{
            'timestamp': entry['timestamp'],
            'Allocated': entry['input']['Allocated'],
            'Requested': entry['input']['Requested'],
            'Available': entry['input']['Available'],
            'Wait_Time': entry['input']['Wait_Time'],
            'System_Load': entry['input']['System_Load']
        } for entry in st.session_state.analysis_history]
        
        # Display time series visualization
        fig = create_time_series_subplot(history_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # 3D visualization of resource states
        st.subheader("3D Resource State Visualization")
        df_3d = pd.DataFrame([{
            'Allocated': entry['input']['Allocated'],
            'Requested': entry['input']['Requested'],
            'Available': entry['input']['Available'],
            'Deadlock_State': entry['deadlock_state']
        } for entry in st.session_state.analysis_history])
        
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=df_3d['Allocated'],
            y=df_3d['Requested'],
            z=df_3d['Available'],
            mode='markers',
            marker=dict(
                size=10,
                color=df_3d['Deadlock_State'],
                colorscale=[[0, '#ccff00'], [1, '#ff0000']],
                opacity=0.8
            )
        )])
        
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Allocated Resources',
                yaxis_title='Requested Resources',
                zaxis_title='Available Resources',
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "#ccff00"},
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # History table
        st.subheader("Analysis History")
        history_df = pd.DataFrame([{
            'Timestamp': entry['timestamp'],
            'Deadlock Status': 'Detected' if entry['deadlock_state'] == 1 else 'Safe',
            'System Load': f"{entry['input']['System_Load']:.2%}",
            'Wait Time': f"{entry['input']['Wait_Time']} ms"
        } for entry in st.session_state.analysis_history])
        
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Run some analyses to see visualization history!")

# Cleanup function
def cleanup():
    if st.session_state.simulation_running:
        st.session_state.simulation_running = False

# Register cleanup handler
st.session_state['cleanup_registered'] = True

