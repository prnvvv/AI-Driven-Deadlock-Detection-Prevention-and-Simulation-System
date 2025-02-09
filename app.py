import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_model():
    try:
        return joblib.load('best_classification_model.pkl')
    except:
        st.error("Please ensure the model file 'best_classification_model.pkl' is in the same directory")
        return None

def create_feature_input():
    st.sidebar.header("Process Parameters")
    input_data = {
        'Process_ID': st.sidebar.number_input("Process ID", min_value=0.0, max_value=100.0, value=1.0),
        'Resource_ID': st.sidebar.number_input("Resource ID", min_value=0.0, max_value=100.0, value=1.0),
        'Allocated': st.sidebar.number_input("Allocated Resources", min_value=0.0, max_value=100.0, value=50.0),
        'Requested': st.sidebar.number_input("Requested Resources", min_value=0.0, max_value=100.0, value=30.0),
        'Available': st.sidebar.number_input("Available Resources", min_value=0.0, max_value=100.0, value=20.0),
        'Priority': st.sidebar.slider("Process Priority", min_value=0.0, max_value=10.0, value=5.0),
        'Wait_Time': st.sidebar.number_input("Wait Time (ms)", min_value=0.0, max_value=1000.0, value=100.0),
        'Total_System_Resources': st.sidebar.number_input("Total System Resources", min_value=0.0, max_value=1000.0, value=200.0),
        'Hold_and_Wait': st.sidebar.slider("Hold and Wait Probability", min_value=0.0, max_value=1.0, value=0.5),
        'System_Load': st.sidebar.slider("System Load", min_value=0.0, max_value=1.0, value=0.7),
        'Circular_Wait_Probability': st.sidebar.slider("Circular Wait Probability", min_value=0.0, max_value=1.0, value=0.3)
    }
    return input_data

def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data

def visualize_system_state(input_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    resources = ['Allocated', 'Requested', 'Available']
    values = [input_data['Allocated'], input_data['Requested'], input_data['Available']]
    ax1.bar(resources, values)
    ax1.set_title('Resource Distribution')
    ax1.set_ylabel('Resource Units')
    probs = ['Hold_and_Wait', 'System_Load', 'Circular_Wait_Probability']
    prob_values = [input_data['Hold_and_Wait'], input_data['System_Load'], input_data['Circular_Wait_Probability']]
    ax2.bar(probs, prob_values)
    ax2.set_title('System Probabilities')
    ax2.set_ylabel('Probability')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def main():
    st.title("Interactive Deadlock Detector")
    st.write("This application helps predict potential deadlock situations based on system parameters.")
    model = load_model()
    if model is None:
        return
    input_data = create_feature_input()
    if st.sidebar.button("Predict Deadlock"):
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)
        st.header("Prediction Results")
        if prediction[0] == 1:
            st.error("⚠️ Warning: Deadlock Likely!")
            st.write(f"Probability of Deadlock: {probability[0][1]:.2%}")
        else:
            st.success("✅ System State: Safe")
            st.write(f"Probability of Safe State: {probability[0][0]:.2%}")
        st.header("System State Visualization")
        visualize_system_state(input_data)
        if hasattr(model, 'feature_importances_'):
            st.header("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': list(input_data.keys()),
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            st.bar_chart(feature_importance.set_index('Feature'))

if __name__ == "__main__":
    main()
