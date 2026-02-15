import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

# --- Page Config ---
st.set_page_config(
    page_title="AI4I Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Design ---
# --- Custom CSS for Premium Design (Light Mode) ---
st.markdown("""
    <style>
    /* Main Background & Font */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
        color: #1f2937;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #111827;
        font-weight: 700;
    }
    
    .main-header {
        font-size: 3rem; 
        background: linear-gradient(90deg, #0056b3 0%, #00a8e8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    /* Cards */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s;
        border: 1px solid #e5e7eb;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
        border-color: #007bff;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #0056b3;
    }
    .metric-label {
        font-size: 1rem;
        color: #6b7280;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0056b3 0%, #007bff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0, 86, 179, 0.2);
    }
    
    /* Tables */
    div[data-testid="stDataFrame"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Input Containers */
    .input-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Models & Data ---
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    df = pd.read_csv(url)
    # Basic cleanup for display
    df.rename(columns={'Air temperature [K]': 'Air_Temp', 'Process temperature [K]': 'Process_Temp', 
                       'Rotational speed [rpm]': 'RPM', 'Torque [Nm]': 'Torque', 'Tool wear [min]': 'Tool_Wear',
                       'Machine failure': 'Target'}, inplace=True)
    return df

@st.cache_resource
def load_assets():
    models = {}
    algo_names = [
        "logistic_regression", "decision_tree", "knn", 
        "naive_bayes", "random_forest", "xgboost"
    ]
    # Check if model folder exists
    if not os.path.exists('model'):
        st.error("Model directory not found. Please run model_training.py first.")
        return None, None, None

    for name in algo_names:
        try:
            models[name] = joblib.load(f"model/{name}.pkl")
        except:
            pass # Handle missing models gracefully
            
    try:
        scaler = joblib.load("model/scaler.pkl")
        le = joblib.load("model/label_encoder.pkl")
    except:
        st.error("Scaler or Label Encoder not found.")
        scaler, le = None, None
        
    return models, scaler, le

models, scaler, le = load_assets()
df_full = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🚀 Model Predictor", "📊 Model Comparison", "📄 Dataset Summary"])

st.sidebar.markdown("---")
st.sidebar.info("System Engineer - Predictive Maintenance Module")

# --- Helper Logic ---
def get_user_input():
    col1, col2 = st.columns(2)
    with col1:
        type_val = st.selectbox("Machine Type", ["L", "M", "H"])
        air_temp = st.slider("Air Temperature [K]", 290.0, 310.0, 300.0)
        process_temp = st.slider("Process Temperature [K]", 300.0, 320.0, 310.0)
    with col2:
        rpm = st.number_input("Rotational Speed [rpm]", 1000, 3000, 1500)
        torque = st.number_input("Torque [Nm]", 10.0, 80.0, 40.0)
        tool_wear = st.number_input("Tool Wear [min]", 0, 300, 0)
    
    data = {
        'Type': type_val,
        'Air_Temp': air_temp,
        'Process_Temp': process_temp,
        'RPM': rpm,
        'Torque': torque,
        'Tool_Wear': tool_wear
    }
    return pd.DataFrame([data])

# --- Page: Model Predictor ---
if page == "🚀 Model Predictor":
    st.markdown('<h1 class="main-header">Predictive Maintenance Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Predict machine failure probability using real-time sensor data.")
    
    st.markdown("### 🛠️ Input Parameters")
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        input_df = get_user_input()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🤖 Select Model")
    model_choice = st.selectbox("Choose a Classification Model", list(models.keys()))

    # --- Single Prediction Button ---
    if st.button("Run Prediction (Single Item)", use_container_width=True):
        if models and scaler and le:
            # Preprocess Single Input
            input_proc = input_df.copy()
            input_proc['Type'] = le.transform(input_proc['Type'])
            input_scaled = scaler.transform(input_proc)
            
            model = models[model_choice]
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else 0.0

            st.write("---")
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.markdown("#### Prediction Status")
                if prediction == 1:
                    st.error("⚠️ FAILURE PREDICTED")
                    st.markdown(f"**Confidence:** {probability:.2%}")
                else:
                    st.success("✅ NO FAILURE")
                    st.markdown(f"**Confidence:** {1-probability:.2%}")
            
            with c2:
                # Gauge Chart for Risk
                fig = px.pie(
                    names=["Safe", "Risk"], values=[1-probability, probability],
                    color_discrete_sequence=['#10b981', '#dc2626'],
                    hole=0.7, title="Failure Risk Probability"
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#1f2937")
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("Models not loaded correctly.")

    # --- Batch Prediction Section ---
    st.write("---")
    st.markdown("### 📂 Batch Prediction (Optional)")
    
    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_file = st.file_uploader("Upload CSV (Test Data)", type=["csv"])
    with col_up2:
        st.write("") # Spacer
        st.write("") 
        use_example = st.button("Use Test Set 📄", help="Load the example test_data.csv")

    df_batch = None
    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
    elif use_example:
        if os.path.exists("test_data.csv"):
            df_batch = pd.read_csv("test_data.csv")
            st.success("Loaded example 'test_data.csv' successfully!")
        else:
            st.error("Example 'test_data.csv' not found. Please train models first.")

    if df_batch is not None:
        try:
            # Expected columns for model
            expected_cols = ['Type', 'Air_Temp', 'Process_Temp', 'RPM', 'Torque', 'Tool_Wear']
            
            # Check if columns exist
            if not all(col in df_batch.columns for col in expected_cols):
                 st.error(f"Dataset must contain columns: {expected_cols}")
            else:
                # Prepare data for prediction
                df_proc = df_batch[expected_cols].copy()
                
                # Handle 'Type' column encoding
                # Check if 'Type' is string (needs encoding) or already numeric
                if df_proc['Type'].dtype == 'object':
                    try:
                        df_proc['Type'] = le.transform(df_proc['Type'])
                    except Exception as e:
                        st.error(f"Error encoding 'Type' column: {e}. Ensure values are L, M, H.")
                        st.stop()
                
                # Scale features
                try:
                    X_batch_scaled = scaler.transform(df_proc)
                except Exception as e:
                    st.error(f"Error scaling data: {e}")
                    st.stop()
                
                # Predict
                model = models[model_choice]
                predictions = model.predict(X_batch_scaled)
                probabilities = model.predict_proba(X_batch_scaled)[:, 1] if hasattr(model, "predict_proba") else [0]*len(predictions)
                
                # Create Result DataFrame
                df_results = df_batch.copy()
                df_results['Failure_Prediction'] = predictions
                df_results['Failure_Prediction'] = df_results['Failure_Prediction'].map({0: 'No Failure', 1: 'Failure'})
                df_results['Failure_Probability'] = probabilities
                
                # Display Summary
                st.markdown(f"### 📊 Batch Prediction Results ({len(df_results)} rows)")
                
                # Color code the dataframe
                def highlight_failure(row):
                    color = '#fecaca' if row['Failure_Prediction'] == 'Failure' else '#d1fae5'
                    return [f'background-color: {color}' for _ in row]

                st.dataframe(df_results.style.apply(highlight_failure, axis=1), use_container_width=True)
                
                # Download Button
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name='prediction_results.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"Error reading file: {e}")

# --- Page: Model Comparison ---
elif page == "📊 Model Comparison":
    st.markdown('<h1 class="main-header">Model Performance Evaluation</h1>', unsafe_allow_html=True)
    
    if os.path.exists("model_evaluation_metrics.csv"):
        results_df = pd.read_csv("model_evaluation_metrics.csv")
        
        # Best Model Highlight
        best_model = results_df.loc[results_df['F1 Score'].idxmax()]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Best Performing Model</div>
                <div class="metric-value" style="font-size: 1.5rem;">{best_model['Model']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Best Accuracy</div>
                <div class="metric-value">{results_df['Accuracy'].max()}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Best AUC Score</div>
                <div class="metric-value">{results_df['AUC'].max()}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("---")
        
        # All-in-One Performance Chart
        st.subheader("Comprehensive Model Comparison")
        
        # Melt dataframe to long format for grouped bar chart
        df_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        
        fig = px.bar(df_melted, x="Model", y="Score", color="Metric", barmode='group',
                     title="All Models - All Metrics Comparison",
                     color_discrete_sequence=px.colors.qualitative.Prism)
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font_color="#1f2937",
            legend_title="Evaluation Metrics",
            xaxis_title="Machine Learning Model",
            yaxis_title="Score (0-1)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Table
        st.subheader("Detailed Metrics Table")
        # Style table: Highlight max
        st.dataframe(results_df.style.highlight_max(axis=0, color='#dbeafe'), use_container_width=True)
        
        # Observations
        st.subheader("📝 Observations")
        st.info("""
        - **Random Forest & XGBoost** perform consistently high across all metrics (F1 > 0.70).
        - **Logistic Regression & Naive Bayes** show significant drops in Recall, indicating they miss many failures.
        - **XGBoost** is the recommended model for deployment.
        """)

        # --- Confusion Matrix Section ---
        st.write("---")
        st.subheader("📉 Confusion Matrix Analysis")
        
        cm_model_choice = st.selectbox("Select Model for Confusion Matrix", list(models.keys()), key="cm_select")
        
        if models and os.path.exists('model/X_test.pkl'):
            try:
                # Load Test Data
                X_test = joblib.load('model/X_test.pkl')
                y_test = joblib.load('model/y_test.pkl')
                
                # Predict
                model = models[cm_model_choice]
                y_pred = model.predict(X_test)
                
                # Generate Confusion Matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Plot using Plotly
                fig_cm = px.imshow(cm, 
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=['No Failure', 'Failure'],
                                y=['No Failure', 'Failure'],
                                text_auto=True,
                                color_continuous_scale='Blues')
                
                fig_cm.update_layout(
                    title=f"Confusion Matrix: {cm_model_choice}",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#1f2937"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                
            except Exception as e:
                st.error(f"Could not load test data for visualization: {e}")
        else:
            st.warning("Test data not found. Please retrain models.")
    else:
        st.warning("No evaluation metrics found. Please train models first.")

# --- Page: Dataset Summary ---
elif page == "📄 Dataset Summary":
    st.markdown('<h1 class="main-header">Dataset Overview</h1>', unsafe_allow_html=True)
    st.markdown("### UCI AI4I 2020 Predictive Maintenance Dataset")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Rows:** {df_full.shape[0]}")
        st.markdown(f"**Columns:** {df_full.shape[1]}")
    with col2:
        st.markdown("**Target Variable:** `Machine failure` (0: No, 1: Yes)")
    
    # Snapshot
    st.subheader("🔍 Data Snapshot (First 5 Rows)")
    st.dataframe(df_full.head(), use_container_width=True)
    
    # Distribution of Target
    st.subheader("📊 Target Distribution (Class Imbalance)")
    target_counts = df_full['Target'].value_counts().reset_index()
    target_counts.columns = ['Status', 'Count']
    target_counts['Status'] = target_counts['Status'].map({0: 'No Failure', 1: 'Failure'})
    
    fig_dist = px.pie(target_counts, names='Status', values='Count', 
                      title="Failure vs No Failure Distribution",
                      color='Status',
                      color_discrete_map={'No Failure': '#10b981', 'Failure': '#dc2626'})
    fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#1f2937")
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Statistical Summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(df_full.describe(), use_container_width=True)
