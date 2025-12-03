import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import time
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION & DARK THEME CSS
# ==========================================
st.set_page_config(
    page_title="PharmaGuard Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Card Styling */
    .metric-card {
        background-color: #262730;
        border: 1px solid #41444C;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    /* Success/Error text visibility */
    .stSuccess, .stWarning, .stError {
        font-weight: bold;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1F2129;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. SESSION & MODEL
# ==========================================
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=['Time', 'Batch', 'Temp', 'Hum', 'Status'])

@st.cache_resource
def load_ai_model():
    # Quick Training simulation
    df = pd.DataFrame({
        'Temp': np.random.normal(22, 5, 1000),
        'Hum': np.random.normal(45, 15, 1000),
        'Chem': np.random.uniform(0.7, 1.0, 1000),
        'Days': np.random.randint(-20, 365, 1000)
    })
    df['Label'] = df.apply(lambda x: 0 if x['Days']<0 else (1 if x['Temp']>30 or x['Hum']>70 else 2), axis=1)
    model = RandomForestClassifier()
    model.fit(df[['Temp', 'Hum', 'Chem', 'Days']], df['Label'])
    return model

model = load_ai_model()

# ==========================================
# 3. HELPER FUNCTION: GAUGE CHARTS (FIXED)
# ==========================================
def create_gauge(value, title, min_val, max_val, safe_min, safe_max, color_hex):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 20, 'color': "white"}}, # Increased font slightly for visibility
        number = {'font': {'color': "white"}},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickcolor': "white"},
            'bar': {'color': color_hex},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [min_val, safe_min], 'color': "#333"},
                {'range': [safe_max, max_val], 'color': "#551A1A"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': safe_max
            }
        }
    ))
    # FIX APPLIED HERE: Changed 't' (Top Margin) from 30 to 60 to prevent title clipping
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)", 
        font = {'color': "white"}, 
        height=250,  # Increased height slightly to accommodate the margin
        margin=dict(l=20,r=20,t=60,b=20) 
    )
    return fig

# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.title("🎛️ Control Center")
    st.info("Adjust simulated IoT sensor values below.")
    
    temp = st.slider("🌡️ Temperature (°C)", 0.0, 50.0, 22.5)
    hum = st.slider("💧 Humidity (%)", 0.0, 100.0, 45.0)
    chem = st.slider("🧪 Purity Index", 0.0, 1.0, 0.96)
    days = st.number_input("📅 Shelf Life (Days)", value=120)
    
    st.divider()
    st.caption("System v3.3 Pro | Connected")

# ==========================================
# 5. MAIN DASHBOARD LAYOUT
# ==========================================
st.markdown("## 🧬 PharmaGuard: Intelligent Quality Control")
st.markdown("Monitor real-time production line metrics using AI.")

# --- ROW 1: LIVE SENSOR GAUGES ---
st.markdown("### 📡 Live Telemetry")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    # Temperature Gauge
    color = "#00FF00" if temp <= 30 else "#FF4B4B"
    st.plotly_chart(create_gauge(temp, "Temperature (°C)", 0, 50, 0, 30, color), use_container_width=True)

with col2:
    # Humidity Gauge
    color = "#00FF00" if hum <= 70 else "#FF4B4B"
    st.plotly_chart(create_gauge(hum, "Humidity (%)", 0, 100, 0, 70, color), use_container_width=True)

with col3:
    # Chemical Purity metric card
    st.markdown(f"""
    <div class="metric-card" style="text-align: center; height: 250px;">
        <h3 style="margin:0; color: #aaa; font-size: 16px;">Chemical Purity Index</h3>
        <h1 style="font-size: 3em; color: {'#00CC96' if chem > 0.85 else '#EF553B'}; margin: 10px 0;">
            {chem*100:.1f}%
        </h1>
        <p style="color: #666; margin:0;">Threshold: >85%</p>
    </div>
    """, unsafe_allow_html=True)

# --- ROW 2: SENSOR TRENDS GRAPH ---
st.divider()
st.markdown("### 📈 Sensor Trend Analysis (Last 20 Mins)")
# Generate fake trend data centered around current slider values
chart_data = pd.DataFrame({
    'Temperature (°C)': np.random.normal(temp, 2, 20),
    'Humidity (%)': np.random.normal(hum, 5, 20)
})
st.line_chart(chart_data, height=250)

# --- ROW 3: AI PREDICTION SECTION ---
st.divider()
col_pred, col_ai = st.columns([1, 1])

with col_pred:
    st.subheader("🧠 Model Diagnosis")
    if st.button("RUN AI DIAGNOSTIC", type="primary", use_container_width=True):
        with st.spinner("Analyzing molecular structure..."):
            time.sleep(1.2)
            
            # Prediction Logic
            input_df = pd.DataFrame([[temp, hum, chem, days]], columns=['Temp', 'Hum', 'Chem', 'Days'])
            pred = model.predict(input_df)[0]
            prob = np.max(model.predict_proba(input_df)) * 100
            
            # Display Logic
            if pred == 2: # Good
                st.success("## ✅ BATCH APPROVED")
                st.write(f"Confidence Score: **{prob:.2f}%**")
                msg = "System Status: **OPTIMAL**"
                status_txt = "Approved"
            elif pred == 1: # Substandard
                st.warning("## ⚠️ QUALITY WARNING")
                st.write(f"Confidence Score: **{prob:.2f}%**")
                msg = "System Status: **CHECK SENSORS**"
                status_txt = "Substandard"
            else: # Expired
                st.error("## ❌ BATCH REJECTED")
                st.write(f"Confidence Score: **{prob:.2f}%**")
                msg = "Reason: **PRODUCT EXPIRED**"
                status_txt = "Expired"
            
            # Save to History
            new_log = {
                'Time': datetime.now().strftime("%H:%M:%S"),
                'Batch': f"#{np.random.randint(1000,9999)}",
                'Temp': f"{temp:.1f}°C",
                'Hum': f"{hum:.1f}%",
                'Status': status_txt
            }
            st.session_state['history'] = pd.concat([pd.DataFrame([new_log]), st.session_state['history']], ignore_index=True)
            
            # Show Analysis Message
            st.markdown(f"""
            <div style="background-color: #262730; padding: 15px; border-radius: 5px; border-left: 5px solid white;">
                {msg}
            </div>
            """, unsafe_allow_html=True)

with col_ai:
    st.subheader("🤖 AI Insights")
    st.markdown("""
    The AI analyzes inputs against ISO pharmaceutical standards.
    
    * **Temperature Impact:** >30°C degrades active ingredients.
    * **Humidity Impact:** >70% causes tablet dissolution.
    * **Purity:** Must remain above 85% for efficacy.
    """)
    if not st.session_state['history'].empty:
         last_status = st.session_state['history'].iloc[-1]['Status']
         if last_status == "Approved":
             st.info("Observation: Current parameters are within safe ISO limits.")
         else:
             st.error("Action Plan: Halt production line. Check HVAC systems immediately.")

# --- DATA LOGS ---
st.divider()
st.subheader("📋 Production Logs (Exportable)")
if not st.session_state['history'].empty:
    st.dataframe(st.session_state['history'], use_container_width=True)
else:
    st.caption("No batch data recorded yet. Run a diagnostic to generate logs.")