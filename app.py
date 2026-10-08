import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import plotly.express as px
import plotly.graph_objects as go
import io
import os
import json

# Import custom modules
from src.data_manager import save_prediction, get_prediction_history, search_history, clear_history
from src.data_preprocessing import DEPARTMENTS
from src.prediction import PlacementPredictor
from src.explainability import PlacementExplainer

# Set page configuration
st.set_page_config(
    page_title="NextCareer AI - Placement Prediction",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for login & theme
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'light'
if 'landing_view' not in st.session_state:
    st.session_state.landing_view = 'home'

def render_login_page():
    # Inject full screen glassmorphism background CSS and Landing styling
    login_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp {
            background-color: #0B0F19 !important;
            background-image: 
                radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.15) 0%, rgba(11, 15, 25, 0) 60%),
                linear-gradient(0deg, rgba(255, 255, 255, 0.008) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.008) 1px, transparent 1px) !important;
            background-size: 100% 100%, 40px 40px, 40px 40px !important;
            background-position: center top !important;
        }
        
        [data-testid="stHeader"], [data-testid="stSidebar"], header[data-testid="stHeader"] {
            display: none !important;
            height: 0px !important;
            min-height: 0px !important;
        }
        
        .block-container,
        [data-testid="stAppViewBlockContainer"],
        .main .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 1rem !important;
            margin-top: 0rem !important;
        }
        
        .brand-hero {
            padding: 10px;
            color: #FFFFFF;
            text-align: center;
        }
        
        .brand-badge {
            background: rgba(99, 102, 241, 0.15);
            color: #818CF8;
            border: 1px solid rgba(99, 102, 241, 0.3);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 25px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-family: 'Inter', sans-serif;
        }
        
        .brand-title {
            font-family: 'Outfit', sans-serif;
            font-size: 56px;
            font-weight: 700;
            line-height: 1.15;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -1.5px;
        }
        
        .brand-subtitle {
            font-family: 'Outfit', sans-serif;
            font-size: 24px;
            color: #818CF8;
            margin-bottom: 20px;
            font-weight: 500;
        }
        
        .brand-desc {
            font-family: 'Inter', sans-serif;
            font-size: 16px;
            color: #94A3B8;
            max-width: 720px;
            margin: 0 auto 35px auto;
            line-height: 1.6;
        }
        
        .feature-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s ease;
            backdrop-filter: blur(12px);
            height: 100%;
        }
        
        .feature-card:hover {
            border-color: rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
            box-shadow: 0 12px 30px -10px rgba(99, 102, 241, 0.2);
        }
        
        .feature-item {
            display: flex;
            align-items: flex-start;
            gap: 16px;
        }
        
        .feature-icon {
            font-size: 24px;
            background: rgba(99, 102, 241, 0.1);
            border-radius: 12px;
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        
        .feature-text strong {
            display: block;
            color: #F8FAFC;
            font-size: 17px;
            font-family: 'Outfit', sans-serif;
            margin-bottom: 6px;
        }
        
        .feature-text p {
            color: #94A3B8;
            font-size: 14px;
            margin: 0;
            line-height: 1.5;
        }
        
        .stats-banner {
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 20px 40px;
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin-top: 30px;
        }
        
        .stat-box {
            text-align: center;
        }
        
        .stat-value {
            font-family: 'Outfit', sans-serif;
            font-size: 32px;
            font-weight: 700;
            color: #818CF8;
        }
        
        .stat-label {
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }
    </style>
    """
    st.markdown(login_css, unsafe_allow_html=True)
    
    # 1. Header Navigation Bar
    col_logo, col_action = st.columns([5, 1])
    with col_logo:
        st.markdown("<h3 style='margin: 0; color: #FFFFFF; font-family: \"Outfit\", sans-serif; font-weight: 700; letter-spacing: -0.5px;'>NextCareer AI</h3>", unsafe_allow_html=True)
    with col_action:
        if st.session_state.landing_view == 'home':
            if st.button("Sign In", use_container_width=True, type="secondary"):
                st.session_state.landing_view = 'login'
                st.rerun()
        else:
            if st.button("Back Home", use_container_width=True, type="secondary"):
                st.session_state.landing_view = 'home'
                st.rerun()
                
    st.write("---")
    
    # 2. Main Stateful Router Layout
    if st.session_state.landing_view == 'home':
        # Hero Branding Page
        st.markdown("""
        <div class="brand-hero">
            <div class="brand-badge">AI-POWERED PLACEMENT INTELLIGENCE</div>
            <div class="brand-title">Predict. Prepare. Elevate.</div>
            <div class="brand-subtitle">Placement Insights & ML Prediction Engine</div>
            <div class="brand-desc" style="text-align: center !important;">
                NextCareer AI evaluates student academic performance, coding scores, and technical skills to forecast placement outcomes, 
                estimate expected salary packages, and deliver visual SHAP XAI feature attributions.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Access Portal Center Button
        _, col_center_btn, _ = st.columns([1.3, 1, 1.3])
        with col_center_btn:
            if st.button("Get Started / Enter Portal", use_container_width=True, type="primary"):
                st.session_state.landing_view = 'login'
                st.rerun()
                
        st.write("")
        st.write("")
        st.write("")
        
        # Core Capabilities Section (4 Cards layout)
        st.markdown("<h3 style='text-align: center; color: #FFFFFF; font-family: \"Outfit\", sans-serif; margin-bottom: 30px;'>Core Capabilities</h3>", unsafe_allow_html=True)
        
        col_f1, col_f2 = st.columns(2, gap="medium")
        with col_f1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-item">
                    <div class="feature-icon" style="color: #6366F1; font-weight: 700; font-family: 'Outfit', sans-serif;">01</div>
                    <div class="feature-text">
                        <strong>Placement Classifier</strong>
                        <p>Forecast placement status and probability percentage based on academic and technical parameters.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
                
        with col_f2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-item">
                    <div class="feature-icon" style="color: #6366F1; font-weight: 700; font-family: 'Outfit', sans-serif;">02</div>
                    <div class="feature-text">
                        <strong>Salary Regressor</strong>
                        <p>Estimate expected salary packages in LPA with 95% Confidence Interval bounds.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
                
        st.write("")
        
        col_f3, col_f4 = st.columns(2, gap="medium")
        with col_f3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-item">
                    <div class="feature-icon" style="color: #6366F1; font-weight: 700; font-family: 'Outfit', sans-serif;">03</div>
                    <div class="feature-text">
                        <strong>Explainable AI (SHAP)</strong>
                        <p>Visualize exact feature contributions and waterfall attributions driving prediction outcomes.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
                
        with col_f4:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-item">
                    <div class="feature-icon" style="color: #6366F1; font-weight: 700; font-family: 'Outfit', sans-serif;">04</div>
                    <div class="feature-text">
                        <strong>Institutional Analytics</strong>
                        <p>Analyze department-wise placement metrics, score correlations, and ML model performance metrics.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.write("")
        
        # Corporate metrics footer banner
        st.markdown("""
        <div class="stats-banner" style="justify-content: center; text-align: center; max-width: 800px; margin: 40px auto 0 auto;">
            <div class="stat-box">
                <div class="stat-value">94.2%</div>
                <div class="stat-label">Placement Success</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">1,000+</div>
                <div class="stat-label">Evaluated Records</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">95.8%</div>
                <div class="stat-label">Model Accuracy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.write("")
        
    else:
        # Centered Login Card
        st.write("")
        st.write("")
        _, col_login, _ = st.columns([1, 1.3, 1])
        
        with col_login:
            with st.container(border=True):
                st.markdown("<h3 style='text-align: center; color: #FFFFFF; font-family: \"Outfit\", sans-serif; margin-bottom: 5px; margin-top: 0;'>Welcome Back</h3>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 13px; margin-bottom: 25px;'>Secure Access Portal</p>", unsafe_allow_html=True)
                
                role = st.selectbox("Select Account Role", ["Student", "Admin"])
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                st.write("")
                login_btn = st.button("Access Account Portal", use_container_width=True, type="primary")
                
                if login_btn:
                    if role == "Admin":
                        if username == "admin" and password == "admin123":
                            st.session_state.logged_in = True
                            st.session_state.user_role = "admin"
                            st.success("Access Granted! Loading Admin Workspace...")
                            st.rerun()
                        else:
                            st.error("Invalid Administrator Credentials.")
                    else:
                        if username == "student" and password == "student123":
                            st.session_state.logged_in = True
                            st.session_state.user_role = "student"
                            st.success("Access Granted! Loading Student Workspace...")
                            st.rerun()
                        else:
                            st.error("Invalid Student Credentials.")
                            
                st.write("---")
                st.markdown("""
                <div style='text-align: center; color: #94A3B8; font-size: 12.5px;'>
                    <b>Development Sandbox Demo Accounts</b><br/>
                    Admin: <code>admin</code> / <code>admin123</code><br/>
                    Student: <code>student</code> / <code>student123</code>
                </div>
                """, unsafe_allow_html=True)

if not st.session_state.logged_in:
    render_login_page()
    st.stop()

# -------------------------------------------------------------
# Custom Theme Styling (Light / Dark Mode Styles)
# -------------------------------------------------------------
if st.session_state.theme_mode == 'dark':
    theme_css = """
    <style>
        .stApp {
            background-color: #0B0F19 !important;
            color: #F8FAFC !important;
        }
        .metric-card {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
            text-align: center;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #818CF8;
        }
        .metric-label {
            font-size: 14px;
            color: #94A3B8;
        }
    </style>
    """
else:
    theme_css = """
    <style>
        .stApp {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
        }
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            text-align: center;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #4F46E5;
        }
        .metric-label {
            font-size: 14px;
            color: #64748B;
        }
    </style>
    """
st.markdown(theme_css, unsafe_allow_html=True)

def toggle_theme():
    if st.session_state.theme_mode == 'light':
        st.session_state.theme_mode = 'dark'
    else:
        st.session_state.theme_mode = 'light'

# Sidebar Setup
with st.sidebar:
    st.title("🌌 NextCareer AI")
    st.markdown("Placement Intelligence Suite")
    st.divider()
    
    # Theme toggle switch
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.write(f"Theme: **{st.session_state.theme_mode.capitalize()} Mode**")
    with col_t2:
        st.button("🌙" if st.session_state.theme_mode == 'light' else "☀️", on_click=toggle_theme, key="theme_toggle_btn")
        
    st.write(f"Logged in as: **{st.session_state.user_role.capitalize()}**")
    st.divider()
    
    if st.session_state.user_role == 'admin':
        nav_options = [
            "Dashboard Analytics",
            "Predict Placement",
            "History & Data Management"
        ]
    else:
        nav_options = [
            "Placement Prediction",
            "History & Data Management"
        ]
        
    nav_selection = st.radio("Navigation Menu", options=nav_options)
    
    routing_selection = nav_selection
    if "Placement" in nav_selection:
        routing_selection = "Predict Placement"
        
    st.divider()
    if st.button("Sign Out", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.session_state.landing_view = 'home'
        st.rerun()

# Asset Loaders (Cached)
@st.cache_resource
def get_predictor_assets():
    return PlacementPredictor()

@st.cache_resource
def get_explainer_assets(_predictor):
    return PlacementExplainer(_predictor)

predictor = get_predictor_assets()
explainer = get_explainer_assets(predictor)

@st.cache_data
def load_dataset():
    if os.path.exists("Placement_Data_Enriched.csv"):
        return pd.read_csv("Placement_Data_Enriched.csv")
    return None

raw_data = load_dataset()

# ==========================================
# MODULE 1: Institutional Dashboard Analytics
# ==========================================
if routing_selection == "Dashboard Analytics":
    st.title("📊 Institutional Placement Analytics")
    st.markdown("High-level institutional insights, department distributions, score correlations, and model metrics.")
    
    if raw_data is None:
        st.warning("Enriched dataset not found. Please run `generate_data.py` to populate data.")
    else:
        # Overview KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        total_students = len(raw_data)
        placed_count = (raw_data['PlacementStatus'] == 'Placed').sum()
        placement_rate = (placed_count / total_students) * 100 if total_students > 0 else 0
        avg_cgpa = raw_data['CGPA'].mean()
        avg_coding = raw_data['Coding Score'].mean() if 'Coding Score' in raw_data.columns else 0.0
        
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_students}</div><div class="metric-label">Total Evaluated Students</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{placement_rate:.1f}%</div><div class="metric-label">Institutional Placement Rate</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_cgpa:.2f}</div><div class="metric-label">Average CGPA Benchmark</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_coding:.1f}</div><div class="metric-label">Avg Coding Score</div></div>', unsafe_allow_html=True)
            
        st.write("")
        st.write("")
        
        # Placement Rate by Department
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Department Placement Rates")
            dept_stats = raw_data.groupby('Department')['PlacementStatus'].apply(lambda x: (x == 'Placed').mean() * 100).reset_index()
            dept_stats.columns = ['Department', 'Placement Rate (%)']
            fig_dept = px.bar(
                dept_stats, x='Department', y='Placement Rate (%)',
                color='Placement Rate (%)',
                color_continuous_scale='Blues',
                text_auto='.1f'
            )
            fig_dept.update_layout(xaxis_title="", yaxis_title="Placement Rate (%)", height=380)
            st.plotly_chart(fig_dept, use_container_width=True)
            
        with col_right:
            st.subheader("Academic CGPA vs. Coding Score")
            fig_scatter = px.scatter(
                raw_data, x='CGPA', y='Coding Score',
                color='PlacementStatus',
                color_discrete_map={'Placed': '#10B981', 'NotPlaced': '#EF4444'},
                hover_data=['Name', 'Department'],
                opacity=0.7
            )
            fig_scatter.update_layout(height=380)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.subheader("Correlation Matrix Across Key Performance Indicators")
        numeric_df = raw_data.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# MODULE 2: Student Placement Prediction
# ==========================================
elif routing_selection == "Predict Placement":
    st.title("🔮 Placement Prediction & AI Insights")
    st.markdown("Input student parameters to forecast placement outcomes, calculate salary packages, and render visual SHAP attribution charts.")
    
    if not predictor.is_ready():
        st.error("🤖 Machine Learning Models are not loaded. Please ensure the pre-trained model files exist in the 'models/' directory.")
    else:
        # Form for student profile input
        with st.form("student_profile_form"):
            st.subheader("Student Academic & Professional Parameters")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                name = st.text_input("Student Name", value="Aarav Sharma")
                age = st.number_input("Age", min_value=18, max_value=30, value=21, step=1)
                department = st.selectbox("Academic Department", options=DEPARTMENTS)
                cgpa = st.slider("Cumulative CGPA (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=7.8, step=0.05)
                attendance = st.slider("Attendance Rate (%)", min_value=40, max_value=100, value=82, step=1)
                
            with col2:
                aptitude = st.slider("Aptitude Test Score (0 - 100)", min_value=0, max_value=100, value=75, step=1)
                coding = st.slider("Coding Challenge Score (0 - 100)", min_value=0, max_value=100, value=72, step=1)
                communication = st.slider("Communication Score (0 - 100)", min_value=0, max_value=100, value=80, step=1)
                technical_interview = st.slider("Technical Interview Score (0 - 100)", min_value=0, max_value=100, value=70, step=1)
                soft_skills = st.slider("Soft Skills Rating (1.0 - 5.0)", min_value=1.0, max_value=5.0, value=3.8, step=0.1)
                
            with col3:
                internships = st.number_input("Internships Completed", min_value=0, max_value=4, value=1, step=1)
                certifications = st.number_input("Technical Certifications", min_value=0, max_value=10, value=2, step=1)
                projects = st.number_input("Projects Completed", min_value=0, max_value=10, value=2, step=1)
                
                languages = st.multiselect(
                    "Programming Languages Known",
                    options=["Python", "Java", "C++", "JavaScript", "SQL", "Go", "Kotlin", "Swift"],
                    default=["Python", "SQL", "Java"]
                )
                
                extra_curricular = st.radio("Active in Extra Curricular Activities?", options=["Yes", "No"], index=0, horizontal=True)
                
            submit_profile = st.form_submit_button("🔮 Run Diagnostic Analysis", use_container_width=True)
            
        if submit_profile:
            student_dict = {
                'name': name,
                'age': age,
                'department': department,
                'cgpa': cgpa,
                'attendance': attendance,
                'aptitude': aptitude,
                'coding': coding,
                'communication': communication,
                'technical_interview': technical_interview,
                'soft_skills': soft_skills,
                'internships': internships,
                'certifications': certifications,
                'projects': projects,
                'languages': languages,
                'extra_curricular': extra_curricular
            }
            
            # Predict outcome
            pred_results = predictor.predict_student(student_dict)
            
            if 'error' in pred_results:
                st.error(pred_results['error'])
            else:
                status = pred_results['placement_status']
                prob = pred_results['placement_probability']
                conf = pred_results['confidence_score']
                salary = pred_results['expected_salary']
                sal_min = pred_results['salary_range_min']
                sal_max = pred_results['salary_range_max']
                
                # Save prediction to history database
                save_prediction(student_dict, status, prob, salary)
                
                st.write("---")
                st.subheader(f"Placement Intelligence Diagnostic Report: {name}")
                
                # Output KPI Cards
                col_res1, col_res2, col_res3 = st.columns(3)
                
                status_color = "#10B981" if status == "Placed" else "#EF4444"
                
                with col_res1:
                    st.markdown(f"""
                    <div class="metric-card" style="border-top: 4px solid {status_color};">
                        <div class="metric-label">Predicted Outcome</div>
                        <div class="metric-value" style="color: {status_color};">{status}</div>
                        <div style="font-size: 13px; color: #64748B; margin-top: 4px;">Classification Model Forecast</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_res2:
                    st.markdown(f"""
                    <div class="metric-card" style="border-top: 4px solid #6366F1;">
                        <div class="metric-label">Placement Probability</div>
                        <div class="metric-value" style="color: #6366F1;">{prob:.1f}%</div>
                        <div style="font-size: 13px; color: #64748B; margin-top: 4px;">Confidence Level: {conf:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_res3:
                    st.markdown(f"""
                    <div class="metric-card" style="border-top: 4px solid #8B5CF6;">
                        <div class="metric-label">Expected Package</div>
                        <div class="metric-value" style="color: #8B5CF6;">{salary:.2f} LPA</div>
                        <div style="font-size: 13px; color: #64748B; margin-top: 4px;">95% CI Range: {sal_min:.1f} - {sal_max:.1f} LPA</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.write("")
                st.write("")
                
                # Explainability Section
                st.subheader("🔍 Explainable AI (XAI) Attribution")
                st.markdown("Deconstruct prediction outcomes into specific positive and negative parameter contributions.")
                
                explanation = explainer.get_shap_explanation(student_dict)
                
                col_ex1, col_ex2 = st.columns(2)
                with col_ex1:
                    st.markdown("##### 📈 Top Positive Drivers")
                    for item in explanation['positive_influences']:
                        st.success(f"**{item['friendly_name']}** (`{item['raw_value']}`): Increased placement probability by +{abs(item['shap_value']) * 100:.1f}%")
                        
                with col_ex2:
                    st.markdown("##### 📉 Top Risk Factors / Constraints")
                    if len(explanation['negative_influences']) == 0:
                        st.info("No significant negative constraints detected in student profile.")
                    else:
                        for item in explanation['negative_influences']:
                            st.error(f"**{item['friendly_name']}** (`{item['raw_value']}`): Reduced placement probability by -{abs(item['shap_value']) * 100:.1f}%")
                            
                st.write("")
                
                tab_wf, tab_bar = st.tabs(["🌊 Waterfall Attribution Chart", "📊 Top Feature Importances"])
                with tab_wf:
                    fig_waterfall = explainer.plot_waterfall_plotly(explanation)
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                with tab_bar:
                    fig_shap = explainer.plot_shap_plotly(explanation)
                    st.plotly_chart(fig_shap, use_container_width=True)

# ==========================================
# MODULE 3: History & Data Management
# ==========================================
elif routing_selection == "History & Data Management":
    st.title("💾 Prediction History & Data Management")
    st.markdown("Query, search, export, or manage past student placement predictions stored in the SQLite database.")
    
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        search_term = st.text_input("🔍 Search student history by Name or Department", placeholder="e.g. Aarav or Computer Science")
    with col_s2:
        st.write("")
        st.write("")
        if st.button("Clear History Log", use_container_width=True, type="secondary"):
            clear_history()
            st.success("Prediction history database cleared.")
            st.rerun()
            
    if search_term.strip():
        history_df = search_history(search_term.strip())
    else:
        history_df = get_prediction_history()
        
    if history_df.empty:
        st.info("No prediction history records found in database.")
    else:
        st.dataframe(history_df, use_container_width=True)
        
        # Export CSV Button
        csv_buffer = io.BytesIO()
        history_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export Prediction Logs to CSV",
            data=csv_buffer.getvalue(),
            file_name="student_placement_predictions_export.csv",
            mime="text/csv",
            type="primary"
        )
