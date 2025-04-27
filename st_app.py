import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import time

# Set page configuration
st.set_page_config(
    page_title="ATS Resume Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_BASE_URL = "https://ats-ai-api.vercel.app"

# Check if dark theme is enabled
def is_dark_theme():
    try:
        return st.get_option("theme.base") == "dark"
    except:
        return False

# Define theme colors based on current theme
def get_theme_colors():
    if is_dark_theme():
        return {
            "primary": "#6D9EEB",  #6D9EEB
            "secondary": "#90CAF9",
            "background": "#1E1E1E",  
            "card_bg": "#2D2D2D",
            "text": "#FFFFFF",
            "text_secondary": "#CCCCCC",   #CCCCCC
            "success": "#81C784",
            "warning": "#FFD54F",
            "error": "#E57373",
            "border": "#444444",
            "highlight_bg": "#3D3D3D",
            "card_border": "#6D9EEB",
            "suggestion_bg": "#2C3E50"
        }
    else:
        return {
            "primary": "#1E88E5", #1E88E5
            "secondary": "#64B5F6",
            "background": "#FFFFFF",
            "card_bg": "#F8F9FA",
            "text": "#212121",
            "text_secondary": "#555555",
            "success": "#4CAF50",
            "warning": "#FFA726",
            "error": "#EF5350",
            "border": "#E0E0E0",
            "highlight_bg": "#E3F2FD",
            "card_border": "#1E88E5",
            "suggestion_bg": "#E3F2FD"
        }

# Get theme colors
colors = get_theme_colors()

# Custom CSS with theme support
def get_custom_css():
    colors = get_theme_colors()
    return f"""
    <style>
        /* General Styling */
        .main-header {{
            font-size: 2.8rem;
            font-weight: 700;
            color: {colors["primary"]};
            margin-bottom: 1rem;
            text-align: center;
            padding: 1rem 0;
        }}
        
        .sub-header {{
            font-size: 1.8rem;
            font-weight: 600;
            color: {colors["primary"]};
            margin-bottom: 1.5rem;
            border-bottom: 2px solid {colors["primary"]};
            padding-bottom: 0.5rem;
        }}
        
        .section-header {{
            font-size: 1.4rem;
            font-weight: 600;
            color: {colors["text"]};
            margin: 1rem 0;
        }}

        /* Card Styling */
        .card {{
            padding: 1.8rem;
            border-radius: 0.8rem;
            background-color: {colors["card_bg"]};
            margin-bottom: 1.5rem;
            border-left: 5px solid {colors["card_border"]};
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
            color: {colors["text"]};
        }}
        
        .card:hover {{
            transform: translateY(-5px);
        }}
        
        .highlight {{
            color: {colors["primary"]};
            font-weight: 600;
        }}

        /* Score Styling */
        .score-high {{
            color: {colors["success"]};
            font-weight: 700;
            font-size: 1.5rem;
        }}
        
        .score-medium {{
            color: {colors["warning"]};
            font-weight: 700;
            font-size: 1.5rem;
        }}
        
        .score-low {{
            color: {colors["error"]};
            font-weight: 700;
            font-size: 1.5rem;
        }}

        /* Issues and Suggestions */
        .suggestion-item {{
            padding: 0.8rem;
            background-color: {colors["suggestion_bg"]};
            border-radius: 0.5rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            color: {colors["text"]};
        }}
        
        .issue-high {{
            background-color: rgba(239, 83, 80, 0.15);
            padding: 0.8rem;
            border-radius: 0.5rem;
            margin-bottom: 0.8rem;
            border-left: 4px solid {colors["error"]};
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            color: {colors["text"]};
        }}
        
        .issue-medium {{
            background-color: rgba(255, 167, 38, 0.15);
            padding: 0.8rem;
            border-radius: 0.5rem;
            margin-bottom: 0.8rem;
            border-left: 4px solid {colors["warning"]};
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            color: {colors["text"]};
        }}
        
        .issue-low {{
            background-color: rgba(76, 175, 80, 0.15);
            padding: 0.8rem;
            border-radius: 0.5rem;
            margin-bottom: 0.8rem;
            border-left: 4px solid {colors["success"]};
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            color: {colors["text"]};
        }}

        p, li, h1, h2, h3, h4, h5, h6, div {{
            color: {colors["text"]};
        }}
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px;
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            white-space: pre-wrap;
            background-color: {colors["card_bg"]};
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding: 10px 16px;
            font-weight: 500;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: {colors["primary"]};
            color: white;
        }}
    </style>
    """

st.markdown(get_custom_css(), unsafe_allow_html=True)

# Helper functions
def get_score_class(score):
    if score >= 80:
        return "score-high"
    elif score >= 60:
        return "score-medium"
    else:
        return "score-low"

def get_issue_class(severity):
    if severity.lower() == "error" or severity.lower() == "high":
        return "issue-high"
    elif severity.lower() == "warning" or severity.lower() == "medium":
        return "issue-medium"
    else:
        return "issue-low"

def analyze_resume_ats_only(resume_file):
    """Submit resume for ATS-only analysis"""
    files = {"resume_file": resume_file}
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze/ats-only/",
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {str(e)}")
        return None

def analyze_resume_job_matching(resume_file, job_description):
    """Submit resume for job matching analysis"""
    files = {"resume_file": resume_file}
    data = {"job_description": job_description}
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze/job-matching/",
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {str(e)}")
        return None

def create_radar_chart(categories):
    """Create a radar chart from category scores"""
    categories_dict = {cat['category']: cat['score'] for cat in categories}
    
    categories_list = ['content', 'format', 'sections', 'skills', 'style']
    values = [categories_dict.get(cat, 0) for cat in categories_list]
    
    # Add the first value to close the radar chart
    categories_list.append(categories_list[0])
    values.append(values[0])
    
    # Capitalized labels
    labels = [cat.capitalize() for cat in categories_list]
    
    # Theme-aware colors
    colors = get_theme_colors()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Resume Score',
        line_color=colors["primary"],
        fillcolor=f"rgba({int(colors['primary'][1:3], 16)}, {int(colors['primary'][3:5], 16)}, {int(colors['primary'][5:7], 16)}, 0.3)"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color=colors["text"]),
                gridcolor=colors["border"],
                linecolor=colors["border"]
            ),
            angularaxis=dict(
                tickfont=dict(color=colors["text"]),
                gridcolor=colors["border"],
                linecolor=colors["border"]
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=400,
        margin=dict(l=70, r=70, t=20, b=20),
        font=dict(color=colors["text"])
    )
    
    return fig

def create_progress_bar(value, max_value=100):
    """Create a styled progress bar"""
    percentage = (value / max_value) * 100
    colors = get_theme_colors()
    
    if percentage >= 80:
        bar_color = colors["success"]
    elif percentage >= 60:
        bar_color = colors["warning"]
    else:
        bar_color = colors["error"]
    
    html = f"""
    <div style="width: 100%; background-color: {colors['border']}; height: 10px; border-radius: 5px;">
        <div style="width: {percentage}%; background-color: {bar_color}; height: 10px; border-radius: 5px;"></div>
    </div>
    """
    return html

def display_results(results, job_match=False):
    """Display analysis results in a structured format"""
    colors = get_theme_colors()
    
    # Ensure results contains necessary keys
    if not results or not isinstance(results, dict):
        st.error("Invalid results data received from API")
        return
    
    # Dashboard Header
    st.markdown(f"<h2 class='sub-header'>Resume Analysis Results</h2>", unsafe_allow_html=True)
    
    # Key Metrics Row
    metric_cols = st.columns(3 if job_match else 2)
    
    # Overall Score
    with metric_cols[0]:
        score_class = get_score_class(results["overall_score"])
        st.markdown(f"""
        <div class="card">
            <div style="font-weight: 600; margin-bottom: 0.5rem;">Overall Score</div>
            <div class="{score_class}">{results["overall_score"]}/100</div>
            {create_progress_bar(results["overall_score"])}
        </div>
        """, unsafe_allow_html=True)
    
    # ATS Parse Rate
    with metric_cols[1]:
        parse_score_class = get_score_class(results["ats_parse_rate"])
        st.markdown(f"""
        <div class="card">
            <div style="font-weight: 600; margin-bottom: 0.5rem;">ATS Parse Rate</div>
            <div class="{parse_score_class}">{results["ats_parse_rate"]}%</div>
            {create_progress_bar(results["ats_parse_rate"])}
        </div>
        """, unsafe_allow_html=True)
    
    # Job Match (if applicable)
    if job_match and len(metric_cols) > 2:
        with metric_cols[2]:
            match_score_class = get_score_class(results["match_percentage"])
            st.markdown(f"""
            <div class="card">
                <div style="font-weight: 600; margin-bottom: 0.5rem;">Job Match</div>
                <div class="{match_score_class}">{results["match_percentage"]}%</div>
                {create_progress_bar(results["match_percentage"])}
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed Analysis Section
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f"<h3 class='section-header'>Detailed Analysis</h3>", unsafe_allow_html=True)
        
        # Categories breakdown
        for category in results["categories"]:
            cat_name = category["category"].capitalize()
            cat_score = category["score"]
            score_class = get_score_class(cat_score)
            
            with st.expander(f"{cat_name} - Score: {cat_score}/100", expanded=False):
                st.markdown(f"{create_progress_bar(cat_score)}", unsafe_allow_html=True)
                
                if category.get("analysis"):
                    st.markdown(f"<p>{category['analysis']}</p>", unsafe_allow_html=True)
                
                if category["issues_found"] > 0:
                    st.markdown("<h4>📋 Issues Found:</h4>", unsafe_allow_html=True)
                    for issue in category.get("issues", []):
                        severity = issue.get("severity", "medium").lower()
                        issue_class = get_issue_class(severity)
                        st.markdown(f"""
                        <div class="{issue_class}">
                            {issue["message"]}
                        </div>
                        """, unsafe_allow_html=True)
                
                if category.get("suggestions"):
                    st.markdown("<h4>💡 Suggestions:</h4>", unsafe_allow_html=True)
                    for suggestion in category["suggestions"]:
                        st.markdown(f"""
                        <div class="suggestion-item">
                            {suggestion}
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        # Radar chart
        st.markdown("<h3 class='section-header'>Category Scores</h3>", unsafe_allow_html=True)
        fig = create_radar_chart(results["categories"])
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary card
        st.markdown("<h3 class='section-header'>Summary</h3>", unsafe_allow_html=True)
        
        # Count issues by severity
        issue_counts = {"high": 0, "medium": 0, "low": 0}
        for category in results["categories"]:
            for issue in category.get("issues", []):
                severity = issue.get("severity", "medium").lower()
                if severity in ["error", "high"]:
                    issue_counts["high"] += 1
                elif severity in ["warning", "medium"]:
                    issue_counts["medium"] += 1
                else:
                    issue_counts["low"] += 1
        
        st.markdown(f"""
        <div class="card">
            <h4>Resume Health Check</h4>
            <p>
                <span style="color: {colors['error']};">Critical Issues: {issue_counts["high"]}</span> | 
                <span style="color: {colors['warning']};">Improvements Needed: {issue_counts["medium"]}</span> | 
                <span style="color: {colors['success']};">Minor Suggestions: {issue_counts["low"]}</span>
            </p>
            <p>Based on our analysis, your resume {"is performing well but has a few areas for improvement" if results["overall_score"] >= 70 else "needs significant improvements to pass most ATS systems effectively"}.</p>
            <p><strong>Next Steps:</strong> {"Focus on addressing the critical issues first, then work on the suggested improvements." if issue_counts["high"] > 0 else "Make the suggested improvements to further optimize your resume for ATS systems."}</p>
        </div>
        """, unsafe_allow_html=True)

def create_animated_loading():
    """Create an animated loading screen"""
    colors = get_theme_colors()
    
    loading_html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 3rem 0;">
        <div style="font-size: 2rem; color: {colors['primary']}; margin-bottom: 1rem;">
            ⚙️ Analyzing Your Resume...
        </div>
        <div style="width: 80%; max-width: 600px; height: 8px; background-color: {colors['border']}; border-radius: 4px; overflow: hidden; margin-top: 1rem;">
            <div id="progress-bar" style="width: 0%; height: 100%; background-color: {colors['primary']}; transition: width 0.3s;"></div>
        </div>
        <div style="margin-top: 1rem; font-style: italic;">This may take a few moments</div>
    </div>
    """
    
    loading_container = st.empty()
    loading_container.markdown(loading_html, unsafe_allow_html=True)
    
    for i in range(0, 101, 10):
        # Update progress bar
        progress_html = f"""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 3rem 0;">
            <div style="font-size: 2rem; color: {colors['primary']}; margin-bottom: 1rem;">
                ⚙️ Analyzing Your Resume...
            </div>
            <div style="width: 80%; max-width: 600px; height: 8px; background-color: {colors['border']}; border-radius: 4px; overflow: hidden; margin-top: 1rem;">
                <div id="progress-bar" style="width: {i}%; height: 100%; background-color: {colors['primary']}; transition: width 0.3s;"></div>
            </div>
            <div style="margin-top: 1rem; font-style: italic;">This may take a few moments</div>
        </div>
        """
        loading_container.markdown(progress_html, unsafe_allow_html=True)
        time.sleep(0.1)
    
    return loading_container

def ats_only_check():
    st.markdown("<h2 class='sub-header'>Check CV ATS Only</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>ATS Compatibility Check</h3>
        <p>Upload your resume to check if it can be properly parsed by Applicant Tracking Systems.</p>
        <p>Get feedback on format, keywords, readability, and overall ATS compatibility.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your resume (PDF  format)", type=["pdf"], key="ats_only_uploader")
    
    if uploaded_file is not None:
        analyze_button = st.button("Analyze Resume", key="ats_only_btn")
        
        if analyze_button:
            loading_container = create_animated_loading()
            
            # Reset the file pointer to the beginning
            uploaded_file.seek(0)
            results = analyze_resume_ats_only(uploaded_file)
            
            # Clear loading animation
            loading_container.empty()
            
            if results:
                display_results(results, job_match=False)

def job_match_check():
    st.markdown("<h2 class='sub-header'>Check Resume with Job Description</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>Job Match Analysis</h3>
        <p>Compare your resume against a specific job description.</p>
        <p>Identify missing keywords and skills to improve your chances of getting past ATS filters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your resume (PDF  format)", type=["pdf"], key="job_match_uploader")
    
    st.markdown("<h3 class='section-header'>Job Description</h3>", unsafe_allow_html=True)
    job_description = st.text_area("Paste the job description here", height=200)
    
    if uploaded_file is not None and job_description:
        analyze_button = st.button("Analyze Resume", key="job_match_btn")
        
        if analyze_button:
            loading_container = create_animated_loading()
            
            # Reset the file pointer to the beginning
            uploaded_file.seek(0)
            results = analyze_resume_job_matching(uploaded_file, job_description)
            
            # Clear loading animation
            loading_container.empty()
            
            if results:
                display_results(results, job_match=True)
    elif uploaded_file is not None and not job_description:
        st.warning("Please enter a job description to continue with the analysis.")

def main():
    # App Header
    st.markdown("<h1 class='main-header'>📝 ATS Resume Analyzer</h1>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Check CV ATS Only", "Check with Job Description"])
    
    with tab1:
        ats_only_check()
    
    with tab2:
        job_match_check()

if __name__ == "__main__":
    main()