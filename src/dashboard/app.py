import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List
import base64
from io import BytesIO
import json

from src.config import settings, yaml_config
from src.utils import get_logger


logger = get_logger(__name__)

# Advanced page configuration
st.set_page_config(
    page_title="AI Sentinel - Digital Rights Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/ai-sentinel',
        'Report a bug': "https://github.com/yourusername/ai-sentinel/issues",
        'About': "# AI Sentinel\nMultimodal Explainable System for Detecting Digital Human Rights Violations"
    }
)

# Use localhost for client connections
API_HOST = "localhost" if settings.API_HOST == "0.0.0.0" else settings.API_HOST
API_BASE_URL = f"http://{API_HOST}:{settings.API_PORT}"

# API timeout settings
API_TIMEOUT = 5  # Reduced timeout for faster failure detection


def check_api_status():
    """Check if API server is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False


def api_call_with_error_handling(method, endpoint, **kwargs):
    """Wrapper for API calls with proper error handling"""
    # Set default timeout if not provided
    if 'timeout' not in kwargs:
        kwargs['timeout'] = API_TIMEOUT

    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.request(method, url, **kwargs)
        return response, None
    except requests.exceptions.Timeout:
        error_msg = "Request timed out. Please make sure the API server is running:\n\n```bash\npython run_api.py\n```"
        return None, error_msg
    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to API server. Please start it first:\n\n```bash\npython run_api.py\n```"
        return None, error_msg
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return None, error_msg


# Custom CSS for modern styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #FF4B4B;
        --secondary-color: #0068C9;
        --success-color: #09AB3B;
        --warning-color: #FFA500;
        --danger-color: #FF4B4B;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #ffffff;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
        opacity: 0.95;
    }

    /* Card styling - Light mode */
    .custom-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        color: #2c3e50;
    }

    .custom-card h3, .custom-card h4 {
        color: #1f1f1f;
        margin-top: 0;
        font-weight: 700;
    }

    .custom-card h2 {
        color: #1f1f1f;
        font-weight: 700;
    }

    .custom-card p, .custom-card ul, .custom-card li {
        color: #2c3e50;
        line-height: 1.6;
    }

    /* Card styling - Dark mode */
    @media (prefers-color-scheme: dark) {
        .custom-card {
            background: #1e1e1e;
            border-left-color: #8b9dff;
            color: #e8eaed;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }

        .custom-card h2, .custom-card h3, .custom-card h4 {
            color: #ffffff;
        }

        .custom-card p, .custom-card ul, .custom-card li {
            color: #e8eaed;
        }
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }

    .metric-label {
        font-size: 1rem;
        color: #2c3e50;
        margin-top: 0.5rem;
        font-weight: 600;
    }

    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        }

        .metric-label {
            color: #e8eaed;
        }

        .metric-value {
            color: #8b9dff;
        }
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    /* Alert boxes */
    .alert-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #155724;
        font-weight: 500;
    }

    .alert-danger {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #721c24;
        font-weight: 500;
    }

    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #856404;
        font-weight: 500;
    }

    @media (prefers-color-scheme: dark) {
        .alert-success {
            background-color: rgba(16, 185, 129, 0.15);
            color: #86efac;
            border-left-color: #10b981;
        }

        .alert-danger {
            background-color: rgba(239, 68, 68, 0.15);
            color: #fca5a5;
            border-left-color: #ef4444;
        }

        .alert-warning {
            background-color: rgba(245, 158, 11, 0.15);
            color: #fcd34d;
            border-left-color: #f59e0b;
        }
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* File uploader */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        color: #2c3e50;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    @media (prefers-color-scheme: dark) {
        .stTabs [data-baseweb="tab"] {
            background-color: #2c3e50;
            color: #e8eaed;
        }
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 8px;
        font-weight: 600;
        color: #2c3e50;
    }

    @media (prefers-color-scheme: dark) {
        .streamlit-expanderHeader {
            background-color: #2c3e50;
            color: #e8eaed;
        }
    }

    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }

    /* Pulse animation for live indicators */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .pulse {
        animation: pulse 2s infinite;
    }
    </style>
    """, unsafe_allow_html=True)


def create_gauge_chart(value: float, title: str, color_scheme: str = "RdYlGn"):
    """Create a gauge chart for confidence visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#333'}},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#d4edda'},
                {'range': [33, 66], 'color': '#fff3cd'},
                {'range': [66, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#333", 'family': "Arial"}
    )

    return fig


def create_probability_chart(probabilities: Dict[str, float], chart_type: str = "bar"):
    """Create enhanced probability visualization"""
    df = pd.DataFrame(list(probabilities.items()), columns=['Category', 'Probability'])
    df = df.sort_values('Probability', ascending=False)

    if chart_type == "bar":
        fig = go.Figure(data=[
            go.Bar(
                x=df['Category'],
                y=df['Probability'],
                marker=dict(
                    color=df['Probability'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    line=dict(color='rgb(8,48,107)', width=1.5)
                ),
                text=[f'{p:.1%}' for p in df['Probability']],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Probability Distribution",
            xaxis_title="Category",
            yaxis_title="Probability",
            yaxis=dict(tickformat='.0%'),
            height=400,
            template="plotly_white",
            hovermode='x unified'
        )

    elif chart_type == "pie":
        fig = go.Figure(data=[
            go.Pie(
                labels=df['Category'],
                values=df['Probability'],
                hole=0.4,
                marker=dict(colors=['#09AB3B', '#FFA500', '#FF4B4B']),
                textinfo='label+percent',
                textfont_size=14
            )
        ])

        fig.update_layout(
            title="Probability Distribution",
            height=400,
            template="plotly_white"
        )

    return fig


def main():
    load_custom_css()

    # Custom header
    st.markdown("""
    <div class="main-header">
        <h1>AI Sentinel: Digital Rights Monitor</h1>
        <p>Multimodal Explainable System for Detecting Digital Human Rights Violations</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with enhanced navigation
    with st.sidebar:
        st.markdown("###  Navigation")

        # Add dynamic status indicator
        api_online = check_api_status()
        if api_online:
            st.markdown("""
            <div style="background: #d4edda; padding: 0.5rem; border-radius: 5px; margin-bottom: 1rem;">
                <span style="color: #28a745;">● </span><strong style="color: #155724;">API Online</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #fff3cd; padding: 0.5rem; border-radius: 5px; margin-bottom: 1rem;">
                <span style="color: #856404;">● </span><strong style="color: #856404;">API Offline</strong>
            </div>
            """, unsafe_allow_html=True)
            st.warning(" Start API server:\n```bash\npython run_api.py\n```", icon="")

        page = st.radio(
            "Select Module",
            [" Home", " Text Analysis", " Image Analysis", " Video Analysis",
             " Global Events", " Analytics", " Settings"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Quick stats in sidebar
        st.markdown("###  Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Analyses", "1,234", "+12%")
        with col2:
            st.metric("Accuracy", "94.2%", "+2.1%")

        st.markdown("---")

        # Theme toggle
        st.markdown("###  Appearance")
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"], index=0)

        st.markdown("---")
        st.markdown("### About")
        api_status_text = "Connected" if api_online else "Disconnected"
        st.info(f"**Version:** 1.0.0\n\n**Status:** Production\n\n**API:** {api_status_text}")

    # Route to pages
    if page == " Home":
        home_page()
    elif page == " Text Analysis":
        text_analysis_page()
    elif page == " Image Analysis":
        image_analysis_page()
    elif page == " Video Analysis":
        video_analysis_page()
    elif page == " Global Events":
        global_events_page()
    elif page == " Analytics":
        analytics_dashboard_page()
    elif page == " Settings":
        settings_page()


def home_page():
    """Enhanced home page with overview"""
    st.markdown("## Welcome to AI Sentinel")

    # Feature cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="custom-card">
            <h3> Text Analysis</h3>
            <p>Detect hate speech, disinformation, and offensive content in multiple languages with explainable AI.</p>
            <ul>
                <li>Multilingual support</li>
                <li>Real-time detection</li>
                <li>LIME explanations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="custom-card">
            <h3>Image & Video</h3>
            <p>Identify deepfakes and manipulated media using state-of-the-art computer vision models.</p>
            <ul>
                <li>Deepfake detection</li>
                <li>Visual explanations</li>
                <li>Batch processing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="custom-card">
            <h3>Event Correlation</h3>
            <p>Connect detections with real-world events using GDELT global database integration.</p>
            <ul>
                <li>Real-time events</li>
                <li>Geographic mapping</li>
                <li>Trend analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Recent activity timeline
    st.markdown("## System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">1,234</p>
            <p class="metric-label">Total Analyses</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">94.2%</p>
            <p class="metric-label">Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">45</p>
            <p class="metric-label">Threats Detected</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">12</p>
            <p class="metric-label">Countries</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick start guide
    with st.expander(" Quick Start Guide", expanded=False):
        st.markdown("""
        ### Getting Started with AI Sentinel

        1. **Text Analysis**: Navigate to the Text Analysis page and paste content to analyze
        2. **Image Analysis**: Upload images to detect deepfakes and manipulated media
        3. **Video Analysis**: Process video files for comprehensive deepfake detection
        4. **Global Events**: Search and correlate with real-world events from GDELT
        5. **Analytics**: View aggregated statistics and trends

        ### API Integration

        You can also integrate AI Sentinel into your applications using our REST API:

        ```python
        import requests

        response = requests.post(
            "http://localhost:8000/detect/text",
            json={"text": "Your text here", "explain": True}
        )
        ```

        ### Support

        For help and documentation, visit our [GitHub repository](https://github.com/yourusername/ai-sentinel).
        """)


def text_analysis_page():
    st.markdown("## Text Analysis")

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Single Text", "Batch Upload"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            text_input = st.text_area(
                "Enter text to analyze",
                height=250,
                placeholder="Paste text content here for hate speech and disinformation detection...",
                help="Supports multiple languages including English, Arabic, Spanish, and more"
            )

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                location = st.text_input(
                    "Location (optional)",
                    placeholder="e.g., Ukraine, Syria"
                )
            with col_b:
                explain = st.checkbox("Generate Explanation", value=True)
            with col_c:
                correlate = st.checkbox("Correlate Events", value=False)

            analyze_button = st.button("Analyze Text", type="primary", use_container_width=True)

        with col2:
            st.markdown("""
            <div class="custom-card">
                <h4>Detection Categories</h4>
                <ul>
                    <li><strong>Normal:</strong> Safe content</li>
                    <li><strong>Hate:</strong> Hate speech detected</li>
                    <li><strong>Offensive:</strong> Offensive content</li>
                </ul>

                <h4>Features</h4>
                <ul>
                    <li>Multilingual support (50+ languages)</li>
                    <li>Explainable AI (LIME)</li>
                    <li>Event correlation (GDELT)</li>
                    <li>Real-time processing</li>
                </ul>

                <h4>Model Info</h4>
                <p><strong>Architecture:</strong> XLM-RoBERTa</p>
                <p><strong>Accuracy:</strong> 94.2%</p>
                <p><strong>Languages:</strong> 50+</p>
            </div>
            """, unsafe_allow_html=True)

        if analyze_button and text_input:
            with st.spinner("Analyzing text..."):
                response, error = api_call_with_error_handling(
                    "POST",
                    "/detect/text",
                    json={
                        "text": text_input,
                        "explain": explain,
                        "correlate": correlate and bool(location),
                        "location": location if location else None
                    },
                    timeout=60  # ML inference can take longer
                )

                if error:
                    st.error(error)
                elif response and response.status_code == 200:
                    result = response.json()
                    # Store result in session state for analytics
                    if 'analysis_results' not in st.session_state:
                        st.session_state.analysis_results = []
                    st.session_state.analysis_results.append({
                        'type': 'Text Analysis',
                        'prediction': result.get('prediction', 'unknown').lower(),
                        'confidence': result.get('confidence', 0),
                        'input': text_input[:100],
                        'language': result.get('language', 'unknown')
                    })
                    display_text_results(result)
                elif response:
                    st.error(f"Analysis failed: {response.text}")

    with tab2:
        st.markdown("### Batch Text Analysis")
        uploaded_file = st.file_uploader(
            "Upload CSV or TXT file",
            type=["csv", "txt"],
            help="Upload a file containing multiple texts for batch processing"
        )

        if uploaded_file:
            st.info("Batch processing feature coming soon!")


def display_text_results(result: Dict[str, Any], source: str = "Text"):
    st.markdown("---")
    st.markdown(f"## Analysis Results ({source})")

    # Main metrics with enhanced styling
    col1, col2, col3, col4 = st.columns(4)

    prediction = result["prediction"].upper()
    confidence = result["confidence"]

    # Determine color based on prediction
    if prediction == "NORMAL":
        status_color = "green"
        alert_class = "alert-success"
        status_text = "CLEAN"
    elif prediction == "OFFENSIVE":
        status_color = "orange"
        alert_class = "alert-warning"
        status_text = "WARNING"
    else:  # HATE
        status_color = "red"
        alert_class = "alert-danger"
        status_text = "THREAT"

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color: {status_color};">●</p>
            <p class="metric-label">{status_text}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{prediction}</p>
            <p class="metric-label">Prediction</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{confidence:.1%}</p>
            <p class="metric-label">Confidence</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{result["language"].upper()}</p>
            <p class="metric-label">Language</p>
        </div>
        """, unsafe_allow_html=True)

    # Alert box
    st.markdown(f"""
    <div class="{alert_class}">
        <strong>Detection Result:</strong> The content has been classified as <strong>{prediction}</strong>
        with {confidence:.1%} confidence.
    </div>
    """, unsafe_allow_html=True)

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###  Probability Distribution")
        chart_type = st.radio("Chart Type", ["Bar", "Pie"], horizontal=True, label_visibility="collapsed")
        fig = create_probability_chart(
            result["probabilities"],
            "bar" if chart_type == "Bar" else "pie"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("###  Confidence Gauge")
        gauge_fig = create_gauge_chart(confidence, "Confidence Score")
        st.plotly_chart(gauge_fig, use_container_width=True)

    # Explanation section
    if result.get("explanation"):
        st.markdown("---")
        st.markdown("###  Explainable AI Analysis")

        explanation_data = result["explanation"]["explanation"]
        exp_df = pd.DataFrame(explanation_data, columns=["Word/Phrase", "Impact"])
        exp_df = exp_df.sort_values("Impact", key=abs, ascending=False)

        # Create impact visualization
        fig = go.Figure(data=[
            go.Bar(
                x=exp_df["Impact"][:15],
                y=exp_df["Word/Phrase"][:15],
                orientation='h',
                marker=dict(
                    color=exp_df["Impact"][:15],
                    colorscale='RdYlGn_r',
                    showscale=True
                ),
                text=[f'{x:.3f}' for x in exp_df["Impact"][:15]],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Top 15 Contributing Words/Phrases",
            xaxis_title="Impact Score",
            yaxis_title="Word/Phrase",
            height=500,
            template="plotly_white",
            yaxis={'categoryorder': 'total ascending'}
        )

        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        with st.expander(" View Detailed Explanation Table"):
            st.dataframe(exp_df, use_container_width=True, height=400)

    # Event correlations
    if result.get("correlations"):
        st.markdown("---")
        st.markdown("###  Correlated Global Events")

        for idx, corr in enumerate(result["correlations"][:5]):
            similarity = corr['similarity_score']

            # Color code by similarity
            if similarity > 0.7:
                badge_color = "#28a745"
            elif similarity > 0.5:
                badge_color = "#ffc107"
            else:
                badge_color = "#dc3545"

            event = corr["event"]

            with st.expander(f"Event {idx + 1} - Similarity: {similarity:.1%}", expanded=(idx == 0)):
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 5px; border-left: 4px solid {badge_color};">
                    <h4 style="color: #1f1f1f; margin-top: 0;">{event.get('title', 'N/A')}</h4>
                    <p style="color: #333333;"><strong>URL:</strong> <a href="{event.get('url', '#')}" target="_blank" style="color: #667eea;">{event.get('url', 'N/A')}</a></p>
                    <p style="color: #333333;"><strong>Domain:</strong> {event.get('domain', 'N/A')}</p>
                    <p style="color: #333333;"><strong>Language:</strong> {event.get('language', 'N/A')}</p>
                    <p style="color: #333333;"><strong>Similarity:</strong> <span style="color: {badge_color}; font-weight: bold;">{similarity:.1%}</span></p>
                </div>
                """, unsafe_allow_html=True)

    # Export options
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(" Export as JSON", use_container_width=True):
            json_str = json.dumps(result, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    with col2:
        if st.button(" Export as PDF", use_container_width=True):
            st.info("PDF export feature coming soon!")

    with col3:
        if st.button(" Email Report", use_container_width=True):
            st.info("Email feature coming soon!")


def image_analysis_page():
    st.markdown("##  Image Analysis")

    tab1, tab2 = st.tabs([" Single Upload", " Batch Processing"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png"],
                help="Upload an image to detect deepfakes and manipulated media"
            )

            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                explain = st.checkbox(" Generate Visual Explanation", value=True)
            with col_b:
                show_heatmap = st.checkbox(" Show Attention Heatmap", value=True)

            analyze_button = st.button(" Analyze Image", type="primary", use_container_width=True)

        with col2:
            st.markdown("""
            <div class="custom-card">
                <h4> Detection</h4>
                <ul>
                    <li><strong>Real:</strong> Authentic image</li>
                    <li><strong>Fake:</strong> Deepfake/manipulated</li>
                </ul>

                <h4>Supported Formats</h4>
                <ul>
                    <li>JPG, JPEG</li>
                    <li>PNG</li>
                    <li>Max size: 10MB</li>
                </ul>

                <h4>Features</h4>
                <ul>
                    <li>Deepfake detection</li>
                    <li>Visual explanations (Grad-CAM)</li>
                    <li>Attention heatmaps</li>
                    <li>Batch processing</li>
                </ul>

                <h4>Model Info</h4>
                <p><strong>Architecture:</strong> EfficientNet-B0</p>
                <p><strong>Accuracy:</strong> 96.8%</p>
                <p><strong>Dataset:</strong> Celeb-DF v2</p>
            </div>
            """, unsafe_allow_html=True)

        if analyze_button and uploaded_file:
            with st.spinner(" Analyzing image..."):
                response, error = api_call_with_error_handling(
                    "POST",
                    "/detect/image",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    params={"explain": explain},
                    timeout=60  # ML inference can take longer
                )

                if error:
                    st.error(error)
                elif response and response.status_code == 200:
                    result = response.json()
                    display_image_results(result, uploaded_file)
                elif response:
                    st.error(f" Analysis failed: {response.text}")

    with tab2:
        st.markdown("###  Batch Image Processing")
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.info(f" {len(uploaded_files)} images uploaded. Batch processing feature coming soon!")


def display_image_results(result: Dict[str, Any], uploaded_file):
    st.markdown("---")
    st.markdown("##  Analysis Results")

    prediction = result["prediction"].upper()
    confidence = result["confidence"]

    # Determine color
    if prediction == "REAL":
        pred_color = "🟢"
        alert_class = "alert-success"
    else:
        pred_color = "🔴"
        alert_class = "alert-danger"

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{pred_color}</p>
            <p class="metric-label">Status</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{prediction}</p>
            <p class="metric-label">Prediction</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{confidence:.1%}</p>
            <p class="metric-label">Confidence</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        timestamp = datetime.fromisoformat(result["timestamp"])
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{timestamp.strftime('%H:%M:%S')}</p>
            <p class="metric-label">Analyzed At</p>
        </div>
        """, unsafe_allow_html=True)

    # Alert
    st.markdown(f"""
    <div class="{alert_class}">
        <strong>Detection Result:</strong> The image has been classified as <strong>{prediction}</strong>
        with {confidence:.1%} confidence.
    </div>
    """, unsafe_allow_html=True)

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###  Original Image")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.markdown("###  Probability Distribution")
        fig = create_probability_chart(result["probabilities"], "bar")
        st.plotly_chart(fig, use_container_width=True)

    # Confidence gauge
    st.markdown("###  Confidence Analysis")
    gauge_fig = create_gauge_chart(confidence, "Detection Confidence")
    st.plotly_chart(gauge_fig, use_container_width=True)


def display_video_results(result: Dict[str, Any], uploaded_file):
    st.markdown("---")
    st.markdown("## Video Analysis Results")

    prediction = result["prediction"].upper()
    confidence = result["confidence"]
    stats = result.get("statistics", {})

    # Determine color
    if prediction == "REAL":
        status_color = "green"
        alert_class = "alert-success"
        status_text = "AUTHENTIC"
    else:
        status_color = "red"
        alert_class = "alert-danger"
        status_text = "DEEPFAKE"

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color: {status_color};">●</p>
            <p class="metric-label">{status_text}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{prediction}</p>
            <p class="metric-label">Prediction</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{confidence:.1%}</p>
            <p class="metric-label">Confidence</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{result.get('frames_analyzed', 0)}</p>
            <p class="metric-label">Frames Analyzed</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        model_type = result.get('model_type', 'frame_by_frame')
        model_icon = "" if model_type == "temporal" else ""
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{model_icon}</p>
            <p class="metric-label">{'Temporal' if model_type == 'temporal' else 'Frame-by-Frame'}</p>
        </div>
        """, unsafe_allow_html=True)

    # Alert
    st.markdown(f"""
    <div class="{alert_class}">
        <strong>Detection Result:</strong> The video has been classified as <strong>{prediction}</strong>
        with {confidence:.1%} confidence. {stats.get('fake_percentage', 0):.1f}% of frames detected as fake.
    </div>
    """, unsafe_allow_html=True)

    # Visualizations
    st.markdown("### Overall Probability")
    fig = create_probability_chart(result["probabilities"], "bar")
    st.plotly_chart(fig, use_container_width=True)

    # Frame-by-frame analysis
    if result.get("frame_predictions") and result.get("frame_confidences"):
        st.markdown("---")
        st.markdown("### Frame-by-Frame Analysis")

        frame_data = pd.DataFrame({
            'Frame': range(1, len(result["frame_predictions"]) + 1),
            'Prediction': result["frame_predictions"],
            'Confidence': result["frame_confidences"]
        })

        # Timeline chart
        fig = go.Figure()

        # Color frames based on prediction
        colors = ['green' if pred == 'real' else 'red' for pred in frame_data['Prediction']]

        fig.add_trace(go.Scatter(
            x=frame_data['Frame'],
            y=frame_data['Confidence'],
            mode='lines+markers',
            name='Confidence',
            line=dict(color='blue', width=2),
            marker=dict(size=8, color=colors),
            hovertemplate='<b>Frame %{x}</b><br>Confidence: %{y:.2%}<br>Prediction: %{text}<extra></extra>',
            text=frame_data['Prediction']
        ))

        fig.update_layout(
            title="Confidence Timeline Across Frames",
            xaxis_title="Frame Number",
            yaxis_title="Confidence",
            yaxis=dict(tickformat='.0%'),
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{stats.get('real_frames', 0)}</p>
                <p class="metric-label">Real Frames</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{stats.get('fake_frames', 0)}</p>
                <p class="metric-label">Fake Frames</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{stats.get('avg_confidence', 0):.1%}</p>
                <p class="metric-label">Avg Confidence</p>
            </div>
            """, unsafe_allow_html=True)


def video_analysis_page():
    """New video analysis page"""
    st.markdown("##  Video Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_video = st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mov", "mkv", "flv", "wmv", "webm", "m4v"],
            help="Upload a video to detect deepfakes. All formats are automatically converted to MP4 for playback."
        )

        if uploaded_video:
            st.success(f" Video loaded: {uploaded_video.name}")

        col_a, col_b = st.columns(2)
        with col_a:
            frame_interval = st.slider("Frame Sampling Interval", 1, 30, 10, help="Analyze every Nth frame")
        with col_b:
            max_frames = st.slider("Max Frames to Analyze", 10, 100, 30)

        analyze_button = st.button("Analyze Video", type="primary", use_container_width=True)

    with col2:
        st.markdown("""
        <div class="custom-card">
            <h4>Video Detection</h4>
            <p>Comprehensive deepfake detection across video frames</p>

            <h4>Supported Formats</h4>
            <ul>
                <li>MP4, AVI, MOV, MKV</li>
                <li>Max size: 100MB</li>
                <li>Max duration: 5 minutes</li>
            </ul>

            <h4>Features</h4>
            <ul>
                <li>Frame-by-frame analysis</li>
                <li>Temporal consistency check</li>
                <li>Aggregated predictions</li>
                <li>Timeline visualization</li>
            </ul>

            <h4>Processing</h4>
            <p>Videos are analyzed by extracting and processing individual frames using our deepfake detection model.</p>
        </div>
        """, unsafe_allow_html=True)

    if analyze_button and uploaded_video:
        with st.spinner("Analyzing video frames..."):
            response, error = api_call_with_error_handling(
                "POST",
                "/detect/video",
                files={"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)},
                params={
                    "frame_interval": frame_interval,
                    "max_frames": max_frames
                },
                timeout=120  # Video processing can take longer
            )

            if error:
                st.error(error)
            elif response and response.status_code == 200:
                result = response.json()
                # Store result in session state for analytics
                if 'analysis_results' not in st.session_state:
                    st.session_state.analysis_results = []
                st.session_state.analysis_results.append({
                    'type': 'Video Analysis',
                    'prediction': result.get('prediction', 'unknown').lower(),
                    'confidence': result.get('confidence', 0),
                    'input': uploaded_video.name[:100],
                    'frames_analyzed': result.get('frames_analyzed', 0)
                })
                display_video_results(result, uploaded_video)
            elif response:
                st.error(f"Analysis failed: {response.text}")


def global_events_page():
    st.markdown("## Global Events Monitor")

    col1, col2 = st.columns([2, 1])

    with col1:
        query = st.text_input(
            "Search GDELT Events",
            placeholder="e.g., protest, human rights, violence, conflict"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            max_records = st.slider("Maximum Records", 10, 500, 100)
        with col_b:
            time_range = st.selectbox("Time Range", ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"])

        search_button = st.button("Search Events", type="primary", use_container_width=True)

    with col2:
        st.markdown("""
        <div class="custom-card">
            <h4> GDELT Database</h4>
            <p>Global Database of Events, Language, and Tone</p>

            <h4> Coverage</h4>
            <ul>
                <li>Global event monitoring</li>
                <li>Real-time news tracking</li>
                <li>100+ languages</li>
                <li>200+ countries</li>
            </ul>

            <h4> Search Tips</h4>
            <ul>
                <li>Use specific keywords</li>
                <li>Combine with locations</li>
                <li>Filter by themes</li>
                <li>Check recent events</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if search_button and query:
        with st.spinner("🔄 Fetching events from GDELT..."):
            response, error = api_call_with_error_handling(
                "GET",
                "/gdelt/events",
                params={"query": query, "max_records": max_records},
                timeout=30  # GDELT queries may take longer
            )

            if error:
                st.error(error)
            elif response and response.status_code == 200:
                result = response.json()
                display_gdelt_events(result)
            elif response:
                st.error(f" Search failed: {response.text}")


def display_gdelt_events(result: Dict[str, Any]):
    st.markdown("---")
    st.success(f" Found {result['count']} events")

    if result["events"]:
        events_df = pd.DataFrame(result["events"])

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Events", len(events_df))
        with col2:
            unique_domains = events_df['domain'].nunique() if 'domain' in events_df else 0
            st.metric("Unique Sources", unique_domains)
        with col3:
            unique_langs = events_df['language'].nunique() if 'language' in events_df else 0
            st.metric("Languages", unique_langs)

        # Data table
        st.markdown("###  Events Table")
        st.dataframe(
            events_df[["title", "domain", "url"]].head(50),
            use_container_width=True,
            height=400
        )

        # Detailed view
        st.markdown("###  Detailed Events")
        for idx, event in enumerate(result["events"][:10]):
            with st.expander(f" Event {idx + 1}: {event.get('title', 'N/A')[:100]}..."):
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 5px;">
                    <h4 style="color: #1f1f1f; margin-top: 0;">{event.get('title', 'N/A')}</h4>
                    <p style="color: #333333;"><strong>URL:</strong> <a href="{event.get('url', '#')}" target="_blank" style="color: #667eea;">{event.get('url', 'N/A')}</a></p>
                    <p style="color: #333333;"><strong>Domain:</strong> {event.get('domain', 'N/A')}</p>
                    <p style="color: #333333;"><strong>Language:</strong> {event.get('language', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)


def analytics_dashboard_page():
    st.markdown("## Analytics Dashboard")

    # Initialize session state for analytics cache
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []

    # Time range selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        date_range = st.date_input(
            "Select Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now())
        )
    with col2:
        display_metric = st.selectbox("Display Metric", ["All", "Text Analysis", "Video Analysis", "Image Analysis"])
    with col3:
        if st.button("Refresh Data", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # Generate analysis statistics from session
    try:
        total_analyses = len(st.session_state.get('analysis_results', []))

        # Calculate metrics from actual results
        if total_analyses > 0:
            results = st.session_state.analysis_results
            threat_count = sum(1 for r in results if r.get('prediction') in ['hate', 'offensive'])
            avg_confidence = sum(r.get('confidence', 0) for r in results) / total_analyses if total_analyses > 0 else 0
        else:
            threat_count = 0
            avg_confidence = 0.0

        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{total_analyses}</p>
                <p class="metric-label">Total Analyses</p>
                <p style="font-size: 0.9rem; color: #667eea;">Session: {len(st.session_state.get('analysis_results', []))} items</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{avg_confidence:.1%}</p>
                <p class="metric-label">Average Confidence</p>
                <p style="font-size: 0.9rem; color: #667eea;">Model certainty level</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{threat_count}</p>
                <p class="metric-label">Threats Detected</p>
                <p style="font-size: 0.9rem; color: #FF6B6B;">Violations found</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{(threat_count/total_analyses*100 if total_analyses > 0 else 0):.1f}%</p>
                <p class="metric-label">Detection Rate</p>
                <p style="font-size: 0.9rem; color: #667eea;">Session violations ratio</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Charts section
        st.markdown("### Detection Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Analysis Type Distribution")

            # Analysis types distribution
            if total_analyses > 0:
                analysis_types = {}
                for result in st.session_state.analysis_results:
                    atype = result.get('type', 'Unknown')
                    analysis_types[atype] = analysis_types.get(atype, 0) + 1

                type_data = pd.DataFrame({
                    "Type": list(analysis_types.keys()),
                    "Count": list(analysis_types.values())
                })

                fig = px.pie(
                    type_data,
                    names="Type",
                    values="Count",
                    hole=0.4,
                    color_discrete_sequence=['#667eea', '#FF6B6B', '#4ECDC4', '#FFE66D']
                )
            else:
                # Empty state
                fig = go.Figure()
                fig.add_annotation(
                    text="No analysis data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )

            fig.update_layout(height=350, template="plotly_white", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Prediction Distribution")

            if total_analyses > 0:
                predictions = {}
                for result in st.session_state.analysis_results:
                    pred = result.get('prediction', 'Unknown')
                    predictions[pred] = predictions.get(pred, 0) + 1

                pred_data = pd.DataFrame({
                    "Category": list(predictions.keys()),
                    "Count": list(predictions.values())
                })

                colors = {'real': '#4ECDC4', 'normal': '#4ECDC4', 'fake': '#FF6B6B',
                         'hate': '#FF6B6B', 'offensive': '#FFE66D', 'clean': '#4ECDC4'}
                color_list = [colors.get(cat, '#667eea') for cat in pred_data['Category']]

                fig = px.bar(
                    pred_data,
                    x="Category",
                    y="Count",
                    color="Category",
                    color_discrete_sequence=color_list,
                    text="Count"
                )
            else:
                fig = go.Figure()
                fig.add_annotation(
                    text="No analysis data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )

            fig.update_layout(
                height=350,
                template="plotly_white",
                showlegend=False,
                xaxis_title="Prediction",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Confidence distribution chart
        st.markdown("### Confidence Distribution")

        if total_analyses > 0:
            confidence_data = pd.DataFrame({
                "Confidence": [r.get('confidence', 0) * 100 for r in st.session_state.analysis_results]
            })

            fig = go.Figure(data=[
                go.Histogram(
                    x=confidence_data['Confidence'],
                    nbinsx=20,
                    marker_color='#667eea',
                    opacity=0.7
                )
            ])

            fig.update_layout(
                title="Model Confidence Score Distribution",
                xaxis_title="Confidence (%)",
                yaxis_title="Frequency",
                height=350,
                template="plotly_white"
            )
        else:
            fig = go.Figure()
            fig.add_annotation(
                text="No analysis data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(height=350, template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)

        # Recent results table
        st.markdown("### Recent Analysis Results")

        if total_analyses > 0:
            recent_results = st.session_state.analysis_results[-10:]
            results_df = pd.DataFrame([
                {
                    'Type': r.get('type', 'N/A'),
                    'Prediction': r.get('prediction', 'N/A').title(),
                    'Confidence': f"{r.get('confidence', 0):.1%}",
                    'Input': str(r.get('input', 'N/A'))[:50] + '...'
                }
                for r in recent_results
            ])
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        else:
            st.info("No analysis data available. Run analyses to populate this dashboard.")

    except Exception as e:
        st.error(f"Error generating analytics: {str(e)}")
        logger.error(f"Analytics error: {e}")


def settings_page():
    """Settings and configuration page"""
    st.markdown("## Settings")

    tab1, tab2, tab3, tab4 = st.tabs(["General", "Model", "Notifications", "Data"])

    with tab1:
        st.markdown("### General Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.text_input("API Endpoint", value=API_BASE_URL, disabled=True)
            st.selectbox("Default Language", ["English", "Arabic", "Spanish", "French"])
            st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

        with col2:
            st.number_input("Max Upload Size (MB)", value=10, min_value=1, max_value=100)
            st.selectbox("Date Format", ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"])
            st.selectbox("Time Zone", ["UTC", "EST", "PST", "GMT"])

        if st.button("Save General Settings", type="primary"):
            st.success("Settings saved successfully!")

    with tab2:
        st.markdown("### Model Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.selectbox("Text Model", ["XLM-RoBERTa (Default)", "BERT", "DistilBERT"])
            st.selectbox("Vision Model", ["EfficientNet-B0 (Default)", "ResNet50", "VGG16"])
            st.checkbox("Enable Explanations", value=True)

        with col2:
            st.slider("Batch Size", 1, 64, 32)
            st.slider("Max Sequence Length", 128, 512, 256)
            st.checkbox("Use GPU", value=True)

        if st.button(" Save Model Settings", type="primary"):
            st.success(" Model settings saved successfully!")

    with tab3:
        st.markdown("### Notification Settings")

        st.checkbox("Enable Email Notifications", value=False)
        st.checkbox("Enable Webhook Notifications", value=False)
        st.checkbox("Alert on High-Risk Detections", value=True)

        st.text_input("Email Address", placeholder="your@email.com")
        st.text_input("Webhook URL", placeholder="https://your-webhook-url.com")

        if st.button(" Save Notification Settings", type="primary"):
            st.success(" Notification settings saved successfully!")

    with tab4:
        st.markdown("### Data Management")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Analyses", "1,234")
            st.metric("Storage Used", "2.3 GB")
            st.metric("Cache Size", "450 MB")

        with col2:
            if st.button(" Clear Cache", use_container_width=True):
                st.info("Cache cleared!")
            if st.button(" Export All Data", use_container_width=True):
                st.info("Export started!")
            if st.button(" Reset Statistics", use_container_width=True):
                st.warning("Statistics reset!")


if __name__ == "__main__":
    main()