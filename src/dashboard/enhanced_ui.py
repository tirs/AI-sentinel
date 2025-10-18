"""
Enhanced UI components and styling for the AI Sentinel Dashboard
This module provides premium visual components and styling without affecting core functionality
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from pathlib import Path


def inject_premium_css():
    """Inject premium CSS styling with modern design patterns"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* CSS Variables for Theme */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --warning-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        --danger-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);

        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
        --shadow-light: 0 8px 32px rgba(31, 38, 135, 0.15);
        --shadow-heavy: 0 15px 35px rgba(0, 0, 0, 0.1);

        --text-primary: #2c3e50;
        --text-secondary: #5a6c7d;
        --text-light: #ffffff;
        --card-bg: #ffffff;
        --border-radius: 16px;
        --border-radius-small: 8px;

        --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-bounce: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #e8eaed;
            --text-secondary: #b0b3b8;
            --card-bg: #1e1e1e;
        }
    }

    /* Global Overrides */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Custom Font Application */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Premium Header Component */
    .premium-header {
        background: var(--primary-gradient);
        padding: 3rem 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-heavy);
    }

    .premium-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.3;
    }

    .premium-header h1 {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        letter-spacing: -0.5px;
    }

    .premium-header p {
        color: #ffffff;
        font-size: 1.25rem;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
        position: relative;
        z-index: 1;
        opacity: 0.95;
    }

    .premium-header .header-badge {
        position: absolute;
        top: 2rem;
        right: 2rem;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(226, 232, 240, 0.5);
        border-radius: var(--border-radius);
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-light);
        transition: var(--transition-smooth);
        position: relative;
        overflow: hidden;
    }

    @media (prefers-color-scheme: dark) {
        .glass-card {
            background: #1e1e1e;
            border-color: rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--primary-gradient);
        border-radius: var(--border-radius) var(--border-radius) 0 0;
    }

    .glass-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        border-color: rgba(102, 126, 234, 0.3);
    }

    @media (prefers-color-scheme: dark) {
        .glass-card:hover {
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            border-color: rgba(139, 157, 255, 0.3);
        }
    }

    .glass-card h2 {
        color: var(--text-primary);
        margin-top: 0;
        font-weight: 700;
    }

    .glass-card h3, .glass-card h4 {
        color: var(--text-primary);
        margin-top: 0;
        font-weight: 700;
    }

    .glass-card p, .glass-card ul, .glass-card li {
        color: var(--text-secondary);
        line-height: 1.6;
    }

    /* Premium Metric Cards */
    .metric-card-premium {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: var(--border-radius);
        padding: 2rem;
        text-align: center;
        transition: var(--transition-bounce);
        position: relative;
        overflow: hidden;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }

    @media (prefers-color-scheme: dark) {
        .metric-card-premium {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            border-color: rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
    }

    .metric-card-premium::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        transition: left 0.5s ease;
    }

    .metric-card-premium:hover::before {
        left: 100%;
    }

    .metric-card-premium:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }

    @media (prefers-color-scheme: dark) {
        .metric-card-premium:hover {
            box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
            border-color: #8b9dff;
        }
    }

    .metric-value-premium {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 1;
    }

    .metric-label-premium {
        font-size: 1rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        position: relative;
        z-index: 1;
    }

    .metric-trend {
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }

    .trend-up { color: #10b981; }
    .trend-down { color: #ef4444; }
    .trend-neutral { color: #6b7280; }

    /* Enhanced Buttons */
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: var(--border-radius-small);
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: var(--transition-smooth);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }

    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* Premium Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 0 var(--border-radius) var(--border-radius) 0;
    }

    .css-1d391kg .css-1v0mbdj {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: var(--border-radius-small);
        margin: 0.5rem 0;
        transition: var(--transition-smooth);
    }

    .css-1d391kg .css-1v0mbdj:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }

    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: 1rem;
        transition: var(--transition-smooth);
    }

    .status-online {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }

    .status-offline {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }

    .status-indicator:hover {
        transform: scale(1.05);
    }

    /* Pulse Animation */
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }

    /* Alert Boxes Enhanced */
    .alert-enhanced {
        padding: 1.5rem;
        border-radius: var(--border-radius);
        margin: 1rem 0;
        font-weight: 500;
        position: relative;
        border-left: 4px solid;
        backdrop-filter: blur(10px);
        transition: var(--transition-smooth);
    }

    .alert-enhanced:hover {
        transform: translateX(5px);
    }

    .alert-success-enhanced {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.05));
        border-left-color: #10b981;
        color: #065f46;
    }

    @media (prefers-color-scheme: dark) {
        .alert-success-enhanced {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.1));
            color: #86efac;
        }
    }

    .alert-danger-enhanced {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.05));
        border-left-color: #ef4444;
        color: #991b1b;
    }

    @media (prefers-color-scheme: dark) {
        .alert-danger-enhanced {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.1));
            color: #fca5a5;
        }
    }

    .alert-warning-enhanced {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(217, 119, 6, 0.05));
        border-left-color: #f59e0b;
        color: #92400e;
    }

    @media (prefers-color-scheme: dark) {
        .alert-warning-enhanced {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.1));
            color: #fcd34d;
        }
    }

    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: var(--primary-gradient);
        border-radius: 10px;
    }

    /* Enhanced File Uploader */
    .uploadedFile {
        border: 2px dashed #cbd5e1;
        border-radius: var(--border-radius);
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        transition: var(--transition-smooth);
        position: relative;
        overflow: hidden;
        color: #2c3e50;
    }

    @media (prefers-color-scheme: dark) {
        .uploadedFile {
            border-color: #4b5563;
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: #e8eaed;
        }
    }

    .uploadedFile::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: var(--primary-gradient);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .uploadedFile:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    }

    @media (prefers-color-scheme: dark) {
        .uploadedFile:hover {
            border-color: #8b9dff;
            background: linear-gradient(135deg, #3a4d66 0%, #42596e 100%);
        }
    }

    .uploadedFile:hover::before {
        opacity: 0.05;
    }

    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(248, 250, 252, 0.8);
        padding: 8px;
        border-radius: var(--border-radius);
        backdrop-filter: blur(10px);
    }

    @media (prefers-color-scheme: dark) {
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(44, 62, 80, 0.8);
        }
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: var(--border-radius-small);
        padding: 12px 24px;
        font-weight: 600;
        transition: var(--transition-smooth);
        border: 1px solid transparent;
        color: #2c3e50;
    }

    @media (prefers-color-scheme: dark) {
        .stTabs [data-baseweb="tab"] {
            color: #e8eaed;
        }
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.2);
    }

    @media (prefers-color-scheme: dark) {
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(139, 157, 255, 0.1);
            border-color: rgba(139, 157, 255, 0.2);
        }
    }

    .stTabs [aria-selected="true"] {
        background: var(--primary-gradient);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
    }

    /* Enhanced Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: var(--border-radius-small);
        font-weight: 600;
        border: 1px solid #e2e8f0;
        transition: var(--transition-smooth);
        color: #2c3e50;
    }

    @media (prefers-color-scheme: dark) {
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            border-color: #4b5563;
            color: #e8eaed;
        }
    }

    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-color: #667eea;
        transform: translateY(-1px);
    }

    @media (prefers-color-scheme: dark) {
        .streamlit-expanderHeader:hover {
            background: linear-gradient(135deg, #3a4d66 0%, #42596e 100%);
            border-color: #8b9dff;
        }
    }

    /* Loading Animations */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Fade In Animation */
    .fade-in {
        animation: fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Slide In Animations */
    .slide-in-left {
        animation: slideInLeft 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }

    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    .slide-in-right {
        animation: slideInRight 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }

    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Floating Action Button */
    .fab {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: 60px;
        height: 60px;
        background: var(--primary-gradient);
        border-radius: 50%;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.5rem;
        cursor: pointer;
        transition: var(--transition-bounce);
        z-index: 1000;
    }

    .fab:hover {
        transform: translateY(-5px) scale(1.1);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.5);
    }

    /* Dark Mode Support */
    @media (prefers-color-scheme: dark) {
        .glass-card {
            background: rgba(30, 41, 59, 0.8);
            color: #f1f5f9;
        }

        .metric-card-premium {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-color: #475569;
        }

        .metric-label-premium {
            color: #cbd5e1;
        }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .premium-header {
            padding: 2rem 1rem;
        }

        .premium-header h1 {
            font-size: 2rem;
        }

        .glass-card {
            padding: 1.5rem;
        }

        .metric-card-premium {
            padding: 1.5rem;
        }

        .metric-value-premium {
            font-size: 2rem;
        }
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    </style>
    """, unsafe_allow_html=True)


def create_premium_header(title: str, subtitle: str, version: str = "v1.0.0"):
    """Create a premium header with glassmorphism effects"""
    st.markdown(f"""
    <div class="premium-header fade-in">
        <div class="header-badge">
            {version}
        </div>
        <h1>🛡️ {title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def create_status_indicator(is_online: bool, label: str = "API Status"):
    """Create an animated status indicator"""
    status_class = "status-online" if is_online else "status-offline"
    status_text = "Connected" if is_online else "Disconnected"
    dot_color = "#10b981" if is_online else "#f59e0b"

    st.markdown(f"""
    <div class="{status_class} status-indicator">
        <div class="pulse-dot" style="background-color: {dot_color};"></div>
        <strong>{label}:</strong> {status_text}
    </div>
    """, unsafe_allow_html=True)


def create_premium_metric_card(value: str, label: str, trend: str = None, trend_type: str = "neutral"):
    """Create premium metric cards with animations"""
    trend_html = ""
    if trend:
        trend_class = f"trend-{trend_type}"
        trend_icon = "↗" if trend_type == "up" else "↘" if trend_type == "down" else "→"
        trend_html = f'<div class="metric-trend {trend_class}">{trend_icon} {trend}</div>'

    return f"""
    <div class="metric-card-premium slide-in-left">
        <div class="metric-value-premium">{value}</div>
        <div class="metric-label-premium">{label}</div>
        {trend_html}
    </div>
    """


def create_glass_card(title: str, content: str, icon: str = "📊"):
    """Create glassmorphism cards"""
    return f"""
    <div class="glass-card slide-in-right">
        <h3>{icon} {title}</h3>
        {content}
    </div>
    """


def create_enhanced_alert(message: str, alert_type: str = "success"):
    """Create enhanced alert boxes"""
    return f"""
    <div class="alert-{alert_type}-enhanced alert-enhanced">
        {message}
    </div>
    """


def create_feature_grid():
    """Create a responsive feature grid"""
    return """
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin: 2rem 0;">
        <div class="glass-card">
            <h3>📝 Advanced Text Analysis</h3>
            <p>Multi-language hate speech detection with explainable AI powered by transformer models.</p>
            <ul>
                <li>50+ language support</li>
                <li>Real-time processing</li>
                <li>LIME explanations</li>
                <li>Confidence scoring</li>
            </ul>
        </div>

        <div class="glass-card">
            <h3>🖼️ Computer Vision</h3>
            <p>State-of-the-art deepfake detection using advanced neural networks and visual analysis.</p>
            <ul>
                <li>Deepfake detection</li>
                <li>Visual explanations</li>
                <li>Batch processing</li>
                <li>High accuracy models</li>
            </ul>
        </div>

        <div class="glass-card">
            <h3>🌍 Global Intelligence</h3>
            <p>Real-world event correlation using GDELT database for comprehensive threat analysis.</p>
            <ul>
                <li>Real-time events</li>
                <li>Geographic mapping</li>
                <li>Trend analysis</li>
                <li>Historical data</li>
            </ul>
        </div>
    </div>
    """


def create_floating_action_button():
    """Create a floating action button for quick actions"""
    st.markdown("""
    <div class="fab" onclick="window.scrollTo({top: 0, behavior: 'smooth'});" title="Back to top">
        ↑
    </div>
    """, unsafe_allow_html=True)


def create_premium_charts():
    """Create enhanced chart configurations for better visual appeal"""

    def get_chart_theme():
        return {
            'layout': {
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'font': {'family': 'Inter, sans-serif', 'color': '#2c3e50'},
                'colorway': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'],
                'margin': {'l': 40, 'r': 40, 't': 60, 'b': 40},
            }
        }

    return get_chart_theme()


def create_loading_spinner(text: str = "Loading..."):
    """Create a loading spinner with text"""
    return f"""
    <div style="display: flex; align-items: center; justify-content: center; padding: 2rem;">
        <div class="loading-spinner"></div>
        <span style="margin-left: 1rem; font-weight: 500; color: #667eea;">{text}</span>
    </div>
    """


def create_breadcrumb_navigation(pages: list):
    """Create breadcrumb navigation"""
    breadcrumb_items = []
    for i, page in enumerate(pages):
        if i == len(pages) - 1:
            breadcrumb_items.append(f'<span style="color: #667eea; font-weight: 600;">{page}</span>')
        else:
            breadcrumb_items.append(f'<span style="color: #64748b;">{page}</span>')

    breadcrumb_html = ' <span style="color: #cbd5e1;">→</span> '.join(breadcrumb_items)

    return f"""
    <div style="margin-bottom: 1rem; padding: 1rem; background: rgba(248, 250, 252, 0.8);
                border-radius: 8px; font-size: 0.875rem;">
        {breadcrumb_html}
    </div>
    """


def create_progress_ring(percentage: float, size: int = 120, stroke_width: int = 8):
    """Create an animated progress ring"""
    radius = (size - stroke_width) / 2
    circumference = 2 * 3.14159 * radius
    offset = circumference - (percentage / 100) * circumference

    return f"""
    <div style="display: flex; justify-content: center; margin: 1rem 0;">
        <svg width="{size}" height="{size}" style="transform: rotate(-90deg);">
            <circle
                cx="{size/2}"
                cy="{size/2}"
                r="{radius}"
                stroke="#e5e7eb"
                stroke-width="{stroke_width}"
                fill="none"
            />
            <circle
                cx="{size/2}"
                cy="{size/2}"
                r="{radius}"
                stroke="url(#gradient)"
                stroke-width="{stroke_width}"
                fill="none"
                stroke-dasharray="{circumference}"
                stroke-dashoffset="{offset}"
                stroke-linecap="round"
                style="transition: stroke-dashoffset 1s ease-in-out;"
            />
            <defs>
                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
                </linearGradient>
            </defs>
            <text
                x="50%"
                y="50%"
                text-anchor="middle"
                dy="0.3em"
                style="font-size: 1.5rem; font-weight: 700; fill: #667eea; transform: rotate(90deg); transform-origin: center;"
            >
                {percentage:.0f}%
            </text>
        </svg>
    </div>
    """


def create_animated_counter(end_value: int, duration: float = 2.0, prefix: str = "", suffix: str = ""):
    """Create an animated counter using JavaScript"""
    counter_id = f"counter_{hash(str(end_value) + prefix + suffix) % 10000}"

    return f"""
    <div style="text-align: center; margin: 1rem 0;">
        <span id="{counter_id}" style="font-size: 2.5rem; font-weight: 800;
                                       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                                       background-clip: text;">
            {prefix}0{suffix}
        </span>
    </div>

    <script>
    function animateCounter(id, start, end, duration, prefix, suffix) {{
        const element = document.getElementById(id);
        const range = end - start;
        const increment = range / (duration * 60); // 60 FPS
        let current = start;

        const timer = setInterval(() => {{
            current += increment;
            if (current >= end) {{
                current = end;
                clearInterval(timer);
            }}
            element.textContent = prefix + Math.floor(current) + suffix;
        }}, 1000 / 60);
    }}

    animateCounter('{counter_id}', 0, {end_value}, {duration}, '{prefix}', '{suffix}');
    </script>
    """


def create_notification_toast(message: str, toast_type: str = "success", duration: int = 3000):
    """Create a notification toast"""
    colors = {
        "success": "#10b981",
        "error": "#ef4444",
        "warning": "#f59e0b",
        "info": "#3b82f6"
    }

    color = colors.get(toast_type, colors["info"])
    toast_id = f"toast_{hash(message) % 10000}"

    return f"""
    <div id="{toast_id}" style="
        position: fixed;
        top: 2rem;
        right: 2rem;
        background: white;
        border: 1px solid {color};
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 1rem 1.5rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        z-index: 1001;
        transform: translateX(400px);
        transition: transform 0.3s ease;
        max-width: 300px;
    ">
        <div style="color: {color}; font-weight: 600; margin-bottom: 0.25rem;">
            {toast_type.title()} Notification
        </div>
        <div style="color: #64748b; font-size: 0.875rem;">
            {message}
        </div>
    </div>

    <script>
    setTimeout(() => {{
        document.getElementById('{toast_id}').style.transform = 'translateX(0)';
    }}, 100);

    setTimeout(() => {{
        const toast = document.getElementById('{toast_id}');
        if (toast) {{
            toast.style.transform = 'translateX(400px)';
            setTimeout(() => toast.remove(), 300);
        }}
    }}, {duration});
    </script>
    """


def create_interactive_timeline(events: list):
    """Create an interactive timeline component"""
    timeline_html = '<div style="position: relative; margin: 2rem 0;">'

    for i, event in enumerate(events):
        position = "left" if i % 2 == 0 else "right"
        timeline_html += f"""
        <div style="
            position: relative;
            margin: 2rem 0;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            border-left: 4px solid #667eea;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        " onmouseover="this.style.transform='translateX(10px)'; this.style.boxShadow='0 8px 25px rgba(102, 126, 234, 0.2)';"
           onmouseout="this.style.transform='translateX(0)'; this.style.boxShadow='0 4px 15px rgba(0, 0, 0, 0.1)';">

            <div style="
                position: absolute;
                left: -12px;
                top: 1rem;
                width: 20px;
                height: 20px;
                background: #667eea;
                border-radius: 50%;
                border: 4px solid white;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            "></div>

            <h4 style="margin: 0 0 0.5rem 0; color: #1e293b; font-weight: 700;">
                {event.get('title', 'Event')}
            </h4>
            <p style="margin: 0; color: #64748b; font-size: 0.875rem;">
                {event.get('description', 'No description available')}
            </p>
            <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #94a3b8;">
                {event.get('timestamp', 'Unknown time')}
            </div>
        </div>
        """

    timeline_html += '</div>'
    return timeline_html


# Additional utility functions for enhanced UI components
def get_icon_for_category(category: str) -> str:
    """Get appropriate icon for different categories"""
    icons = {
        "normal": "🟢",
        "hate": "🔴",
        "offensive": "🟡",
        "real": "✅",
        "fake": "❌",
        "warning": "⚠️",
        "info": "ℹ️",
        "success": "✅",
        "error": "❌"
    }
    return icons.get(category.lower(), "📊")


def format_large_number(num: int) -> str:
    """Format large numbers with appropriate suffixes"""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def get_confidence_color(confidence: float) -> str:
    """Get color based on confidence level"""
    if confidence >= 0.8:
        return "#10b981"  # Green
    elif confidence >= 0.6:
        return "#f59e0b"  # Yellow
    else:
        return "#ef4444"  # Red