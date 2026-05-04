import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="✨ EDA Dashboard | Interactive",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== THEME TOGGLE ==========
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

# ========== SIDEBAR FIXED CSS (black text, light background) ==========
# ========== SIDEBAR FIXED CSS (bold black text, shaded upload area) ==========
sidebar_fixed_css = """
<style>
    /* Force sidebar light background */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fc 100%) !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    /* All sidebar text – bold and black */
    .css-1d391kg, .css-1d391kg *,
    [data-testid="stSidebar"], [data-testid="stSidebar"] * {
        color: #1a1a1a !important;
        font-weight: 600 !important;   /* makes text bold */
    }
    /* Sidebar headings */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3,
    .css-1d391kg .stMarkdown h3, .css-1d391kg .stMarkdown p,
    .css-1d391kg label, .css-1d391kg li, .css-1d391kg .stCaption {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    /* Sidebar title */
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700 !important;
        color: #1a1a1a !important;
        text-align: center;
    }
    /* Sidebar button (toggle) – keep text white */
    .css-1d391kg .stButton > button {
        background: linear-gradient(120deg, #1f77b4, #2a9d8f);
        color: white !important;
        font-weight: 600 !important;
    }
    /* ===== FILE UPLOADER STYLING ===== */
    /* The entire upload container */
    [data-testid="stFileUploader"] {
        background-color: #f1f5f9 !important;   /* soft shaded background */
        border-radius: 16px !important;
        padding: 0.5rem !important;
        border: 1px solid #cbd5e1 !important;
        transition: all 0.2s ease;
    }
    /* Hover effect on the upload area */
    [data-testid="stFileUploader"]:hover {
        background-color: #e2e8f0 !important;
        border-color: #94a3b8 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    /* The "Browse files" button */
    [data-testid="stFileUploader"] button {
        background-color: #e2e8f0 !important;
        color: #1e293b !important;
        border: none !important;
        border-radius: 40px !important;
        font-weight: 600 !important;
        padding: 0.4rem 1rem !important;
        transition: background 0.2s;
    }
    /* Hover on the button */
    [data-testid="stFileUploader"] button:hover {
        background-color: #cbd5e1 !important;
        color: #0f172a !important;
    }
    /* The drag‑and‑drop text */
    [data-testid="stFileUploader"] small {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    /* Remove any default black border or background from uploader */
    .stFileUploader > div {
        background: transparent !important;
    }
</style>
"""
st.markdown(sidebar_fixed_css, unsafe_allow_html=True)

# ========== DYNAMIC CSS FOR MAIN CONTENT (theme toggles) ==========
light_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f9fafb 0%, #eef2f7 100%); }
    .metric-card { background: white; border-radius: 20px; padding: 1.2rem 1rem; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.05); transition: transform 0.2s, box-shadow 0.2s; border: 1px solid rgba(255,255,255,0.3); }
    .metric-card:hover { transform: translateY(-5px); box-shadow: 0 12px 28px rgba(0,0,0,0.1); }
    .metric-value { font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #1f77b4, #4c9ed9); -webkit-background-clip: text; background-clip: text; color: transparent; }
    .metric-label { font-size: 0.85rem; font-weight: 500; color: #4a5568; margin-top: 0.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: rgba(255,255,255,0.7); backdrop-filter: blur(4px); padding: 0.5rem; border-radius: 60px; }
    .stTabs [data-baseweb="tab"] { border-radius: 40px; padding: 0.5rem 1.2rem; font-weight: 600; color: #2c3e50; transition: all 0.2s; }
    .stTabs [data-baseweb="tab"]:hover { background-color: rgba(31,119,180,0.1); }
    .stTabs [aria-selected="true"] { background: linear-gradient(120deg, #1f77b4, #2a9d8f); color: white !important; }
    h1, h2, h3 { font-weight: 700; color: #1a1a1a !important; background: none !important; }
    .stButton > button { border-radius: 40px; background: linear-gradient(120deg, #1f77b4, #2a9d8f); color: white; font-weight: 600; border: none; padding: 0.5rem 1.2rem; transition: transform 0.1s; }
    .stButton > button:hover { transform: scale(1.02); background: linear-gradient(120deg, #2a9d8f, #1f77b4); }
    .dataframe { border-radius: 16px; overflow: hidden; border: none; font-size: 0.8rem; }
    .streamlit-expanderHeader { background: rgba(31,119,180,0.05); border-radius: 12px; font-weight: 600; }
    .custom-divider { background: linear-gradient(90deg, #1f77b4, #2a9d8f, #e9c46a); height: 3px; border-radius: 3px; margin: 1rem 0; }
    /* Empty state heading – black */
    .empty-state h2 {
        color: #1a1a1a !important;
    }
</style>
"""

dark_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #1a1e2c 0%, #0f1219 100%); }
    .metric-card { background: #2d3748; border-radius: 20px; padding: 1.2rem 1rem; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.3); transition: transform 0.2s, box-shadow 0.2s; border: 1px solid #4a5568; }
    .metric-card:hover { transform: translateY(-5px); box-shadow: 0 12px 28px rgba(0,0,0,0.5); }
    .metric-value { font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #63b3ed, #90cdf4); -webkit-background-clip: text; background-clip: text; color: transparent; }
    .metric-label { font-size: 0.85rem; font-weight: 500; color: #cbd5e0; margin-top: 0.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: #2d3748; backdrop-filter: blur(4px); padding: 0.5rem; border-radius: 60px; }
    .stTabs [data-baseweb="tab"] { border-radius: 40px; padding: 0.5rem 1.2rem; font-weight: 600; color: #e2e8f0; transition: all 0.2s; }
    .stTabs [data-baseweb="tab"]:hover { background-color: rgba(99,179,237,0.2); }
    .stTabs [aria-selected="true"] { background: linear-gradient(120deg, #4299e1, #48bb78); color: white !important; }
    h1, h2, h3 { font-weight: 700; color: #f7fafc !important; background: none !important; }
    .stButton > button { border-radius: 40px; background: linear-gradient(120deg, #4299e1, #48bb78); color: white; font-weight: 600; border: none; padding: 0.5rem 1.2rem; transition: transform 0.1s; }
    .stButton > button:hover { transform: scale(1.02); background: linear-gradient(120deg, #48bb78, #4299e1); }
    .dataframe { border-radius: 16px; overflow: hidden; border: none; font-size: 0.8rem; background-color: #2d3748; color: #e2e8f0; }
    .streamlit-expanderHeader { background: rgba(66,153,225,0.2); border-radius: 12px; font-weight: 600; color: #e2e8f0; }
    .custom-divider { background: linear-gradient(90deg, #4299e1, #48bb78, #f6ad55); height: 3px; border-radius: 3px; margin: 1rem 0; }
    /* Empty state heading – keep black even in dark mode */
    .empty-state h2 {
        color: #1a1a1a !important;
    }
</style>
"""

if st.session_state.theme == "light":
    st.markdown(light_css, unsafe_allow_html=True)
else:
    st.markdown(dark_css, unsafe_allow_html=True)

# ========== SIDEBAR CONTENT ==========
with st.sidebar:
    st.markdown("<div class='sidebar-title'>📊 Exploratory Data Lab</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.button("🌓 Toggle Light/Dark Mode", on_click=toggle_theme, use_container_width=True)
    
    st.markdown("### 📂 Upload your CSV")
    uploaded_file = st.file_uploader("", type="csv", help="Any CSV file – the dashboard adapts automatically")
    
    st.markdown("---")
    st.markdown("### 🎨 Color Legend")
    st.markdown("🔵 **Numerical** – Blue shades  \n🟢 **Categorical** – Green shades  \n🟠 **Relationships** – Orange/Red  \n🟣 **Missing** – Purple")
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("- Hover over charts for details  \n- Use multi‑select to pick columns  \n- Pairplot may be slow for large data")
    st.markdown("---")
    st.caption("Built with Streamlit · Data magic ✨")

# ========== MAIN CONTENT ==========
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>📊 Interactive EDA Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 2rem;'>Upload any CSV – automatic insights & stunning visuals</p>", unsafe_allow_html=True)

if uploaded_file is not None:
    with st.spinner("🔍 Loading and analyzing your data..."):
        df = pd.read_csv(uploaded_file)
    
    num_cols_list = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols_list = df.select_dtypes(include=['object', 'category']).columns.tolist()
    missing_total = df.isnull().sum().sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem;">📄</div>
            <div class="metric-value">{df.shape[0]}</div>
            <div class="metric-label">Rows</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem;">📊</div>
            <div class="metric-value">{df.shape[1]}</div>
            <div class="metric-label">Columns</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem;">⚠️</div>
            <div class="metric-value">{missing_total}</div>
            <div class="metric-label">Missing Cells</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem;">🔢</div>
            <div class="metric-value">{len(num_cols_list)} / {len(cat_cols_list)}</div>
            <div class="metric-label">Num / Cat Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📋 Data Preview", "📈 Distributions", "🔗 Relationships", "🧹 Missing", "📊 Correlation"]
    )
    
    with tab1:
        st.subheader("📋 Raw Data (first 100 rows)")
        st.dataframe(df.head(100), use_container_width=True)
        with st.expander("📐 Numerical Summary"):
            st.dataframe(df.describe(), use_container_width=True)
        with st.expander("🏷️ Categorical Summary"):
            st.dataframe(df.describe(include=['object', 'category']), use_container_width=True)
    
    with tab2:
        if len(num_cols_list) > 0:
            st.markdown("#### 🔵 Numerical Features")
            selected_nums = st.multiselect("Select columns", num_cols_list, default=num_cols_list[:min(3, len(num_cols_list))], key="num_multi")
            for col in selected_nums:
                fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}", 
                                   color_discrete_sequence=["#1f77b4"], template="plotly_white")
                fig.update_layout(bargap=0.05)
                st.plotly_chart(fig, use_container_width=True)
        
        if len(cat_cols_list) > 0:
            st.markdown("#### 🟢 Categorical Features")
            selected_cats = st.multiselect("Select categorical columns", cat_cols_list, default=cat_cols_list[:min(3, len(cat_cols_list))], key="cat_multi")
            for col in selected_cats:
                freq = df[col].value_counts().reset_index()
                freq.columns = [col, 'count']
                fig = px.bar(freq, x=col, y='count', title=f"Counts of {col}", 
                             color_discrete_sequence=["#2ca02c"], template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### 🟠 Numerical vs Numerical (Scatter)")
        if len(num_cols_list) >= 2:
            colx, coly = st.columns(2)
            with colx:
                x_axis = st.selectbox("X axis", num_cols_list, key="x_scatter")
            with coly:
                y_axis = st.selectbox("Y axis", num_cols_list, key="y_scatter")
            color_by = st.selectbox("Color by", [None] + cat_cols_list + num_cols_list, key="color_scatter")
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, 
                             title=f"{y_axis} vs {x_axis}", template="plotly_white",
                             color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 📦 Boxplots (Category vs Numerical)")
        if len(cat_cols_list) > 0 and len(num_cols_list) > 0:
            col_cat, col_num = st.columns(2)
            with col_cat:
                cat_box = st.selectbox("Categorical", cat_cols_list, key="cat_box")
            with col_num:
                num_box = st.selectbox("Numerical", num_cols_list, key="num_box")
            fig = px.box(df, x=cat_box, y=num_box, color=cat_box, 
                         title=f"{num_box} by {cat_box}", template="plotly_white",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("#### 🟣 Missing Values Analysis")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ['Column', 'Missing Count']
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        if len(missing_df) > 0:
            fig = px.bar(missing_df, x='Column', y='Missing Count', 
                         title="Missing Values per Column", 
                         color='Missing Count', color_continuous_scale="Purples",
                         template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(missing_df)
        else:
            st.success("✅ No missing values! Your data is complete.")
    
    with tab5:
        st.markdown("#### 🔥 Correlation Heatmap (Numerical)")
        if len(num_cols_list) >= 2:
            corr = df[num_cols_list].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", 
                            color_continuous_scale="RdBu_r", 
                            title="Correlation Matrix",
                            width=800, height=700,
                            zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least two numerical columns for correlation.")
        
        if 2 <= len(num_cols_list) <= 6:
            st.markdown("#### 🧩 Pairplot (Seaborn)")
            if st.button("✨ Generate Pairplot"):
                with st.spinner("Rendering pairplot (might take a moment)..."):
                    pair_fig = sns.pairplot(df[num_cols_list], diag_kind='kde', 
                                            plot_kws={'alpha':0.6, 's':30},
                                            diag_kws={'fill':True})
                    st.pyplot(pair_fig)

else:
    st.markdown("""
    <div class="empty-state" style="text-align: center; padding: 4rem 2rem; background: white; border-radius: 32px; margin-top: 2rem; box-shadow: 0 8px 24px rgba(0,0,0,0.05);">
        <div style="font-size: 4rem;">📊✨</div>
        <h2 style="margin: 1rem 0; color: #1a1a1a;">Ready to explore your data?</h2>
        <p style="color: #5a6e8a;">Upload a CSV file from the sidebar and watch the magic happen.<br>Automatic visualizations, statistics, and insights.</p>
    </div>
    """, unsafe_allow_html=True)