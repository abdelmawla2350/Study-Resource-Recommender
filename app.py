"""
Study Resource Recommender - Streamlit App
==========================================
A web application that recommends YouTube educational videos
based on student quiz performance using Machine Learning.

Author: Eman-Omar-Yehia-Abdelmawla
Project: Study Resource Recommender (Educational Track)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Study Resource Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0D47A1; /* Dark blue */
        text-align: center;
        margin-bottom: 1rem;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #333; /* Darker gray */
        text-align: center;
        margin-bottom: 2rem;
    }

    .metric-card {
        background-color: #BBDEFB; /* Strong light blue */
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: #0D47A1;
        font-weight: 600;
    }

    .skill-mastered {
        background-color: #81C784; /* Visible green */
        border-left: 6px solid #2E7D32;
        padding: 12px;
        margin: 8px 0;
        border-radius: 5px;
        color: #1B5E20;
        font-weight: 600;
    }

    .skill-learning {
        background-color: #FFD54F; /* Strong yellow */
        border-left: 6px solid #FF8F00;
        padding: 12px;
        margin: 8px 0;
        border-radius: 5px;
        color: #E65100;
        font-weight: 600;
    }

    .skill-needs-help {
        background-color: #E57373; /* Strong red */
        border-left: 6px solid #B71C1C;
        padding: 12px;
        margin: 8px 0;
        border-radius: 5px;
        color: #7F0000;
        font-weight: 600;
    }

    .video-card {
        background-color: #64B5F6; /* Strong blue */
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 6px solid #0D47A1;
        color: #0D47A1;
        font-weight: 600;
    }

    .priority-high {
        color: #B71C1C;
        font-weight: bold;
    }

    .priority-medium {
        color: #FF6F00;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)



# ============== Load Models and Data ==============

@st.cache_resource
def load_recommender():
    """Load the recommender package."""
    try:
        with open('recommender_package.pkl', 'rb') as f:
            package = pickle.load(f)
        return package
    except FileNotFoundError:
        st.error("❌ Recommender package not found. Please ensure 'recommender_package.pkl' is in the app directory.")
        return None


@st.cache_data
def load_student_data():
    """Load student performance data."""
    try:
        df = pd.read_csv('student_data_for_app.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Student data not found. Please ensure 'student_data_for_app.csv' is in the app directory.")
        return None


def add_engineered_features(df):
    """Add engineered features if they don't exist."""
    df = df.copy()
    
    if 'efficiency_score' not in df.columns:
        df['efficiency_score'] = df['total_correct'] / (df['total_hints_used'] + 1)
    
    if 'struggle_score' not in df.columns:
        df['struggle_score'] = (
            (1 - df['accuracy']) * 0.4 + 
            df['avg_hint_ratio'] * 0.3 + 
            (df['avg_attempts'] / df['avg_attempts'].max()) * 0.3
        )
    
    if 'speed_score' not in df.columns:
        df['speed_score'] = 1 - (df['avg_response_time'] / df['avg_response_time'].max()).clip(0, 1)
    
    if 'hint_dependency' not in df.columns:
        df['hint_dependency'] = (df['avg_hint_ratio'] + df['pct_hint_first']) / 2
    
    if 'attempts_per_correct' not in df.columns:
        df['attempts_per_correct'] = df['total_attempts'] / (df['total_correct'] + 1)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    
    return df


def predict_mastery(student_features, package):
    """Predict mastery level for given features."""
    model = package['model']
    scaler = package['scaler']
    label_encoder = package['label_encoder']
    feature_columns = package['feature_config']['feature_columns']
    needs_scaling = package['feature_config'].get('needs_scaling', True)
    
    features = student_features[feature_columns].copy()
    
    if needs_scaling:
        features_processed = scaler.transform(features)
    else:
        features_processed = features.values
    
    predictions_encoded = model.predict(features_processed)
    probabilities = model.predict_proba(features_processed)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    
    return predictions, probabilities


def get_videos_for_skill(skill_name, skill_video_mapping, top_n=5):
    """Get top videos for a specific skill."""
    videos = skill_video_mapping[
        skill_video_mapping['skill_name'] == skill_name
    ].copy()
    
    if len(videos) == 0:
        return pd.DataFrame()
    
    if 'keyword_score' in videos.columns and 'views' in videos.columns:
        videos['ranking_score'] = (
            videos['keyword_score'] * 0.6 + 
            (videos['views'] / videos['views'].max()) * 0.4
        )
        videos = videos.sort_values('ranking_score', ascending=False)
    
    return videos.head(top_n)


def get_recommendations(student_data, package, top_skills=5, videos_per_skill=3):
    """Get video recommendations for a student."""
    predictions, probabilities = predict_mastery(student_data, package)
    label_encoder = package['label_encoder']
    skill_video_mapping = package['skill_video_mapping']
    
    # Create analysis DataFrame
    analysis = student_data[['skill_name']].copy()
    analysis['predicted_mastery'] = predictions
    analysis['accuracy'] = student_data['accuracy'].values
    
    for i, label in enumerate(label_encoder.classes_):
        analysis[f'prob_{label}'] = probabilities[:, i]
    
    analysis['confidence'] = probabilities.max(axis=1)
    
    # Categorize skills
    weak_skills = analysis[analysis['predicted_mastery'] == 'needs_help'].sort_values('accuracy').head(top_skills)
    learning_skills = analysis[analysis['predicted_mastery'] == 'learning'].sort_values('accuracy').head(top_skills)
    mastered_skills = analysis[analysis['predicted_mastery'] == 'mastered']
    
    # Get video recommendations
    video_recommendations = []
    
    for _, row in weak_skills.iterrows():
        skill = row['skill_name']
        videos = get_videos_for_skill(skill, skill_video_mapping, top_n=videos_per_skill)
        
        if len(videos) > 0:
            for _, video in videos.iterrows():
                video_recommendations.append({
                    'skill_name': skill,
                    'student_accuracy': row['accuracy'],
                    'mastery_level': 'needs_help',
                    'priority': 'HIGH',
                    'video_title': video.get('video_title', 'N/A'),
                    'video_id': video.get('video_id', 'N/A'),
                    'views': video.get('views', 0),
                    'likes': video.get('likes', 0)
                })
    
    for _, row in learning_skills.iterrows():
        skill = row['skill_name']
        videos = get_videos_for_skill(skill, skill_video_mapping, top_n=2)
        
        if len(videos) > 0:
            for _, video in videos.iterrows():
                video_recommendations.append({
                    'skill_name': skill,
                    'student_accuracy': row['accuracy'],
                    'mastery_level': 'learning',
                    'priority': 'MEDIUM',
                    'video_title': video.get('video_title', 'N/A'),
                    'video_id': video.get('video_id', 'N/A'),
                    'views': video.get('views', 0),
                    'likes': video.get('likes', 0)
                })
    
    recommendations_df = pd.DataFrame(video_recommendations)
    
    summary = {
        'total_skills_analyzed': len(analysis),
        'skills_mastered': len(mastered_skills),
        'skills_learning': len(learning_skills),
        'skills_need_help': len(weak_skills),
        'videos_recommended': len(recommendations_df)
    }
    
    return {
        'summary': summary,
        'skill_analysis': analysis,
        'weak_skills': weak_skills,
        'learning_skills': learning_skills,
        'mastered_skills': mastered_skills,
        'video_recommendations': recommendations_df
    }


# ============== Main App ==============

def main():
    # Header
    st.markdown('<p class="main-header">📚 Study Resource Recommender</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Get personalized YouTube video recommendations based on your quiz performance</p>', unsafe_allow_html=True)
    
    # Load data
    package = load_recommender()
    student_df = load_student_data()
    
    if package is None or student_df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["🏠 Home", "📊 Get Recommendations", "📈 Analytics", "ℹ️ About"]
    )
    
    # ============== Home Page ==============
    if page == "🏠 Home":
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🎓 Students</h3>
                <h2>{:,}</h2>
            </div>
            """.format(student_df['user_id'].nunique()), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📖 Skills</h3>
                <h2>{:,}</h2>
            </div>
            """.format(student_df['skill_name'].nunique()), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🎬 Videos</h3>
                <h2>{:,}</h2>
            </div>
            """.format(len(package['skill_video_mapping'])), unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        ### How It Works
        
        1. **Select a Student** - Choose a student ID from the dropdown
        2. **Analyze Performance** - Our ML model analyzes quiz performance across all skills
        3. **Get Recommendations** - Receive personalized YouTube video recommendations for weak areas
        
        ### Features
        
        - 🤖 **5 ML Models** trained to predict student mastery levels
        - 📊 **Skill Analysis** showing mastered, learning, and struggling areas
        - 🎬 **Video Recommendations** matched to your weak skills
        - 📈 **Visual Analytics** to understand your progress
        """)
    
    # ============== Recommendations Page ==============
    elif page == "📊 Get Recommendations":
        st.markdown("---")
        st.subheader("🎯 Get Personalized Recommendations")
        
        # Student selection
        student_ids = sorted(student_df['user_id'].unique())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_student = st.selectbox(
                "Select Student ID:",
                options=student_ids,
                index=0
            )
        
        with col2:
            videos_per_skill = st.slider("Videos per skill:", 1, 5, 3)
        
        if st.button("🔍 Analyze & Recommend", type="primary", use_container_width=True):
            with st.spinner("Analyzing student performance..."):
                # Get student data
                student_data = student_df[student_df['user_id'] == selected_student].copy()
                student_data = add_engineered_features(student_data)
                
                # Get recommendations
                recommendations = get_recommendations(
                    student_data, 
                    package, 
                    top_skills=5, 
                    videos_per_skill=videos_per_skill
                )
                
                # Display summary
                st.markdown("---")
                st.subheader("📊 Analysis Summary")
                
                summary = recommendations['summary']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Skills", summary['total_skills_analyzed'])
                with col2:
                    st.metric("✅ Mastered", summary['skills_mastered'])
                with col3:
                    st.metric("📚 Learning", summary['skills_learning'])
                with col4:
                    st.metric("❌ Needs Help", summary['skills_need_help'])
                
                # Skill breakdown visualization
                st.markdown("---")
                st.subheader("📈 Skill Mastery Breakdown")
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Mastered', 'Learning', 'Needs Help'],
                    values=[summary['skills_mastered'], summary['skills_learning'], summary['skills_need_help']],
                    hole=0.4,
                    marker_colors=['#28a745', '#ffc107', '#dc3545']
                )])
                fig.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
                
                # Skills that need help
                st.markdown("---")
                st.subheader("❌ Skills That Need Improvement")
                
                weak_skills = recommendations['weak_skills']
                if len(weak_skills) > 0:
                    for _, row in weak_skills.iterrows():
                        st.markdown(f"""
                        <div class="skill-needs-help">
                            <strong>📌 {row['skill_name'].title()}</strong><br>
                            Accuracy: {row['accuracy']:.1%} | Confidence: {row['confidence']:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("🎉 Great job! No skills need immediate attention.")
                
                # Video recommendations
                st.markdown("---")
                st.subheader("🎬 Recommended Videos")
                
                videos = recommendations['video_recommendations']
                if len(videos) > 0:
                    current_skill = None
                    for _, row in videos.iterrows():
                        if row['skill_name'] != current_skill:
                            current_skill = row['skill_name']
                            priority_class = "priority-high" if row['priority'] == 'HIGH' else "priority-medium"
                            st.markdown(f"### 📌 {current_skill.title()} <span class='{priority_class}'>({row['priority']} Priority)</span>", unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="video-card">
                            <strong>🎥 {row['video_title']}</strong><br>
                            <small>Views: {row['views']:,} | Likes: {row['likes']:,}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No video recommendations available for the identified skills.")
                
                # Mastered skills (collapsible)
                with st.expander("✅ View Mastered Skills"):
                    mastered = recommendations['mastered_skills']
                    if len(mastered) > 0:
                        for _, row in mastered.iterrows():
                            st.markdown(f"""
                            <div class="skill-mastered">
                                <strong>✅ {row['skill_name'].title()}</strong> - Accuracy: {row['accuracy']:.1%}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No skills mastered yet. Keep practicing!")
    
    # ============== Analytics Page ==============
    elif page == "📈 Analytics":
        st.markdown("---")
        st.subheader("📈 Overall Analytics")
        
        # Accuracy distribution
        fig1 = px.histogram(
            student_df, 
            x='accuracy', 
            nbins=30,
            title='Distribution of Student Accuracy',
            labels={'accuracy': 'Accuracy', 'count': 'Frequency'},
            color_discrete_sequence=['#1E88E5']
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Top difficult skills
        skill_difficulty = student_df.groupby('skill_name')['accuracy'].mean().sort_values().head(10)
        
        fig2 = px.bar(
            x=skill_difficulty.values,
            y=skill_difficulty.index,
            orientation='h',
            title='Top 10 Most Challenging Skills',
            labels={'x': 'Average Accuracy', 'y': 'Skill'},
            color_discrete_sequence=['#dc3545']
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Correlation between hints and accuracy
        fig3 = px.scatter(
            student_df.sample(min(1000, len(student_df))),
            x='avg_hint_ratio',
            y='accuracy',
            title='Hint Usage vs Accuracy',
            labels={'avg_hint_ratio': 'Hint Ratio', 'accuracy': 'Accuracy'},
            opacity=0.5,
            color_discrete_sequence=['#1E88E5']
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # ============== About Page ==============
    elif page == "ℹ️ About":
        st.markdown("---")
        st.subheader("ℹ️ About This Project")
        
        st.markdown("""
        ### Study Resource Recommender
        
        This application uses Machine Learning to analyze student quiz performance and recommend 
        relevant YouTube educational videos for improvement.
        
        ---
        
        ### 🔧 Technical Details
        
        **Datasets Used:**
        - **ASSISTments 2009-2010**: Student quiz performance data with 346K+ interactions
        - **Khan Academy YouTube**: Educational video database
        
        **ML Models Trained:**
        1. Random Forest Classifier
        2. XGBoost Classifier
        3. Logistic Regression
        4. K-Nearest Neighbors (KNN)
        5. Neural Network (MLP)
        
        **Features Used:**
        - Accuracy, Total Attempts, Hint Usage
        - Response Time, Efficiency Score
        - Struggle Score, Speed Score
        
        ---
        
        ### 📊 Project Phases
        
        1. **Phase 1**: Data Collection & Cleaning
        2. **Phase 2**: Feature Engineering & ML Models
        3. **Phase 3**: Recommendation Engine
        4. **Phase 4**: GUI & Deployment (This App!)
        
        ---
        
        ### 👨‍💻 Developer
        Machine Learning project 
                    

                    Eman-Omar-Yehia-Abdelmawla
        
        Nile University
        
        ---
        
        ### 📚 References
        
        - ASSISTments Dataset: https://sites.google.com/site/assistmentsdata/
        - Khan Academy: https://www.khanacademy.org/
        """)


if __name__ == "__main__":
    main()
