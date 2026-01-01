# 📚 Study Resource Recommender

A Machine Learning-powered web application that recommends YouTube educational videos based on student quiz performance.

## 🎯 Project Overview

**Input:** Student quiz results  
**Output:** Personalized YouTube video recommendations for weak topics

## 🚀 Features

- 🤖 **5 ML Models** trained to predict student mastery levels
- 📊 **Skill Analysis** showing mastered, learning, and struggling areas
- 🎬 **Video Recommendations** matched to weak skills
- 📈 **Visual Analytics** dashboard
- 🌐 **Web Interface** built with Streamlit

## 📁 Project Structure

```
study-resource-recommender/
├── app.py                      # Streamlit web application
├── recommender.py              # Recommendation engine module
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/
│   ├── student_data_for_app.csv    # Student performance data
│   └── skill_video_mapping.csv     # Skill to video mappings
│
├── models/
│   ├── recommender_package.pkl     # All models bundled
│   ├── best_model.pkl              # Best performing model
│   ├── scaler.pkl                  # Feature scaler
│   ├── label_encoder.pkl           # Label encoder
│   └── feature_config.pkl          # Feature configuration
│
└── notebooks/
    ├── Phase1_Data_Collection.ipynb
    ├── Phase2_ML_Models.ipynb
    └── Phase3_Recommendation_Engine.ipynb
```

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/study-resource-recommender.git
cd study-resource-recommender
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```

## 📊 Datasets Used

### Dataset 1: ASSISTments 2009-2010
- **Source:** [ASSISTments Data](https://sites.google.com/site/assistmentsdata/)
- **Size:** 346,860 interactions, 4,217 students, 123 skills
- **Features:** Student ID, Problem ID, Skill Name, Correctness, Hints, Response Time

### Dataset 2: Khan Academy YouTube Videos
- **Source:** [Kaggle](https://www.kaggle.com/datasets/zq1200/khan-academy-youtube-channel)
- **Content:** Educational videos with titles, views, likes, duration

### Connection
Both datasets are connected through **SKILL/TOPIC** matching:
```
ASSISTments skill: "Adding Fractions" → Khan Academy video: "Adding Fractions Tutorial"
```

## 🤖 ML Models

| Model | Type | Purpose |
|-------|------|---------|
| Random Forest | Ensemble | Predict mastery level |
| XGBoost | Gradient Boosting | Predict mastery level |
| Logistic Regression | Linear | Baseline model |
| KNN | Instance-based | Find similar students |
| Neural Network (MLP) | Deep Learning | Complex patterns |

## 📈 Features Used

| Feature | Description |
|---------|-------------|
| accuracy | % correct answers |
| total_attempts | Total problems attempted |
| avg_hint_ratio | Average hints used / available |
| avg_response_time | Average time per problem |
| efficiency_score | Correct answers / hints used |
| struggle_score | Combined difficulty indicator |

## 🎯 Target Variable

```
mastery_level:
  - needs_help: Struggling (accuracy < 60%)
  - learning: Progressing (60% ≤ accuracy < 80%)
  - mastered: Proficient (accuracy ≥ 80%)
```

## 🌐 Deployment

### Option 1: Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy!

### Option 2: HuggingFace Spaces

1. Create a new Space on [HuggingFace](https://huggingface.co/spaces)
2. Select "Streamlit" as the SDK
3. Upload your files
4. Deploy!

### Option 3: Local

```bash
streamlit run app.py
```

## 🚀 Live Demo

The model is deployed using **Streamlit** and can be accessed here:

🔗 https://ml-nu-study-resource-recommender.streamlit.app/

## 👨‍💻 Author

- Abdelmawla
- Eman
- Omar
- Yehia

Nile University, MAchine-intel project

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- ASSISTments for the student performance dataset
- Khan Academy for educational content
- Streamlit for the web framework
