# 🚀 Deployment Instructions

## Files Needed for Deployment

Make sure you have these files in one folder:

```
📁 your-app-folder/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Dependencies
├── README.md                       # Project description
├── recommender_package.pkl         # ML models + data (from Phase 3)
└── student_data_for_app.csv        # Student data (from Phase 3)
```

---

## Option 1: Streamlit Cloud (FREE & Easiest) ⭐

### Step 1: Create GitHub Repository

1. Go to [github.com](https://github.com) and sign in
2. Click "New repository"
3. Name it: `study-resource-recommender`
4. Make it **Public**
5. Click "Create repository"

### Step 2: Upload Files to GitHub

**Option A: Using GitHub Web Interface**
1. Click "uploading an existing file"
2. Drag and drop all your files
3. Click "Commit changes"

**Option B: Using Git Command Line**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/study-resource-recommender.git
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Sign in with GitHub"
3. Click "New app"
4. Select your repository: `study-resource-recommender`
5. Branch: `main`
6. Main file path: `app.py`
7. Click "Deploy!"

### Step 4: Wait & Share

- Deployment takes 2-5 minutes
- You'll get a URL like: `https://your-app-name.streamlit.app`
- Share this URL with your TA!

---

## Option 2: HuggingFace Spaces (FREE)

### Step 1: Create HuggingFace Account

1. Go to [huggingface.co](https://huggingface.co)
2. Sign up for free

### Step 2: Create New Space

1. Click your profile → "New Space"
2. Name: `study-resource-recommender`
3. Select SDK: **Streamlit**
4. Select visibility: **Public**
5. Click "Create Space"

### Step 3: Upload Files

1. Click "Files" tab
2. Click "Add file" → "Upload files"
3. Upload all your files:
   - `app.py`
   - `requirements.txt`
   - `recommender_package.pkl`
   - `student_data_for_app.csv`
4. Commit changes

### Step 4: Wait & Share

- Auto-builds in 2-5 minutes
- URL: `https://huggingface.co/spaces/YOUR_USERNAME/study-resource-recommender`

---

## Option 3: Run Locally

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the App

```bash
streamlit run app.py
```

### Step 3: Open Browser

- App opens automatically at: `http://localhost:8501`
- Or manually open this URL

---

## Troubleshooting

### Error: "Module not found"
```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly
```

### Error: "File not found"
- Make sure `recommender_package.pkl` and `student_data_for_app.csv` are in the same folder as `app.py`

### Error: "Memory limit exceeded" (on Streamlit Cloud)
- Your `recommender_package.pkl` might be too large
- Try reducing the data or using a smaller model

### App is slow
- First load takes longer (loading models)
- Subsequent uses are faster (cached)

---

## Demo Video (Optional)

To record a demo:
1. Use screen recording software (OBS, Loom, etc.)
2. Show:
   - Home page
   - Select a student
   - View recommendations
   - Analytics page
3. Keep it under 2 minutes

---

## Checklist Before Submission

- [ ] App runs locally without errors
- [ ] All files uploaded to GitHub/HuggingFace
- [ ] Deployment successful
- [ ] Got the public URL
- [ ] Tested the live app
- [ ] (Optional) Recorded demo video

---

## Need Help?

If deployment fails:
1. Check the error logs on Streamlit Cloud/HuggingFace
2. Make sure all files are uploaded
3. Verify `requirements.txt` has all dependencies
4. Try redeploying

Good luck! 🎉
