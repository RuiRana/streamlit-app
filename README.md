# 🚀 Student Completion Risk Predictor

## 📌 Overview
This project is a machine learning application designed to identify students at risk of not completing their course. It analyses student data such as attendance, engagement, and performance to predict completion likelihood and provide actionable insights.

The model is deployed as an interactive web application using Streamlit, allowing users to upload data and instantly view predictions and analytics.

---

## 🎯 Objective
To demonstrate how machine learning can be applied to real-world education data in order to:
- Identify at-risk students early  
- Support intervention planning  
- Improve overall student outcomes  

---

## 🧠 Features
- Upload student dataset (CSV format)  
- Predict completion probability using a trained ML model  
- Identify at-risk students  
- Display key metrics:
  - Total students  
  - Completion rate  
  - At-risk count  
  - Model accuracy  
- Visualisations:
  - Attendance vs completion  
  - Feature importance (what drives outcomes most)  
- Ranked list of students by risk level  

---

## 🛠️ Technologies Used
- **Python**
- **pandas** – data processing  
- **scikit-learn** – machine learning model (Random Forest)  
- **matplotlib** – data visualisation  
- **Streamlit** – web app interface  

---

## 📊 How It Works
1. User uploads a dataset containing student information  
2. Data is processed and cleaned  
3. A machine learning model predicts completion probability  
4. Results are displayed in a dashboard with insights and visualisations  

---

## 📁 Project Structure

├── app.py # Main Streamlit application
├── requirements.txt # Dependencies
└── students.csv # Sample dataset


---

## ▶️ Running the App

### Option 1 – Local

pip install -r requirements.txt
streamlit run app.py


### Option 2 – Live App
👉 Access the deployed app here:  
**[[[Insert your Streamlit link here](https://app-app-3qljs2ncjtpzmssxozrgo2.streamlit.app/)]**](https://app-app-3qljs2ncjtpzmssxozrgo2.streamlit.app/)

---

## 📈 Example Use Case
This tool can be used by:
- Colleges and training providers  
- Education managers  
- Data teams  

To:
- Monitor student performance  
- Identify intervention needs  
- Support data-driven decision making  

---

## 🔍 Key Insight
The model highlights which factors most influence student completion, helping organisations focus on the areas that matter most (e.g. attendance, engagement).

---

## 🚀 Future Improvements
- Add real-time data integration  
- Improve model accuracy with larger datasets  
- Introduce advanced forecasting  
- Add intervention recommendations  

---

## 👤 Author
Developed as part of an AI and data learning journey, focusing on applying machine learning to real-world problems and deploying usable tools.
