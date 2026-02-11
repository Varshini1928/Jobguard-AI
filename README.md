JobShield AI – Fake Job Post Detection System

Project Overview

JobShield AI is an intelligent, web-based system designed to detect fraudulent job postings using Natural Language Processing (NLP) and Machine Learning (ML).

The system analyzes job descriptions and job posters (via OCR) to classify postings as FAKE or REAL. In addition to classification, it generates a Risk Score, Risk Level, and human-readable explanations using Explainable AI (XAI), ensuring transparency and trust.

This project is designed with a strong focus on real-world applicability, secure system architecture, and user accessibility, making it suitable for recruitment platforms, job portals, and fraud detection systems.

Key Features

Fake Job Detection using Logistic Regression
NLP-based text analysis using TF-IDF Vectorizer
OCR support for job posters using Tesseract OCR
Risk Score calculation (0–10)
Risk Level classification (Low / Medium / High)
Explainable AI with clear prediction reasons
Role-Based Access Control (User / Admin / Super Admin)
User and Admin Dashboards
Prediction history tracking
Secure authentication using Email & Password
Downloadable reports in CSV and PDF formats
Modern and responsive UI using Tailwind CSS
User Roles

User

Submit job descriptions or job posters
View prediction results
Understand risk explanations
Access personal prediction history
Admin

Monitor user activity
Analyze system-wide risk trends
View analytics dashboards
Manage generated reports
Super Admin

Complete system-level control
User and Admin management
Platform monitoring and analytics
Technology Stack

Backend: Python, Flask
Frontend: HTML, Tailwind CSS

Machine Learning

Logistic Regression
TF-IDF Vectorizer
OCR: Tesseract OCR

Database: MySQL

Authentication & Services

GitHub OAuth
Session-based Authentication
Flask-Mail (OTP-based Password Reset)


How to Run the Project

Step 1 – Clone the Repository

git clone https://github.com/Varshini1928/Jobguard-AI

Step 2 – Navigate to the Project Directory

cd Infosys_Project

Step 3 – Install Dependencies

pip install -r requirements.txt

Step 4 – Run the Application

python app.py

Step 5 – Open in Browser

http://127.0.0.1:5000

Sample Output

Result: FAKE JOB
Confidence: 95% 
Risk Score: 9 / 10
Risk Level: High Risk

Explanation

Urgency-based language detected
Direct personal contact information found
Unrealistic income claims
Missing company verification details
Author

Varshini S 
Infosys Internship Project Submission
Year: 2026

Final Note

JobShield AI was developed with a strong emphasis on Explainable AI, system security, and real-world deployment.
This project demonstrates how NLP, Machine Learning, and OCR can be effectively combined to combat online recruitment fraud while maintaining clarity, trust, and usability.                                                                                                                                      License

This project is licensed under the MIT License.
