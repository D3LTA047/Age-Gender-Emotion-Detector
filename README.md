# Age-Gender-Emotion-Detector

This project is a real-time application that detects a person’s age, gender, and emotion using deep learning models with OpenCV. It captures video input from a webcam, detects faces, and then predicts the age group, gender, and emotional state of each detected face. This application could be useful in a variety of contexts, such as sentiment analysis, customer feedback, or demographic data collection.

## Features
Real-Time Face Detection: Uses a deep learning model to detect faces within live video frames.
Age Prediction: Predicts the age group of the detected faces from pre-defined categories.
Gender Detection: Classifies detected faces as Male or Female.
Emotion Recognition: Detects the current emotional state of the detected faces, such as Happy, Sad, Angry, etc.
On-Screen Labels: Displays bounding boxes around detected faces and annotates them with the predicted age, gender, and emotion.
Technologies Used
OpenCV: For capturing video from the webcam and displaying results.
Deep Learning Models: Pre-trained models for face detection, age prediction, gender classification, and emotion recognition.
Python: The primary programming language for the application.
How It Works
Face Detection: The application processes each video frame to detect faces using a pre-trained face detection model.
Feature Extraction: For each detected face, the application crops the face region and passes it to age, gender, and emotion models.
Prediction and Display: The models return predictions for age, gender, and emotion, which are displayed on the video feed in real-time.
Prerequisites
Python 3.x
OpenCV
Pre-trained models for age, gender, and emotion detection (available in .prototxt and .caffemodel formats)
Installation
Clone this repository:
bash
Copy code
git clone https://github.com/YourUsername/Age-Gender-Emotion-Detection.git
Install the required packages:
bash
Copy code
pip install opencv-python
Download the pre-trained models and place them in the project directory.
Usage
Run the application with the following command:

bash
Copy code
python age_gender_emotion_detection.py
Press q to quit the application.

Future Enhancements
Expand the emotion recognition model to support more emotions.
Optimize the model for better real-time performance.
Add support for multiple cameras or video sources.
