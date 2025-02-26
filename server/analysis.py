# # # analysis.py

import numpy as np
import cv2
import speech_recognition as sr
import librosa
import soundfile as sf
import time
import subprocess
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tensorflow as tf
import matplotlib.pyplot as plt

# Load pre-trained models for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load the trained confidence model
model = tf.keras.models.load_model('model/confidence_model_Adam.h5')  # Change model path if needed

# Define image dimensions
IMG_HEIGHT = 64
IMG_WIDTH = 64

def extract_audio(video_path, audio_path):
    command = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
    subprocess.run(command, check=True)

def load_audio(audio_path):
    audio, sr_rate = sf.read(audio_path)
    return audio, sr_rate

def transcribe_audio(audio_path, retries=3):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        for attempt in range(retries):
            try:
                text = recognizer.recognize_google(audio_data)
                return text
            except sr.UnknownValueError:
                print("Google Web Speech API could not understand audio")
                return ""
            except sr.RequestError as e:
                print(f"Request error: {e}. Attempt {attempt + 1} of {retries}")
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    return ""

def measure_voice_confidence(text, audio, sr_rate):
    fillers = ["uh", "um", "like", "you know", "so", "actually"]
    filler_count = sum(text.lower().count(filler) for filler in fillers)
    filler_penalty = -filler_count * 0.1

    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr_rate)
    avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches) else 0
    pitch_score = 0.6 if avg_pitch < 120 else 0.8 if avg_pitch < 200 else 1.0

    duration_seconds = librosa.get_duration(y=audio, sr=sr_rate)
    word_count = len(text.split())
    wpm = word_count / (duration_seconds / 60) if duration_seconds > 0 else 0
    speech_rate_score = 0.6 if wpm < 100 else 0.8 if wpm < 160 else 1.0

    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    confidence = (compound_score + 1) / 2

    final_voice_confidence = (confidence + pitch_score + speech_rate_score + filler_penalty) / 4
    final_voice_confidence = max(0, min(final_voice_confidence, 1))
    return final_voice_confidence, wpm, filler_count

def predict_confidence(frame, model):
    frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
    frame = np.expand_dims(frame, axis=0) / 255.0
    prediction = model.predict(frame)
    return prediction[0][0] * 100

def measure_face_confidence(video_path):
    cap = cv2.VideoCapture(video_path)
    confidence_levels = []
    timestamps = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        confidence = predict_confidence(frame, model)
        confidence_levels.append(confidence)
        timestamps.append(time.time() - start_time)

    cap.release()

    overall_confidence = np.mean(confidence_levels) / 100 if confidence_levels else 0
    return overall_confidence

def generate_suggestions(wpm, filler_count, face_confidence, voice_confidence):
    suggestions = []
    if wpm < 100:
        suggestions.append("Speak a bit faster to maintain a confident pace.")
    elif wpm > 160:
        suggestions.append("Slow down your speech slightly for better clarity.")

    if filler_count > 2:
        suggestions.append("Reduce filler words like 'um' and 'uh' for more authoritative speech.")

    if face_confidence < 0.5:
        suggestions.append("Maintain steady eye contact and engage facial expressions for better confidence.")

    if voice_confidence < 0.5:
        suggestions.append("Use a more positive and enthusiastic tone to convey confidence.")

    if not suggestions:
        suggestions.append("Great job! Keep up the confident delivery.")

    return suggestions

def calculate_overall_confidence(face_confidence, voice_confidence):
    return (face_confidence + voice_confidence) / 2

def main():
    video_path = 'video1.mp4'
    audio_path = 'extracted_audio.wav'

    extract_audio(video_path, audio_path)
    audio, sr_rate = load_audio(audio_path)
    transcribed_text = transcribe_audio(audio_path)

    voice_confidence, wpm, filler_count = measure_voice_confidence(transcribed_text, audio, sr_rate)
    face_confidence = measure_face_confidence(video_path)
    overall_confidence = calculate_overall_confidence(face_confidence, voice_confidence)
    suggestions = generate_suggestions(wpm, filler_count, face_confidence, voice_confidence)

    print("Overall Confidence:", round(overall_confidence * 100, 2), "%")
    print("Suggestions:", suggestions)

if __name__ == "__main__":
    main()



# import numpy as np
# import cv2
# import speech_recognition as sr
# import librosa
# import soundfile as sf
# import time
# import subprocess
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# # Load pre-trained models for face detection
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# # Function to extract audio from video using FFmpeg
# def extract_audio(video_path, audio_path):
#     command = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
#     subprocess.run(command, check=True)

# # Function to load audio using soundfile
# def load_audio(audio_path):
#     audio, sr_rate = sf.read(audio_path)
#     return audio, sr_rate

# # Function to transcribe audio to text using Google's speech recognition service
# def transcribe_audio(audio_path, retries=3):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_path) as source:
#         audio_data = recognizer.record(source)
#         for attempt in range(retries):
#             try:
#                 text = recognizer.recognize_google(audio_data)
#                 return text
#             except sr.UnknownValueError:
#                 print("Google Web Speech API could not understand audio")
#                 return ""
#             except sr.RequestError as e:
#                 print(f"Request error: {e}. Attempt {attempt + 1} of {retries}")
#                 if attempt < retries - 1:
#                     time.sleep(2)
#                 else:
#                     return ""

# # Function to measure voice confidence using sentiment analysis
# def measure_voice_confidence(text, audio, sr_rate):
#     fillers = ["uh", "um", "like", "you know", "so", "actually"]
#     filler_count = sum(text.lower().count(filler) for filler in fillers)
#     filler_penalty = -filler_count * 0.1

#     pitches, magnitudes = librosa.piptrack(y=audio, sr=sr_rate)
#     avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches) else 0
#     pitch_score = 0.6 if avg_pitch < 120 else 0.8 if avg_pitch < 200 else 1.0

#     duration_seconds = librosa.get_duration(y=audio, sr=sr_rate)
#     word_count = len(text.split())
#     wpm = word_count / (duration_seconds / 60) if duration_seconds > 0 else 0
#     speech_rate_score = 0.6 if wpm < 100 else 0.8 if wpm < 160 else 1.0

#     analyzer = SentimentIntensityAnalyzer()
#     sentiment_scores = analyzer.polarity_scores(text)
#     compound_score = sentiment_scores['compound']
#     confidence = (compound_score + 1) / 2

#     final_voice_confidence = (confidence + pitch_score + speech_rate_score + filler_penalty) / 4
#     final_voice_confidence = max(0, min(final_voice_confidence, 1))
#     return final_voice_confidence, wpm, filler_count

# # Function to measure face confidence based on facial features
# def measure_face_confidence(video_path):
#     cap = cv2.VideoCapture(video_path)
#     total_confidence = 0
#     num_frames = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         num_frames += 1
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         for (x, y, w, h) in faces:
#             face_roi = gray[y:y+h, x:x+w]
#             eyes = eye_cascade.detectMultiScale(face_roi)
#             num_eyes = len(eyes)
#             confidence = min(num_eyes / 2, 1.0)
#             total_confidence += confidence

#     cap.release()
#     overall_confidence = total_confidence / num_frames if num_frames > 0 else 0
#     return overall_confidence

# # Generate suggestions based on analysis
# def generate_suggestions(wpm, filler_count, face_confidence, voice_confidence):
#     suggestions = []

#     if wpm < 100:
#         suggestions.append("Speak a bit faster to maintain a confident pace.")
#     elif wpm > 160:
#         suggestions.append("Slow down your speech slightly for better clarity.")

#     if filler_count > 2:
#         suggestions.append("Reduce filler words like 'um' and 'uh' for more authoritative speech.")

#     if face_confidence < 0.5:
#         suggestions.append("Maintain steady eye contact and engage facial expressions for better confidence.")

#     if voice_confidence < 0.5:
#         suggestions.append("Use a more positive and enthusiastic tone to convey confidence.")

#     if not suggestions:
#         suggestions.append("Great job! Keep up the confident delivery.")

#     return suggestions

# # Ensemble method: Taking the average of confidence levels
# def calculate_overall_confidence(face_confidence, voice_confidence):
#     return (face_confidence + voice_confidence) / 2

# # Example usage
# def main():
#     video_path = 'video1.mp4'
#     audio_path = 'extracted_audio.wav'

#     extract_audio(video_path, audio_path)
#     audio, sr_rate = load_audio(audio_path)
#     transcribed_text = transcribe_audio(audio_path)

#     voice_confidence, wpm, filler_count = measure_voice_confidence(transcribed_text, audio, sr_rate)
#     face_confidence = measure_face_confidence(video_path)
#     overall_confidence = calculate_overall_confidence(face_confidence, voice_confidence)
#     suggestions = generate_suggestions(wpm, filler_count, face_confidence, voice_confidence)

#     print("Overall Confidence:", round(overall_confidence * 100, 2), "%")
#     print("Suggestions:", suggestions)

# if __name__ == "__main__":
#     main()







# import cv2
# import numpy as np
# import moviepy.editor as mp
# import speech_recognition as sr
# import librosa
# import time
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# # Load pre-trained models for face detection
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# # Function to extract audio from video
# def extract_audio(video_path, audio_path):
#     video = mp.VideoFileClip(video_path)
#     audio = video.audio
#     audio.write_audiofile(audio_path, codec='pcm_s16le')  # Exports in WAV format
#     audio.close()
#     video.close()

# # Function to transcribe audio to text using Google's speech recognition service
# def transcribe_audio(audio_path, retries=3):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_path) as source:
#         audio_data = recognizer.record(source)
#         for attempt in range(retries):
#             try:
#                 text = recognizer.recognize_google(audio_data)
#                 return text
#             except sr.UnknownValueError:
#                 print("Google Web Speech API could not understand audio")
#                 return ""
#             except sr.RequestError as e:
#                 print(f"Request error: {e}. Attempt {attempt + 1} of {retries}")
#                 if attempt < retries - 1:
#                     time.sleep(2)
#                 else:
#                     return ""

# # Function to measure voice confidence using sentiment analysis
# def measure_voice_confidence(text, audio, sr_rate):
#     fillers = ["uh", "um", "like", "you know", "so", "actually"]
#     filler_count = sum(text.lower().count(filler) for filler in fillers)
#     filler_penalty = -filler_count * 0.1

#     pitches, magnitudes = librosa.piptrack(y=audio, sr=sr_rate)
#     avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches) else 0
#     pitch_score = 0.6 if avg_pitch < 120 else 0.8 if avg_pitch < 200 else 1.0

#     duration_seconds = librosa.get_duration(y=audio, sr=sr_rate)
#     word_count = len(text.split())
#     wpm = word_count / (duration_seconds / 60)
#     speech_rate_score = 0.6 if wpm < 100 else 0.8 if wpm < 160 else 1.0

#     analyzer = SentimentIntensityAnalyzer()
#     sentiment_scores = analyzer.polarity_scores(text)
#     compound_score = sentiment_scores['compound']
#     confidence = (compound_score + 1) / 2

#     final_voice_confidence = (confidence + pitch_score + speech_rate_score + filler_penalty) / 4
#     final_voice_confidence = max(0, min(final_voice_confidence, 1))
#     return final_voice_confidence, wpm, filler_count

# # Function to measure face confidence based on facial features
# def measure_face_confidence(video_path):
#     cap = cv2.VideoCapture(video_path)
#     total_confidence = 0
#     num_frames = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         num_frames += 1

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         for (x, y, w, h) in faces:
#             face_roi = gray[y:y+h, x:x+w]
#             eyes = eye_cascade.detectMultiScale(face_roi)
#             num_eyes = len(eyes)
#             confidence = min(num_eyes / 2, 1.0)
#             total_confidence += confidence

#     cap.release()
#     overall_confidence = total_confidence / num_frames if num_frames > 0 else 0
#     return overall_confidence

# # Generate suggestions based on analysis
# def generate_suggestions(wpm, filler_count, face_confidence, voice_confidence):
#     suggestions = []

#     # WPM suggestions
#     if wpm < 100:
#         suggestions.append("Speak a bit faster to maintain a confident pace.")
#     elif wpm > 160:
#         suggestions.append("Slow down your speech slightly for better clarity.")

#     # Filler word suggestions
#     if filler_count > 2:
#         suggestions.append("Reduce filler words like 'um' and 'uh' for more authoritative speech.")

#     # Facial confidence suggestions
#     if face_confidence < 0.5:
#         suggestions.append("Maintain steady eye contact and engage facial expressions for better confidence.")

#     # Voice confidence suggestions
#     if voice_confidence < 0.5:
#         suggestions.append("Use a more positive and enthusiastic tone to convey confidence.")

#     if not suggestions:
#         suggestions.append("Great job! Keep up the confident delivery.")

#     return suggestions

# # Ensemble method: Taking the average of confidence levels
# def calculate_overall_confidence(face_confidence, voice_confidence):
#     return (face_confidence + voice_confidence) / 2

# # Example usage
# def main():
#     video_path = 'video1.mp4'
#     audio_path = 'extracted_audio.wav'

#     extract_audio(video_path, audio_path)
#     audio, sr_rate = librosa.load(audio_path)
#     transcribed_text = transcribe_audio(audio_path)

#     voice_confidence, wpm, filler_count = measure_voice_confidence(transcribed_text, audio, sr_rate)
#     face_confidence = measure_face_confidence(video_path)
#     overall_confidence = calculate_overall_confidence(face_confidence, voice_confidence)
#     suggestions = generate_suggestions(wpm, filler_count, face_confidence, voice_confidence)

#     print("Overall Confidence:", round(overall_confidence * 100, 2), "%")
#     print("Suggestions:", suggestions)

# if __name__ == "__main__":
#     main()
