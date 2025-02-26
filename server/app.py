# app.py
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from analysis import *

app = Flask(__name__, template_folder='template')
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform analysis
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_audio.wav')
        extract_audio(video_path, audio_path)
        audio, sr_rate = librosa.load(audio_path)
        transcribed_text = transcribe_audio(audio_path)

        voice_confidence, wpm, filler_count = measure_voice_confidence(transcribed_text, audio, sr_rate)
        face_confidence = measure_face_confidence(video_path)
        overall_confidence = round(calculate_overall_confidence(face_confidence, voice_confidence) * 100, 2)
        suggestions = generate_suggestions(wpm, filler_count, face_confidence, voice_confidence)

        return jsonify({
            'confidence': f"{overall_confidence}%",
            'face_confidence': f"{round(face_confidence * 100, 2)}%",
            'voice_confidence': f"{round(voice_confidence * 100, 2)}%",
            'transcribed_text': transcribed_text,
            'suggestions': suggestions
        })
    else:
        return jsonify({'error': 'File type not allowed'})

if __name__ == '__main__':
    app.run(debug=True)
