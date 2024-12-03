from flask import Flask, render_template, request, jsonify, send_file
import json
from event_recommender import EventRecommender
import os
from dotenv import load_dotenv
from gtts import gTTS
import uuid
from pathlib import Path

# Create a directory for temporary audio files
AUDIO_DIR = Path("temp_audio")
AUDIO_DIR.mkdir(exist_ok=True)

# Load environment variables
load_dotenv()

# Get API key and verify it exists
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("No OpenAI API key found in .env file")

print("Starting Flask application...")

app = Flask(__name__)

print("Initializing EventRecommender...")
# Initialize the recommender
recommender = EventRecommender(openai_api_key=api_key)

print("Loading events data...")
# Load events data
try:
    with open("mdc_events.json", "r") as f:
        events_data = json.load(f)
    recommender.load_events(events_data)
    print(f"Loaded {len(events_data)} events successfully")
except FileNotFoundError:
    print("Error: mdc_events.json file not found!")
    events_data = []
except json.JSONDecodeError:
    print("Error: Invalid JSON in mdc_events.json!")
    events_data = []

# @app.route('/')
# def home():
#     print("Received request for home page")
#     return render_template('index.html')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    preferences = request.form.get('preferences', '')
    print(f"Received recommendation request with preferences: {preferences}")
    
    if not preferences:
        return jsonify({'error': 'No preferences provided'}), 400
    
    try:
        response = recommender.generate_recommendation_response(preferences)
        print("Generated recommendations successfully")
        
        # Generate unique filename for this audio
        audio_filename = f"speech_{uuid.uuid4()}.mp3"
        audio_path = AUDIO_DIR / audio_filename
        
        # Create speech from text
        tts = gTTS(text=response, lang='en')
        tts.save(str(audio_path))
        
        return jsonify({
            'recommendations': response,
            'audio_file': audio_filename
        })
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    try:
        return send_file(
            AUDIO_DIR / filename,
            mimetype='audio/mpeg',
            as_attachment=False
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404

# Cleanup old audio files
@app.after_request
def cleanup_audio(response):
    # Delete files older than 5 minutes
    import time
    current_time = time.time()
    for audio_file in AUDIO_DIR.glob("*.mp3"):
        if current_time - audio_file.stat().st_mtime > 300:  # 5 minutes
            try:
                audio_file.unlink()
            except Exception as e:
                print(f"Error deleting old audio file: {e}")
    return response

if __name__ == '__main__':
    print("Starting Flask development server...")
    app.run(debug=True, port=8000)