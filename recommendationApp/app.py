from flask import Flask, render_template, request, jsonify, send_file
import json
from event_recommender import EventRecommender
import os
from dotenv import load_dotenv
from gtts import gTTS
import uuid
from pathlib import Path
import qrcode
from io import BytesIO
from urllib.parse import urlparse
import whisper

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

# Load Whisper model
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    preferences = request.form.get('preferences', '')
    if not preferences:
        return jsonify({'error': 'No preferences provided'}), 400
    
    try:
        # Get event recommendations
        recommendations = recommender.get_recommendations(preferences)
        events = [
            {
                'title': rec['event'].get('Event Title', 'Untitled Event'),
                'date': rec['event'].get('Event Date', 'Date not specified'),
                'url': rec['event'].get('URL')
            }
            for rec in recommendations if 'URL' in rec['event']
        ]
        
        # Generate dynamic text response
        response_text = "Here are some events that you may find interesting"
        # Create an audio file for the response
        audio_filename = f"{uuid.uuid4()}.mp3"
        tts = gTTS(text=response_text, lang='en')
        audio_path = AUDIO_DIR / audio_filename
        tts.save(audio_path)
        
        # Return recommendations and audio file URL
        return jsonify({
            'recommendations': response_text,
            'events': events,
            'audio_url': f"/audio/{audio_filename}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({'error': 'No audio file uploaded.'}), 400

    try:
        # Save audio temporarily
        temp_path = AUDIO_DIR / f"{uuid.uuid4()}.wav"
        audio_file.save(temp_path)

        # Transcribe audio using Whisper
        transcription = whisper_model.transcribe(str(temp_path))["text"]

        # Clean up temporary file
        temp_path.unlink()

        return jsonify({'transcription': transcription})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_qr_code', methods=['POST'])
def get_qr_code():
    url = request.form.get('url')
    event_title = request.form.get('title')
    event_date = request.form.get('date')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    # Validate URL
    def is_valid_url(url):
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)
    
    if not is_valid_url(url):
        return jsonify({'error': 'Invalid URL provided'}), 400

    try:
        # Combine event details into QR code data
        qr_data = f"{event_title}\n{event_date}\n{url}"
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")

        # Save the QR code to a BytesIO object
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        # Return the image as a response
        return send_file(buffer, mimetype='image/png')
    except Exception as e:
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
