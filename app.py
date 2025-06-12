# app.py - Serveur Flask pour Whisper avec timestamps (optimisé pour Railway)
from flask import Flask, request, jsonify
import requests
import base64
import tempfile
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration depuis les variables d'environnement
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Variable Railway
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB limite OpenAI

@app.route('/whisper', methods=['POST'])
def transcribe_audio():
    try:
        # Récupérer les données de n8n
        data = request.get_json()
        
        if not data or 'audioData' not in data:
            return jsonify({"error": "No audio data provided"}), 400
            
        # Décoder les données base64
        audio_data = base64.b64decode(data['audioData'])
        file_name = data.get('fileName', 'audio.mp3')
        
        # Vérifier la taille du fichier
        if len(audio_data) > MAX_FILE_SIZE:
            return jsonify({"error": "File too large (max 25MB)"}), 400
        
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Préparer la requête pour OpenAI Whisper
            with open(temp_file_path, 'rb') as audio_file:
                files = {
                    'file': (file_name, audio_file, 'audio/mpeg')
                }
                
                data_params = {
                    'model': 'whisper-1',
                    'response_format': 'verbose_json',
                    'timestamp_granularities[]': 'word',
                    'language': 'en'
                }
                
                headers = {
                    'Authorization': f'Bearer {OPENAI_API_KEY}'
                }
                
                # Appeler l'API Whisper
                response = requests.post(
                    'https://api.openai.com/v1/audio/transcriptions',
                    files=files,
                    data=data_params,
                    headers=headers,
                    timeout=300  # 5 minutes timeout
                )
            
            # Nettoyer le fichier temporaire
            os.unlink(temp_file_path)
            
            if response.status_code != 200:
                return jsonify({
                    "error": "OpenAI API error",
                    "details": response.text,
                    "status": response.status_code
                }), response.status_code
            
            # Traiter la réponse
            transcription_data = response.json()
            
            # Extraire les segments avec timestamps
            segments = []
            if 'segments' in transcription_data:
                for segment in transcription_data['segments']:
                    segments.append({
                        'id': segment['id'],
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'].strip(),
                        'duration': segment['end'] - segment['start']
                    })
            
            # Extraire les mots avec timestamps (si disponibles)
            words = []
            if 'words' in transcription_data:
                for word in transcription_data['words']:
                    words.append({
                        'word': word['word'],
                        'start': word['start'],
                        'end': word['end']
                    })
            
            # Retourner les données formatées
            return jsonify({
                "success": True,
                "full_text": transcription_data.get('text', ''),
                "language": transcription_data.get('language', ''),
                "duration": transcription_data.get('duration', 0),
                "segments": segments,
                "words": words,
                "segments_count": len(segments),
                "words_count": len(words)
            })
            
        except Exception as e:
            # Nettoyer en cas d'erreur
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
            
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "whisper-webhook"})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Railway fournit PORT automatiquement
    print("🎙️  Whisper Webhook Server démarré!")
    print(f"📡 Port: {port}")
    print("🔍 Health check: /health")
    app.run(host='0.0.0.0', port=port, debug=False)
