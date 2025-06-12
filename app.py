from flask import Flask, request, jsonify
import requests
import tempfile
import os

app = Flask(__name__)

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Route d'accueil (pour éviter l'erreur "Not Found")
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Service de timestamps audio actif",
        "service": "whisper-webhook",
        "endpoints": {
            "process": "POST /process-audio",
            "health": "GET /health"
        }
    })

# Route principale pour n8n
@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        print("🎵 Traitement audio démarré...")
        
        # Vérifier qu'un fichier a été envoyé
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'Aucun fichier audio fourni'}), 400
        
        audio_file = request.files['audio']
        segment_duration = float(request.form.get('segment_duration', 4))
        
        print(f"📁 Fichier reçu: {audio_file.filename}")
        print(f"⏱️ Durée de segment: {segment_duration}s")
        
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            audio_file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Appeler Whisper
            print("🤖 Appel à Whisper...")
            transcription = call_whisper(temp_path)
            
            # Créer les segments
            print("✂️ Création des segments...")
            segments = create_segments(transcription, segment_duration)
            
            print(f"✅ Traitement terminé: {len(segments)} segments créés")
            
            return jsonify({
                'success': True,
                'total_duration': transcription.get('duration', 0),
                'segments': segments
            })
            
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def call_whisper(audio_path):
    """Appeler l'API Whisper"""
    with open(audio_path, 'rb') as audio_file:
        files = {'file': audio_file}
        data = {
            'model': 'whisper-1',
            'response_format': 'verbose_json',
            'timestamp_granularities[]': 'word'
        }
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}'
        }
        
        response = requests.post(
            'https://api.openai.com/v1/audio/transcriptions',
            files=files,
            data=data,
            headers=headers,
            timeout=300
        )
        
        if response.status_code != 200:
            raise Exception(f'Erreur Whisper ({response.status_code}): {response.text}')
            
        return response.json()

def create_segments(transcription, segment_duration):
    """Créer des segments de durée fixe"""
    segments = []
    total_duration = transcription.get('duration', 0)
    total_segments = int(total_duration / segment_duration) + 1
    
    print(f"📊 Durée totale: {total_duration}s, {total_segments} segments")
    
    for i in range(total_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, total_duration)
        
        # Trouver les mots dans ce segment
        words_in_segment = []
        if 'words' in transcription:
            words_in_segment = [
                word for word in transcription['words']
                if word['start'] >= start_time and word['start'] < end_time
            ]
        
        segments.append({
            'segment_id': i,
            'start_time': round(start_time, 2),
            'end_time': round(end_time, 2),
            'duration': round(end_time - start_time, 2),
            'has_lyrics': len(words_in_segment) > 0,
            'lyrics': ' '.join([w['word'] for w in words_in_segment]).strip(),
            'words': words_in_segment
        })
    
    return segments

# Route de santé
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "service": "whisper-webhook",
        "openai_configured": bool(OPENAI_API_KEY)
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print("🎙️ Whisper Webhook Server démarré!")
    print(f"📡 Port: {port}")
    print("🔍 Endpoints:")
    print("  GET  / - Page d'accueil")
    print("  POST /process-audio - Traitement audio")
    print("  GET  /health - Status")
    app.run(host='0.0.0.0', port=port, debug=False)
