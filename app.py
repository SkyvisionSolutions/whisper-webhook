from flask import Flask, request, jsonify
import requests
import tempfile
import os
import base64

app = Flask(__name__)

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Service de timestamps audio actif",
        "service": "whisper-webhook",
        "endpoints": {
            "process": "POST /process-audio",
            "process_base64": "POST /process-audio-base64",
            "health": "GET /health"
        }
    })

# Route pour fichiers multipart (n8n form-data)
@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        print("🎵 Traitement audio (multipart)...")
        
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'Aucun fichier audio fourni'}), 400
        
        audio_file = request.files['audio']
        segment_duration = float(request.form.get('segment_duration', 4))
        
        print(f"📁 Fichier: {audio_file.filename}")
        
        # Traiter le fichier
        result = process_audio_file(audio_file, segment_duration)
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Erreur multipart: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Route alternative pour données base64 (si form-data ne fonctionne pas)
@app.route('/process-audio-base64', methods=['POST'])
def process_audio_base64():
    try:
        print("🎵 Traitement audio (base64)...")
        
        data = request.get_json()
        if not data or 'audio_data' not in data:
            return jsonify({'success': False, 'error': 'Données audio manquantes'}), 400
        
        # Décoder base64
        audio_data = base64.b64decode(data['audio_data'])
        segment_duration = float(data.get('segment_duration', 4))
        filename = data.get('filename', 'audio.mp3')
        
        print(f"📁 Fichier: {filename}")
        
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        try:
            # Traiter
            transcription = call_whisper(temp_path)
            segments = create_segments(transcription, segment_duration)
            
            return jsonify({
                'success': True,
                'total_duration': transcription.get('duration', 0),
                'segments': segments
            })
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        print(f"❌ Erreur base64: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def process_audio_file(audio_file, segment_duration):
    """Traiter un fichier audio"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        audio_file.save(temp_file.name)
        temp_path = temp_file.name
    
    try:
        print("🤖 Appel à Whisper...")
        transcription = call_whisper(temp_path)
        
        print("✂️ Création des segments...")
        segments = create_segments(transcription, segment_duration)
        
        print(f"✅ Traitement terminé: {len(segments)} segments")
        
        return {
            'success': True,
            'total_duration': transcription.get('duration', 0),
            'segments': segments
        }
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def call_whisper(audio_path):
    """Appeler l'API Whisper"""
    try:
        with open(audio_path, 'rb') as audio_file:
            files = {'file': audio_file}
            data = {
                'model': 'whisper-1',
                'response_format': 'verbose_json',
                'timestamp_granularities[]': 'word',
                'language': 'fr'  # Français
            }
            headers = {
                'Authorization': f'Bearer {OPENAI_API_KEY}'
            }
            
            print("📡 Envoi à OpenAI...")
            response = requests.post(
                'https://api.openai.com/v1/audio/transcriptions',
                files=files,
                data=data,
                headers=headers,
                timeout=300
            )
            
            if response.status_code != 200:
                raise Exception(f'Erreur Whisper ({response.status_code}): {response.text}')
                
            result = response.json()
            print(f"✅ Transcription reçue: {len(result.get('words', []))} mots")
            return result
            
    except Exception as e:
        print(f"❌ Erreur Whisper: {str(e)}")
        raise

def create_segments(transcription, segment_duration):
    """Créer des segments de durée fixe"""
    segments = []
    total_duration = transcription.get('duration', 0)
    total_segments = int(total_duration / segment_duration) + 1
    
    print(f"📊 Durée: {total_duration}s → {total_segments} segments de {segment_duration}s")
    
    for i in range(total_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, total_duration)
        
        # Mots dans ce segment
        words_in_segment = []
        if 'words' in transcription:
            words_in_segment = [
                word for word in transcription['words']
                if word['start'] >= start_time and word['start'] < end_time
            ]
        
        lyrics_text = ' '.join([w['word'] for w in words_in_segment]).strip()
        
        segments.append({
            'segment_id': i,
            'start_time': round(start_time, 2),
            'end_time': round(end_time, 2),
            'duration': round(end_time - start_time, 2),
            'has_lyrics': len(words_in_segment) > 0,
            'lyrics': lyrics_text,
            'word_count': len(words_in_segment),
            'words': words_in_segment
        })
    
    segments_with_lyrics = len([s for s in segments if s['has_lyrics']])
    print(f"🎤 {segments_with_lyrics} segments avec paroles")
    
    return segments

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "service": "whisper-webhook",
        "openai_configured": bool(OPENAI_API_KEY)
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print("🎙️ Whisper Webhook Server v2 démarré!")
    print(f"📡 Port: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
