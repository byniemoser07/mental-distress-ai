from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, numpy as np, librosa, os, tempfile, threading
import whisper

app = Flask(__name__)
CORS(app)

with open("model.pkl", "rb") as f:   model = pickle.load(f)
with open("scaler.pkl", "rb") as f:  scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f: le = pickle.load(f)

whisper_model = None
whisper_ready = False

def load_whisper():
    global whisper_model, whisper_ready
    print("⏳ Loading Whisper in background...")
    whisper_model = whisper.load_model("base")
    whisper_ready = True
    print("✅ Whisper ready!")

threading.Thread(target=load_whisper, daemon=True).start()

DISTRESS_KEYWORDS = [
    # Calls for help
    "help", "save", "please", "emergency", "call",
    # Fear / pain
    "scared", "afraid", "terrified", "fear", "panic", "pain", "hurt", "dying",
    # Sadness / hopelessness
    "crying", "alone", "hopeless", "worthless", "depressed", "suicidal",
    "can't", "cannot", "no more", "give up", "end it",
    # Anger / aggression (real-world distress)
    "fuck", "shit", "hate", "kill", "angry", "rage", "stop", "leave",
    "awful", "terrible", "horrible", "worst",
    # Danger
    "danger", "attack", "threat", "trapped", "stuck"
]

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)
    mfcc   = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    zcr    = np.mean(librosa.feature.zero_crossing_rate(audio))
    return np.hstack([mfcc, chroma, zcr])

def nlp_distress_score(text):
    text_lower = text.lower()
    hits = sum(1 for w in DISTRESS_KEYWORDS if w in text_lower)
    return min(hits / 3.0, 1.0)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "whisper_ready": whisper_ready})

@app.route("/predict", methods=["POST"])
def predict():
    if not whisper_ready:
        return jsonify({"error": "Whisper model still loading, please wait 30 seconds and try again"}), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path)
        features_scaled = scaler.transform([features])
        audio_probs = model.predict_proba(features_scaled)[0]
        distress_idx = list(le.classes_).index("Distress")
        audio_conf = float(audio_probs[distress_idx])

        result = whisper_model.transcribe(tmp_path)
        text = result["text"]
        nlp_conf = nlp_distress_score(text)

        # Slightly lower threshold for real-world audio
        final_score = round(0.7 * audio_conf + 0.3 * nlp_conf, 2)
        prediction = "Distress" if final_score > 0.45 else "No Distress"

        return jsonify({
            "prediction":       prediction,
            "confidence":       final_score,
            "audio_confidence": round(audio_conf, 2),
            "nlp_score":        round(nlp_conf, 2),
            "transcript":       text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    app.run(debug=False, port=5000)
