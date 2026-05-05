from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, numpy as np, librosa, os, tempfile

app = Flask(__name__)
CORS(app)

with open("model.pkl", "rb") as f:   model = pickle.load(f)
with open("scaler.pkl", "rb") as f:  scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f: le = pickle.load(f)

DISTRESS_KEYWORDS = [
    "help","scared","afraid","dying","emergency","pain","hurt",
    "please","no","stop","terrible","awful","terrified","save",
    "danger","crying","alone","hopeless","fuck","shit","hate",
    "kill","angry","rage","worthless","depressed","trapped"
]

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)
    mfcc   = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    zcr    = np.mean(librosa.feature.zero_crossing_rate(audio))
    return np.hstack([mfcc, chroma, zcr])

def nlp_distress_score(text):
    return min(sum(1 for w in DISTRESS_KEYWORDS if w in text.lower()) / 3.0, 1.0)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
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

        # No Whisper on deployment — audio model only
        final_score = round(audio_conf, 2)

        return jsonify({
            "prediction":       "Distress" if final_score > 0.45 else "No Distress",
            "confidence":       final_score,
            "audio_confidence": round(audio_conf, 2),
            "nlp_score":        0.0,
            "transcript":       "NLP unavailable on free tier"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    app.run(debug=False, port=5000)
