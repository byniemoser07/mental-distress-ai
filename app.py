from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, numpy as np, librosa, os, tempfile

app = Flask(__name__)
CORS(app)

# Load models
with open("model.pkl", "rb") as f: model = pickle.load(f)
with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f: le = pickle.load(f)

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050, duration=3)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    return np.hstack([mfcc, chroma, zcr])

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_file.save(tmp.name)
    tmp.close()

    try:
        features = extract_features(tmp.name)
        features_scaled = scaler.transform([features])
        audio_probs = model.predict_proba(features_scaled)[0]
        distress_idx = list(le.classes_).index("Distress")
        audio_conf = float(audio_probs[distress_idx])
        final_score = round(audio_conf, 2)

        return jsonify({
            "prediction": "Distress" if final_score > 0.45 else "No Distress",
            "confidence": final_score,
            "audio_confidence": round(audio_conf, 2),
            "nlp_score": 0.0,
            "transcript": "NLP unavailable on free tier"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp.name)

if __name__ == "__main__":
    app.run(debug=False, port=5000)
