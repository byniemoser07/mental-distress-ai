import streamlit as st
import requests

st.set_page_config(page_title="Mental Distress Detector", page_icon="🧠", layout="centered")
st.title("🧠 Mental Distress Detection System")
st.markdown("---")

BACKEND = "http://127.0.0.1:5000"

uploaded = st.file_uploader("🎙️ Upload Audio File", type=["wav","mp3","ogg","m4a"])

if uploaded:
    st.audio(uploaded)
    st.write(f"File name: {uploaded.name}")
    st.write(f"File size: {uploaded.size} bytes")
    st.write(f"File type: {uploaded.type}")

    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                audio_bytes = uploaded.getvalue()
                st.write(f"Bytes being sent: {len(audio_bytes)}")
                
                files = {"audio": (uploaded.name, audio_bytes, "audio/wav")}
                resp = requests.post(f"{BACKEND}/predict", files=files, timeout=300)
                
                st.write(f"Status code: {resp.status_code}")
                st.write(f"Raw response: {resp.text}")
                
                r = resp.json()

                if r.get("error"):
                    st.error(f"Backend error: {r['error']}")
                else:
                    if r["prediction"] == "Distress":
                        st.error("## ⚠️ DISTRESS DETECTED")
                    else:
                        st.success("## ✅ NO DISTRESS DETECTED")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("🎯 Overall",     f"{r['confidence']*100:.1f}%")
                    c2.metric("🔊 Audio Model", f"{r['audio_confidence']*100:.1f}%")
                    c3.metric("📝 NLP Score",   f"{r['nlp_score']*100:.1f}%")
                    st.progress(r["confidence"])
                    st.markdown("### 📝 Transcript")
                    st.info(r["transcript"] if r["transcript"].strip() else "*(no speech detected)*")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach backend. Make sure Flask is running.")
            except Exception as e:
                st.error(f"❌ Exception: {type(e).__name__}: {e}")
