import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ---- Configuration ----
MODEL_PATH = "final_model.h5"
IMG_SIZE = 384
CLASS_NAMES = ['acne', 'wrinkles', 'dry', 'oily', 'normal']

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except:
        return None

model = load_model()

# ---- Page Setup ----
st.set_page_config(page_title="SkinSnap AI", layout="centered")

st.markdown("""
    <style>
    .big-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0.3em;
    }
    .subtitle {
        font-size: 1.2em;
        color: #4f4f4f;
        text-align: center;
    }
    .question-block {
        padding: 20px;
        border-radius: 10px;
        background-color: #f9f9f9;
        margin-bottom: 20px;
        animation: fadeInSlide 0.6s ease-in-out;
        position: relative;
    }
    .character {
        position: absolute;
        right: -60px;
        top: 0;
        width: 80px;
        animation: floaty 2s ease-in-out infinite;
    }
    @keyframes fadeInSlide {
        0% {opacity: 0; transform: translateY(40px);}
        100% {opacity: 1; transform: translateY(0);}
    }
    @keyframes floaty {
        0% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>SkinSnap AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a photo and answer a few skin-related questions for a personalized analysis</div>", unsafe_allow_html=True)

st.markdown("---")

question_stage = st.session_state.get("question_stage", 0)
tips_from_questionnaire = []

# Character animation image
character_html = """<img src='https://cdn-icons-png.flaticon.com/512/4345/4345572.png' class='character'>"""

if question_stage == 0:
    with st.container():
        st.markdown("<div class='question-block'>" + character_html, unsafe_allow_html=True)
        if st.button("Start Questionnaire"):
            st.session_state.question_stage = 1
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

elif question_stage == 1:
    with st.container():
        st.markdown("<div class='question-block'>" + character_html, unsafe_allow_html=True)
        humid = st.radio("Do you live in a humid environment?", ["Yes", "No"], horizontal=True)
        if st.button("Next", key="q1"):
            st.session_state.humid = humid
            st.session_state.question_stage = 2
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

elif question_stage == 2:
    with st.container():
        st.markdown("<div class='question-block'>" + character_html, unsafe_allow_html=True)
        sensitive = st.radio("Is your skin generally sensitive?", ["Yes", "No"], horizontal=True)
        if st.button("Next", key="q2"):
            st.session_state.sensitive = sensitive
            st.session_state.question_stage = 3
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

elif question_stage == 3:
    with st.container():
        st.markdown("<div class='question-block'>" + character_html, unsafe_allow_html=True)
        allergy_options = ["Fragrance", "Alcohol", "Salicylic Acid", "Sulfates", "Other"]
        selected_allergies = st.multiselect("Select any known skin allergies:", allergy_options)

        custom_allergy = ""
        if "Other" in selected_allergies:
            custom_allergy = st.text_input("Please specify your allergy:")
            if custom_allergy:
                selected_allergies = [a for a in selected_allergies if a != "Other"] + [custom_allergy]

        if st.button("Finish Questionnaire"):
            st.session_state.allergies = selected_allergies
            st.session_state.question_stage = 4
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

elif question_stage == 4:
    st.success("Questionnaire completed!")
    time.sleep(1)
    if st.button("Proceed to Upload Image"):
        st.session_state.question_stage = 5
        st.experimental_rerun()

if question_stage >= 4:
    if st.session_state.get("humid") == "Yes":
        tips_from_questionnaire.append("Use lightweight, water-based moisturizers in humid environments.")
    if st.session_state.get("sensitive") == "Yes":
        tips_from_questionnaire.append("Avoid alcohol, fragrance, and harsh exfoliants for sensitive skin.")
    if st.session_state.get("allergies"):
        tips_from_questionnaire.append("Avoid products containing: " + ", ".join(st.session_state.get("allergies")))

# ---- File Upload ----
if question_stage == 5:
    st.subheader("Upload Your Skin Image")
    file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if file:
        image = Image.open(file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if model:
            prediction = model.predict(img_array)
            pred_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            st.subheader("AI Prediction")
            st.write(f"**Detected Condition:** {pred_class.capitalize()}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            st.subheader("Skincare Tip Based on AI")
            tips = {
                'acne': "Use a gentle cleanser and avoid oil-based products. Try salicylic acid.",
                'wrinkles': "Hydrate regularly and apply SPF daily. Consider using retinol.",
                'dry': "Use thick moisturizers and avoid alcohol-based toners.",
                'oily': "Use foaming cleansers and oil-free moisturizers.",
                'normal': "Maintain with gentle, balanced skincare products."
            }
            st.info(tips.get(pred_class, "No suggestion available."))
        else:
            st.warning("Model not loaded. Predictions will be available once the model file is added.")

        if tips_from_questionnaire:
            st.markdown("---")
            st.subheader("Personalized Tips Based on Your Answers")
            for t in tips_from_questionnaire:
                st.write("•", t)

        st.markdown("---")
        st.caption("This is a demo AI tool and not a substitute for professional dermatological advice.")
