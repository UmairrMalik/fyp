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
    body {
        background-color: #fdf6ec;
    }
    .big-title {
        font-size: 3em;
        font-weight: bold;
        color: #6b4c3b;
        text-align: center;
        margin-bottom: 0.5em;
        font-family: 'Georgia', serif;
    }
    .subtitle {
        font-size: 1.4em;
        color: #8b6d5c;
        text-align: center;
        font-family: 'Georgia', serif;
    }
    .question-block {
        padding: 30px;
        border-radius: 12px;
        background-color: #fff9f0;
        margin: auto;
        margin-top: 2em;
        width: 70%;
        animation: fadeInSlide 0.6s ease-in-out;
    }
    .question-text {
        font-size: 1.8em;
        text-align: center;
        font-family: 'Georgia', serif;
        color: #4b372e;
        margin-bottom: 1em;
    }
    .btn-container button {
        background-color: #a67c52 !important;
        color: white !important;
        border: none;
        font-size: 1.4em !important;
        margin: 0.5em 1em !important;
        padding: 0.6em 2em !important;
        border-radius: 8px;
        font-family: 'Georgia', serif;
    }
    @keyframes fadeInSlide {
        0% {opacity: 0; transform: translateY(40px);}
        100% {opacity: 1; transform: translateY(0);}
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>SkinSnap AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A personalized skincare experience</div>", unsafe_allow_html=True)

# Session state init
if "question_stage" not in st.session_state:
    st.session_state.question_stage = 0

tips_from_questionnaire = []

# ---- Questionnaire ----
if st.session_state.question_stage == 0:
    with st.container():
        st.markdown("<div class='question-block'><div class='question-text'>Ready to begin your skincare journey?</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Yes", key="start_yes"):
                st.session_state.question_stage = 1
        with col2:
            if st.button("No", key="start_no"):
                st.stop()
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.question_stage == 1:
    with st.container():
        st.markdown("<div class='question-block'><div class='question-text'>Do you live in a humid environment?</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Yes", key="humid_yes"):
                st.session_state.humid = "Yes"
                st.session_state.question_stage = 2
        with col2:
            if st.button("No", key="humid_no"):
                st.session_state.humid = "No"
                st.session_state.question_stage = 2
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.question_stage == 2:
    with st.container():
        st.markdown("<div class='question-block'><div class='question-text'>Is your skin generally sensitive?</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Yes", key="sensitive_yes"):
                st.session_state.sensitive = "Yes"
                st.session_state.question_stage = 3
        with col2:
            if st.button("No", key="sensitive_no"):
                st.session_state.sensitive = "No"
                st.session_state.question_stage = 3
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.question_stage == 3:
    with st.container():
        st.markdown("<div class='question-block'><div class='question-text'>Do you have any known skin allergies?</div>", unsafe_allow_html=True)
        allergy_options = ["Fragrance", "Alcohol", "Salicylic Acid", "Sulfates", "Other"]
        selected_allergies = st.multiselect("Select any known skin allergies:", allergy_options)

        if "Other" in selected_allergies:
            custom_allergy = st.text_input("Please specify your allergy:")
            if custom_allergy:
                selected_allergies = [a for a in selected_allergies if a != "Other"] + [custom_allergy]

        if st.button("Next", key="next_allergy"):
            st.session_state.allergies = selected_allergies
            st.session_state.question_stage = 4
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.question_stage == 4:
    st.success("Questionnaire completed!")
    if st.button("Proceed to Upload Image"):
        st.session_state.question_stage = 5

if st.session_state.question_stage >= 4:
    if st.session_state.get("humid") == "Yes":
        tips_from_questionnaire.append("Use lightweight, water-based moisturizers in humid environments.")
    if st.session_state.get("sensitive") == "Yes":
        tips_from_questionnaire.append("Avoid alcohol, fragrance, and harsh exfoliants for sensitive skin.")
    if st.session_state.get("allergies"):
        tips_from_questionnaire.append("Avoid products containing: " + ", ".join(st.session_state.get("allergies")))

# ---- File Upload ----
if st.session_state.question_stage == 5:
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
