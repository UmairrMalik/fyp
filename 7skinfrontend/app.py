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
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        background-color: #fdf6ec;
        text-align: center;
    }
    .big-title {
        font-size: 3.2em;
        font-weight: bold;
        color: #5a4032;
        margin: 1.2em 0 0.4em 0;
    }
    .subtitle {
        font-size: 1.3em;
        color: #8b6d5c;
        margin-bottom: 2em;
    }
    .question-block {
        padding: 2em;
        background-color: #fff8ee;
        border-radius: 12px;
        max-width: 600px;
        margin: auto;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    }
    .question-text {
        font-size: 1.6em;
        margin-bottom: 1em;
        color: #4b372e;
    }
    .button-row {
        display: flex;
        justify-content: center;
        gap: 2em;
    }
    .stButton button {
        background-color: #a67c52;
        color: white;
        font-size: 1.2em;
        padding: 0.6em 2em;
        border-radius: 8px;
        border: none;
        margin-top: 1em;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #8d6441;
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>SkinSnap AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A personalized skincare experience</div>", unsafe_allow_html=True)

# Session state init
if "question_stage" not in st.session_state:
    st.session_state.question_stage = 0

tips_from_questionnaire = []

# ---- Questionnaire ----
def question_block(text, yes_key, no_key, yes_next, no_next):
    st.markdown(f"<div class='question-block'><div class='question-text'>{text}</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        if st.button("Yes", key=yes_key):
            st.session_state.question_stage = yes_next
    with col2:
        if st.button("No", key=no_key):
            st.session_state.question_stage = no_next
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.question_stage == 0:
    question_block("Ready to begin your skincare journey?", "start_yes", "start_no", 1, 0)

elif st.session_state.question_stage == 1:
    question_block("Do you live in a humid environment?", "humid_yes", "humid_no", 2, 2)
    if "humid" not in st.session_state:
        st.session_state.humid = None
    if st.button("Yes", key="humid_yes_val"):
        st.session_state.humid = "Yes"
        st.session_state.question_stage = 2
    if st.button("No", key="humid_no_val"):
        st.session_state.humid = "No"
        st.session_state.question_stage = 2

elif st.session_state.question_stage == 2:
    question_block("Is your skin generally sensitive?", "sensitive_yes", "sensitive_no", 3, 3)
    if st.button("Yes", key="sensitive_yes_val"):
        st.session_state.sensitive = "Yes"
        st.session_state.question_stage = 3
    if st.button("No", key="sensitive_no_val"):
        st.session_state.sensitive = "No"
        st.session_state.question_stage = 3

elif st.session_state.question_stage == 3:
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
