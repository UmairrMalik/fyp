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

st.title("SkinSnap AI")
st.subheader("Answer a few questions and upload a photo to analyze your skin condition")

if "question_stage" not in st.session_state:
    st.session_state.question_stage = 0

tips_from_questionnaire = []

# ---- Step-by-step questions ----
if st.session_state.question_stage == 0:
    if st.button("Start Questionnaire"):
        st.session_state.question_stage = 1

elif st.session_state.question_stage == 1:
    humid = st.radio("Do you live in a humid environment?", ["Yes", "No"], horizontal=True)
    if st.button("Next", key="q1"):
        st.session_state.humid = humid
        st.session_state.question_stage = 2

elif st.session_state.question_stage == 2:
    sensitive = st.radio("Is your skin generally sensitive?", ["Yes", "No"], horizontal=True)
    if st.button("Next", key="q2"):
        st.session_state.sensitive = sensitive
        st.session_state.question_stage = 3

elif st.session_state.question_stage == 3:
    st.markdown("**Do you have any known skin allergies?**")
    allergy_options = ["Fragrance", "Alcohol", "Salicylic Acid", "Sulfates", "Other"]
    selected_allergies = st.multiselect("Select any known skin allergies:", allergy_options)

    if "Other" in selected_allergies:
        custom_allergy = st.text_input("Please specify your allergy:")
        if custom_allergy:
            selected_allergies = [a for a in selected_allergies if a != "Other"] + [custom_allergy]

    if st.button("Finish Questionnaire"):
        st.session_state.allergies = selected_allergies
        st.session_state.question_stage = 4

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
            st.subheader("Additional Tips Based on Your Answers")
            for t in tips_from_questionnaire:
                st.write("•", t)

        st.markdown("---")
        st.caption("This is a demo AI tool and not a substitute for professional dermatological advice.")
