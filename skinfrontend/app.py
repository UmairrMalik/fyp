import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---- Configuration ----
MODEL_PATH = "final_model.h5"  # <-- change if your file is named differently
IMG_SIZE = 384
CLASS_NAMES = ['acne', 'wrinkles', 'dry', 'oily', 'normal']

# ---- Load model ----
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except:
        return None

model = load_model()

# ---- Page Setup ----
st.set_page_config(page_title="AI Skin Analyzer", layout="centered")
st.title("🧴 AI Skin Condition Analyzer")
st.markdown("Upload a clear photo of your skin and answer a few quick questions for a personalized skincare tip.")

# ---- File Upload ----
file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# ---- Questionnaire ----
st.subheader("📝 Quick Questionnaire")
humidity = st.selectbox("Do you live in a humid environment?", ["Yes", "No"])
sensitivity = st.selectbox("Is your skin generally sensitive?", ["Yes", "No"])
allergies = st.text_input("List any known skin allergies (leave blank if none):")

tips_from_questionnaire = []
if humidity == "Yes":
    tips_from_questionnaire.append("In humid climates, lighter moisturizers or gels are better than heavy creams.")
if sensitivity == "Yes":
    tips_from_questionnaire.append("Avoid products with alcohol, fragrance, or strong exfoliants.")
if allergies:
    tips_from_questionnaire.append("Be cautious with new products. Patch test and avoid: " + allergies)

if file:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # ---- Preprocess ----
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---- Predict ----
    if model:
        prediction = model.predict(img_array)
        pred_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.subheader("🧠 Prediction")
        st.write(f"**Detected Condition:** {pred_class.capitalize()}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # ---- Basic Advice ----
        st.subheader("💡 Suggested Skincare Tip")
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
        st.subheader("🧬 Additional Tips Based on Your Answers")
        for t in tips_from_questionnaire:
            st.write("•", t)

    st.markdown("---")
    st.caption("This is a demo AI tool and not a substitute for professional dermatological advice.")
