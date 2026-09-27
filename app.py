import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Waste Classification AI",
    page_icon="♻️",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    main_model = tf.keras.models.load_model(
        "models/final_main_category_model.keras",
        compile=False
    )

    hazardous_model = tf.keras.models.load_model(
        "models/hazardous_model.keras",
        compile=False
    )

    organic_model = tf.keras.models.load_model(
        "models/final_organic_model.keras",
        compile=False
    )

    recyclable_model = tf.keras.models.load_model(
        "models/final_recyclable_model.keras",
        compile=False
    )

    non_recyclable_model = tf.keras.models.load_model(
        "models/non_recyclable_model.keras",
        compile=False
    )

    return {
        "main": main_model,
        "hazardous": hazardous_model,
        "organic": organic_model,
        "recyclable": recyclable_model,
        "non_recyclable": non_recyclable_model
    }


models = load_models()


# --------------------------------------------------
# LOAD LABELS
# --------------------------------------------------

with open(
    "models/class_labels.json",
    "r"
) as file:

    labels = json.load(file)


# --------------------------------------------------
# PREPROCESS IMAGE
# --------------------------------------------------

def prepare_image(uploaded_image):

    img = uploaded_image.convert("RGB")

    img = img.resize(
        (224, 224)
    )

    img_array = np.array(
        img,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    return img_array


# --------------------------------------------------
# HIERARCHICAL PREDICTION
# --------------------------------------------------

def predict_waste(uploaded_image):

    img = prepare_image(
        uploaded_image
    )

    # ---------------- LEVEL 1 ----------------

    main_prediction = models["main"].predict(
        img,
        verbose=0
    )[0]

    main_index = int(
        np.argmax(main_prediction)
    )

    main_confidence = float(
        main_prediction[main_index] * 100
    )

    category = labels["main"][
        str(main_index)
    ]


    # ---------------- LEVEL 2 ----------------

    if category == "Hazardous":

        model = models["hazardous"]
        label_group = "hazardous"

    elif category == "Organic":

        model = models["organic"]
        label_group = "organic"

    elif category == "Recyclable":

        model = models["recyclable"]
        label_group = "recyclable"

    else:

        model = models["non_recyclable"]
        label_group = "non_recyclable"


    detail_prediction = model.predict(
        img,
        verbose=0
    )[0]

    detail_index = int(
        np.argmax(detail_prediction)
    )

    detail_confidence = float(
        detail_prediction[detail_index] * 100
    )

    waste_type = labels[label_group][
        str(detail_index)
    ]


    # Top 3 detailed predictions

    top_indices = np.argsort(
        detail_prediction
    )[-3:][::-1]

    top_predictions = []

    for index in top_indices:

        top_predictions.append({

            "class": labels[label_group][
                str(int(index))
            ],

            "confidence": float(
                detail_prediction[index] * 100
            )
        })


    return {

        "category": category,

        "category_confidence":
            main_confidence,

        "waste_type":
            waste_type,

        "waste_confidence":
            detail_confidence,

        "top_predictions":
            top_predictions
    }


# --------------------------------------------------
# DISPLAY NAME HELPER
# --------------------------------------------------

def clean_name(name):

    replacements = {

        "e-waste":
            "E-Waste",

        "coffee_tea_bags":
            "Coffee / Tea Bags",

        "egg_shells":
            "Egg Shells",

        "food_scraps":
            "Food Scraps",

        "kitchen_waste":
            "Kitchen Waste",

        "yard_trimmings":
            "Yard Trimmings",

        "cans_all_type":
            "Cans",

        "glass_containers":
            "Glass Containers",

        "paper_products":
            "Paper Products",

        "plastic_bottles":
            "Plastic Bottles",

        "ceramic_product":
            "Ceramic Products",

        "diapers":
            "Diapers",

        "platics_bags_wrappers":
            "Plastic Bags / Wrappers",

        "sanitary_napkin":
            "Sanitary Napkins",

        "stroform_product":
            "Styrofoam Products",

        "batteries":
            "Batteries",

        "paints":
            "Paints",

        "pesticides":
            "Pesticides"
    }

    return replacements.get(
        name,
        name.replace("_", " ").title()
    )


# --------------------------------------------------
# DISPOSAL GUIDANCE
# --------------------------------------------------

def disposal_guidance(category):

    guidance = {

        "Hazardous":
            "⚠️ Keep this item separate from general waste. "
            "Use an approved hazardous-waste collection or disposal facility.",

        "Organic":
            "🌱 This material may be suitable for composting or "
            "organic-waste processing where appropriate.",

        "Recyclable":
            "♻️ Clean the item when possible and place it in the "
            "appropriate recycling collection stream.",

        "Non-Recyclable":
            "🗑️ This item generally requires residual/general-waste "
            "disposal according to local waste-management rules."
    }

    return guidance.get(
        category,
        "Follow your local waste-management guidelines."
    )


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title(
    "♻️ Smart Waste Classification AI"
)

st.caption(
    "Hierarchical Deep Learning System • MobileNetV2 Transfer Learning"
)

st.markdown("---")


uploaded_file = st.file_uploader(

    "Upload a waste image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


if uploaded_file is not None:

    uploaded_image = Image.open(
        uploaded_file
    )


    col1, col2 = st.columns(
        [1, 1.3]
    )


    with col1:

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            uploaded_image,
            use_container_width=True
        )


    with col2:

        st.subheader(
            "AI Classification"
        )


        with st.spinner(
            "Analyzing waste image..."
        ):

            result = predict_waste(
                uploaded_image
            )


        st.success(
            "Classification complete"
        )


        st.metric(

            "Main Waste Category",

            result["category"],

            f'{result["category_confidence"]:.2f}% confidence'
        )


        st.metric(

            "Specific Waste Type",

            clean_name(
                result["waste_type"]
            ),

            f'{result["waste_confidence"]:.2f}% confidence'
        )


        st.markdown(
            "### Main Category Confidence"
        )

        st.progress(
            min(
                result["category_confidence"] / 100,
                1.0
            )
        )


        st.markdown(
            "### Specific Classification Confidence"
        )

        st.progress(
            min(
                result["waste_confidence"] / 100,
                1.0
            )
        )


    st.markdown("---")


    st.subheader(
        "🔍 Top Predictions"
    )


    for i, prediction in enumerate(
        result["top_predictions"],
        start=1
    ):

        st.write(
            f'**{i}. '
            f'{clean_name(prediction["class"])}** '
            f'— {prediction["confidence"]:.2f}%'
        )


    st.markdown("---")


    st.subheader(
        "Waste Handling Guidance"
    )


    st.info(
        disposal_guidance(
            result["category"]
        )
    )


    if result["category_confidence"] < 50:

        st.warning(
            "The main-category confidence is relatively low. "
            "Consider capturing another image with the waste item "
            "clearly visible against a simple background."
        )


    if result["waste_confidence"] < 50:

        st.warning(
            "The detailed classification confidence is relatively low. "
            "The item may visually resemble multiple waste types."
        )


else:

    st.info(
        "Upload a waste image to begin classification."
    )


st.markdown("---")

st.caption(
    "IT2011 Artificial Intelligence and Machine Learning • "
    "Hierarchical Waste Classification Project"
)