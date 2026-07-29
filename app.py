import os
import pandas as pd
import numpy as np
import joblib
import gradio as gr
from tensorflow.keras.models import load_model


model = None
encoders = joblib.load("encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

@spaces.GPU
def predict_price(
    property_type,
    room_type,
    bed_type,
    bathrooms,
    bedrooms,
    beds,
    accommodates,
    amenities
):

    global model
    if model is None:
        model = load_model("airbnb_model.keras")

    property_encoded = encoders["property_type"].transform([property_type])[0]
    room_encoded = encoders["room_type"].transform([room_type])[0]
    bed_encoded = encoders["bed_type"].transform([bed_type])[0]

    amenity_dict = {
        "pool":0,
        "breakfast":0,
        "internet":0,
        "kitchen":0,
        "free parking on premises":0,
        "air-conditioning or heating":0,
        "hot tub":0,
        "washer":0,
        "dryer":0,
        "self check-in":0,
        "tv":0
    }

    for item in amenities:
        amenity_dict[item] = 1

    amenities_count = len(amenities)

    
    input_df = pd.DataFrame([{
        "bathrooms": bathrooms,
        "bedrooms": bedrooms,
        "beds": beds,
        "accommodates": accommodates,
        "amenities_count": amenities_count,

        "pool": amenity_dict["pool"],
        "breakfast": amenity_dict["breakfast"],
        "internet": amenity_dict["internet"],
        "kitchen": amenity_dict["kitchen"],
        "free parking on premises": amenity_dict["free parking on premises"],
        "air-conditioning or heating": amenity_dict["air-conditioning or heating"],
        "hot tub": amenity_dict["hot tub"],
        "washer": amenity_dict["washer"],
        "dryer": amenity_dict["dryer"],
        "self check-in": amenity_dict["self check-in"],
        "tv": amenity_dict["tv"],

        "property_type_encoded": property_encoded,
        "room_type_encoded": room_encoded,
        "bed_type_encoded": bed_encoded
    }])

    # 학습과 동일한 순서
    input_df = input_df[feature_columns]

    # Tensor로 변환
    X = input_df.astype(np.float32).to_numpy()

    # 예측
    pred = model.predict(X, verbose=0)
    price = np.expm1(pred[0][0])

    return f"💰 예상 숙박 가격은 ${price:.2f} / 1박 입니다."


with gr.Blocks(title="Airbnb Price Prediction") as demo:

    gr.Markdown("# 🏠 Airbnb Price Prediction")
    gr.Markdown("숙소 정보를 입력하면 예상 1박 가격을 예측합니다.")

    with gr.Row():

        property_type = gr.Dropdown(
            choices=[
                "Apartment",
                "Bed & Breakfast",
                "Bungalow",
                "Condominium",
                "Guest suite",
                "Guesthouse",
                "House",
                "Loft",
                "Vacation home"
            ],
            label="Property Type",
            value="Apartment"
        )

        room_type = gr.Dropdown(
            choices=[
                "Entire home/apt",
                "Private room",
                "Shared room"
            ],
            label="Room Type",
            value="Entire home/apt"
        )

        bed_type = gr.Dropdown(
            choices=[
                "Airbed",
                "Couch",
                "Futon",
                "Pull-out Sofa",
                "Real Bed"
            ],
            label="Bed Type",
            value="Real Bed"
        )


    with gr.Row():

        bathrooms = gr.Number(
            value=1,
            label="Bathrooms"
        )

        bedrooms = gr.Number(
            value=1,
            label="Bedrooms"
        )

        beds = gr.Number(
            value=1,
            label="Beds"
        )

        accommodates = gr.Number(
            value=2,
            label="Accommodates"
        )

    
    amenities = gr.CheckboxGroup(
        choices=[
            "pool",
            "breakfast",
            "internet",
            "kitchen",
            "free parking on premises",
            "air-conditioning or heating",
            "hot tub",
            "washer",
            "dryer",
            "self check-in",
            "tv"
        ],
        label="Amenities"
    )


    predict_button = gr.Button(
        "Predict Price",
        variant="primary"
    )


    output = gr.Textbox(
        label="Predicted Price",
        interactive=False
    )


    predict_button.click(
        fn=predict_price,
        inputs=[
            property_type,
            room_type,
            bed_type,
            bathrooms,
            bedrooms,
            beds,
            accommodates,
            amenities
        ],
        outputs=output
    )

demo.launch()
