import os
import numpy as np
import pandas as pd
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model


# =========================================================
# 모델
# =========================================================

MODEL_PATH = "airbnb_model.keras"

model = None


def get_model():
    global model

    if model is None:
        model = load_model(MODEL_PATH)

    return model


# =========================================================
# 학습할 때 사용한 LabelEncoder의 매핑
# =========================================================
# 주의:
# 아래 값은 반드시 학습 당시 LabelEncoder가 만든 mapping과
# 동일해야 합니다.
#
# 따라서 학습에 사용한 원본 데이터를 app.py에서 직접 읽어서
# LabelEncoder를 다시 fit하는 방식으로 구성합니다.
# =========================================================


DATA_PATH = "airbnb.xlsx"

df = pd.read_excel(DATA_PATH)


# 학습 당시와 동일하게 categorical column을 찾음
categorical_columns = df.select_dtypes(
    include=["object", "bool"]
).columns


# 학습 당시와 동일한 LabelEncoder 생성
from sklearn.preprocessing import LabelEncoder

encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    le.fit(df[column])
    encoders[column] = le


# =========================================================
# Feature 순서
# =========================================================

feature_columns = [
    "bathrooms",
    "bedrooms",
    "beds",
    "amenities_count",
    "accommodates",
    "latitude",
    "longitude",
    "number_of_reviews",
    "review_scores_rating",
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
    "tv",
    "property_type_encoded",
    "room_type_encoded",
    "bed_type_encoded",
    "cancellation_policy_encoded",
    "cleaning_fee_encoded",
    "city_encoded"
]


# =========================================================
# Gradio에서 사용할 선택지
# =========================================================

property_types = sorted(
    df["property_type"].dropna().unique().tolist()
)

room_types = sorted(
    df["room_type"].dropna().unique().tolist()
)

cities = sorted(
    df["city"].dropna().unique().tolist()
)

bed_types = sorted(
    df["bed_type"].dropna().unique().tolist()
)

cancellation_policies = sorted(
    df["cancellation_policy"].dropna().unique().tolist()
)

cleaning_fee_values = sorted(
    df["cleaning_fee"].dropna().unique().tolist()
)


# Amenities
amenity_columns = [
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
]


# =========================================================
# 예측 함수
# =========================================================

def predict_price(
    property_type,
    room_type,
    city,
    accommodates,
    bathrooms,
    bedrooms,
    beds,
    bed_type,
    cancellation_policy,
    cleaning_fee,
    latitude,
    longitude,
    number_of_reviews,
    review_scores_rating,
    selected_amenities
):

    try:

        # 선택된 amenities가 없는 경우
        if selected_amenities is None:
            selected_amenities = []


        # ---------------------------------------------
        # amenities 개수
        # ---------------------------------------------

        amenities_count = len(selected_amenities)


        # ---------------------------------------------
        # LabelEncoder 적용
        # ---------------------------------------------

        property_type_encoded = encoders["property_type"].transform(
            [property_type]
        )[0]

        room_type_encoded = encoders["room_type"].transform(
            [room_type]
        )[0]

        bed_type_encoded = encoders["bed_type"].transform(
            [bed_type]
        )[0]

        cancellation_policy_encoded = encoders[
            "cancellation_policy"
        ].transform(
            [cancellation_policy]
        )[0]

        cleaning_fee_encoded = encoders[
            "cleaning_fee"
        ].transform(
            [cleaning_fee]
        )[0]

        city_encoded = encoders["city"].transform(
            [city]
        )[0]


        # ---------------------------------------------
        # Amenities binary feature
        # ---------------------------------------------

        amenity_values = {}

        for amenity in amenity_columns:

            if amenity in selected_amenities:
                amenity_values[amenity] = 1
            else:
                amenity_values[amenity] = 0


        # ---------------------------------------------
        # 모델 입력 데이터
        # ---------------------------------------------

        input_data = pd.DataFrame([{

            "bathrooms": bathrooms,
            "bedrooms": bedrooms,
            "beds": beds,

            "amenities_count": amenities_count,

            "accommodates": accommodates,

            "latitude": latitude,
            "longitude": longitude,

            "number_of_reviews": number_of_reviews,

            "review_scores_rating": review_scores_rating,

            "pool": amenity_values["pool"],
            "breakfast": amenity_values["breakfast"],
            "internet": amenity_values["internet"],
            "kitchen": amenity_values["kitchen"],
            "free parking on premises":
                amenity_values["free parking on premises"],
            "air-conditioning or heating":
                amenity_values["air-conditioning or heating"],
            "hot tub": amenity_values["hot tub"],
            "washer": amenity_values["washer"],
            "dryer": amenity_values["dryer"],
            "self check-in": amenity_values["self check-in"],
            "tv": amenity_values["tv"],

            "property_type_encoded":
                property_type_encoded,

            "room_type_encoded":
                room_type_encoded,

            "bed_type_encoded":
                bed_type_encoded,

            "cancellation_policy_encoded":
                cancellation_policy_encoded,

            "cleaning_fee_encoded":
                cleaning_fee_encoded,

            "city_encoded":
                city_encoded

        }])


        # Feature 순서 정확히 맞추기
        input_data = input_data[feature_columns]


        # ---------------------------------------------
        # 예측
        # ---------------------------------------------

        prediction = get_model().predict(
            input_data,
            verbose=0
        )


        # 모델이 예측하는 값은 log_price
        log_price = float(prediction[0][0])


        # log_price → 실제 가격
        price = np.exp(log_price)


        return f"예상 1박 가격: ${price:,.2f}"


    except Exception as e:

        return f"예측 오류: {str(e)}"


# =========================================================
# Gradio UI
# =========================================================

with gr.Blocks(
    title="Airbnb Price Prediction"
) as demo:

    gr.Markdown(
        """
        # 🏠 Airbnb 숙소 가격 예측

        숙소 정보를 입력하면 **1박당 예상 가격**을 예측합니다.
        """
    )


    # ---------------------------------------------
    # 숙소 기본 정보
    # ---------------------------------------------

    with gr.Row():

        property_type = gr.Dropdown(
            choices=property_types,
            label="Property Type",
            value=property_types[0]
        )

        room_type = gr.Dropdown(
            choices=room_types,
            label="Room Type",
            value=room_types[0]
        )

        city = gr.Dropdown(
            choices=cities,
            label="City",
            value=cities[0]
        )


    # ---------------------------------------------
    # 숙소 규모
    # ---------------------------------------------

    with gr.Row():

        accommodates = gr.Number(
            label="Accommodates",
            value=2,
            minimum=1
        )

        bathrooms = gr.Number(
            label="Bathrooms",
            value=1,
            minimum=0
        )

        bedrooms = gr.Number(
            label="Bedrooms",
            value=1,
            minimum=0
        )

        beds = gr.Number(
            label="Beds",
            value=1,
            minimum=0
        )


    # ---------------------------------------------
    # 추가 정보
    # ---------------------------------------------

    with gr.Row():

        bed_type = gr.Dropdown(
            choices=bed_types,
            label="Bed Type",
            value=bed_types[0]
        )

        cancellation_policy = gr.Dropdown(
            choices=cancellation_policies,
            label="Cancellation Policy",
            value=cancellation_policies[0]
        )

        cleaning_fee = gr.Dropdown(
            choices=cleaning_fee_values,
            label="Cleaning Fee",
            value=cleaning_fee_values[0]
        )


    # ---------------------------------------------
    # 위치 / 리뷰
    # ---------------------------------------------

    with gr.Row():

        latitude = gr.Number(
            label="Latitude",
            value=float(df["latitude"].median())
        )

        longitude = gr.Number(
            label="Longitude",
            value=float(df["longitude"].median())
        )

        number_of_reviews = gr.Number(
            label="Number of Reviews",
            value=0,
            minimum=0
        )

        review_scores_rating = gr.Number(
            label="Review Scores Rating",
            value=float(
                df["review_scores_rating"].median()
            ),
            minimum=0
        )


    # ---------------------------------------------
    # Amenities
    # ---------------------------------------------

    amenities = gr.CheckboxGroup(
        choices=amenity_columns,
        label="Amenities",
        info="여러 항목을 선택할 수 있습니다."
    )


    # ---------------------------------------------
    # 예측 버튼
    # ---------------------------------------------

    predict_button = gr.Button(
        "가격 예측",
        variant="primary"
    )


    # ---------------------------------------------
    # 결과
    # ---------------------------------------------

    result = gr.Textbox(
        label="예측 결과"
    )


    predict_button.click(
        fn=predict_price,

        inputs=[
            property_type,
            room_type,
            city,

            accommodates,
            bathrooms,
            bedrooms,
            beds,

            bed_type,
            cancellation_policy,
            cleaning_fee,

            latitude,
            longitude,
            number_of_reviews,
            review_scores_rating,

            amenities
        ],

        outputs=result
    )


# =========================================================
# 실행
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 7860)
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
