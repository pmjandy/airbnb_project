# 🏠Airbnb Price Prediction

## 🛜Render URL
https://airbnb-prediction-rvnm.onrender.com

## 📌Project Overview
에어비앤비 숙소의 다양한 정보를 활용하여 숙소 가격을 예측하는 머신러닝 프로젝트입니다.  
Kaggle에서 제공하는 Airbnb 데이터를 활용하여 데이터 전처리, Feature Engineering, 범주형 변수 인코딩을 수행한 후 TensorFlow 기반 회귀 모델을 구현하여 가격을 에측했습니다.

## 🛠️Development Environment
### Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- TensorFlow(Keras)
- Matplotlib
- Seaborn


## 📊Dataset

Airbnb price prediction  
https://www.kaggle.com/datasets/stevezhenghp/airbnb-price-prediction


## ⚙️Data Preprocessing
- 결측치 및 이상치 제거
- 무의미한 변수 제거(id, name, thumbnail_url 등)
- Amenities 항목 중 필수 11개를 원핫인코딩 하여 별개의 칼럼으로 생성
- property_type 칼럼에서 무의미한 항목 제거(Boat, Castle, Cave 등)
- property_type 정규화 ex) hostel -> guesthouse

최종 데이터  
columns: 27
rows: 55528


##  🤖Model
TensorFlow(Keras)를 활용하여 숙소 가격을 에측하는 회귀 모델을 구현했습니다.

kaggle dataset  
↓  
data preprocessing  
↓  
feature engineering  
↓  
train/test split  
↓  
TensorFlow regression model  
↓  
price prediction


## 📈Result
|	R2 score	|	Mean Squared Error	|
|---|---|
|	0.6255	|	0.1452	|


## 💡Insight
- 숙소의 편의시설이 많을수록 숙박 가격이 높아지는 경향을 확인
- 객실유형, 숙소유형 또한 가격과 높은 연관성을 가짐

## 📷 Demo
사용자는 다음의 정보를 선택합니다.
- property type(apartment / bed&breakfast / bungalow / condominium / guest suite / guesthouse / house / loft / vacation home 중 택1)
- room type(entire home/apt / private room / shared room 중 택1)
- bed type(real bed / airbed / couch / futon / pull-out sofa 중 택1)
- bathrooms
- bedrooms
- beds
- accommodates
- amenities(pool / breakfast / internet / kitchen / free parking on premises / air-conditioning or heating / hot tub / washer / dryer / self check-in / tv 중 복수선택 가능)

  
출력결과
- 1박 당 예상 가격($)
