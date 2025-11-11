import streamlit as st
import numpy as np
import pickle

# =============================
# ✅ Load Trained Models + Scaler
# =============================
xgb_model, rf_model, scaler = pickle.load(open("house_price_model.pkl", "rb"))

# =============================
# ✅ Streamlit UI
# =============================
st.set_page_config(page_title="🏡 House Price Prediction", layout="centered")

st.title("🏡 House Price Prediction App")
st.write("Enter the details below to predict the estimated house price.")

# =============================
# ✅ Input fields
# =============================
area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=2000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
stories = st.number_input("Number of Stories", min_value=1, max_value=4, value=2)

mainroad = st.selectbox("Main Road Access", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
parking = st.slider("Parking Spaces", 0, 5, 1)
prefarea = st.selectbox("Preferred Area", ["yes", "no"])
furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# =============================
# ✅ Convert categorical to numeric
# =============================
def encode_binary(value):
    return 1 if value == "yes" else 0

def encode_furnishing(value):
    if value == "furnished":
        return 2
    elif value == "semi-furnished":
        return 1
    else:
        return 0

mainroad = encode_binary(mainroad)
guestroom = encode_binary(guestroom)
basement = encode_binary(basement)
hotwaterheating = encode_binary(hotwaterheating)
airconditioning = encode_binary(airconditioning)
prefarea = encode_binary(prefarea)
furnishingstatus = encode_furnishing(furnishingstatus)

# =============================
# ✅ Feature Engineering (same as model)
# =============================
rooms_total = bedrooms + bathrooms
price_per_sqft = area / (stories + 1)

# =============================
# ✅ Prepare input for prediction
# =============================
input_data = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom,
                        basement, hotwaterheating, airconditioning, parking,
                        prefarea, furnishingstatus, rooms_total, price_per_sqft]])

# =============================
# ✅ Prediction
# =============================
if st.button("🔍 Predict House Price"):
    input_scaled = scaler.transform(input_data)

    xgb_pred = xgb_model.predict(input_scaled)
    rf_pred = rf_model.predict(input_scaled)

    final_pred = (0.7 * xgb_pred) + (0.3 * rf_pred)
    prediction = np.expm1(final_pred[0])  # reverse log transform

    st.success(f"🏠 Estimated House Price: ₹{prediction:,.2f}")

st.caption("Model Accuracy: ~90% | Ensemble of XGBoost + Random Forest")
