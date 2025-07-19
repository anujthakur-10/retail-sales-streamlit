# import streamlit as st
# import requests

# st.set_page_config(page_title="Sales Forecaster", layout="centered")

# st.title("📈 Monthly Sales Forecasting App")
# st.markdown("Enter the last 3 months' sales to predict the next month's sales.")

# # Input fields
# sales1 = st.number_input("Month 1 Sales", value=45000.0)
# sales2 = st.number_input("Month 2 Sales", value=47000.0)
# sales3 = st.number_input("Month 3 Sales", value=49000.0)

# # Predict button
# if st.button("🔮 Predict Next Month's Sales"):
#     payload = {
#         "sales_sequence": [sales1, sales2, sales3]
#     }

#     try:
#         # Call FastAPI
#         response = requests.post("https://retail-sales-fastapi-production.up.railway.app/predict", json=payload)

#         if response.status_code == 200:
#             result = response.json()
#             st.success(f"✅ Predicted Sales: ₹{result['predicted_sales']:,.2f}")
#         else:
#             st.error("❌ Something went wrong with the prediction API.")

#     except Exception as e:
#         st.error(f"⚠️ Could not connect to API: {e}")
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

st.title("📈 Retail Sales Predictor")

# Load the LSTM model (make sure it's uploaded in the same folder)
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_model.h5")

model = load_lstm_model()

# User input
sales_input = st.number_input("Enter last month's sales (₹):", min_value=0)

if st.button("Predict"):
    sequence = np.array([[sales_input]])
    prediction = model.predict(sequence)[0][0]
    st.success(f"📊 Predicted Sales for Next Month: ₹{prediction:.2f}")

