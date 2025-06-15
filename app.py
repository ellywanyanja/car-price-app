import streamlit as st
import pandas as pd
import joblib

# Load model and tools
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder_columns = joblib.load('encoder_columns.pkl')

# Page settings
st.set_page_config(page_title="Car Price Predictor", page_icon="", layout="centered")

st.title("Car Price Prediction App")
st.markdown("Use this smart tool to estimate your car's resale value!")

# Sidebar Inputs
st.sidebar.header("Enter Car Details")

brand = st.sidebar.selectbox("Brand", ['Tata', 'Maruti', 'Honda', 'Mahindra', 'Toyota', 'Renault',
                                       'Volkswagen', 'Kia', 'Ford', 'Hyundai'])

model_name = st.sidebar.selectbox("Model", ['Nexon', 'XUV500', 'City', 'Swift', 'Duster', 'Polo',
                                            'Ecosport', 'Seltos', 'Innova', 'Creta'])

fuel_type = st.sidebar.selectbox("Fuel Type", ['Hybrid', 'Diesel', 'Electric', 'Petrol', 'CNG'])
transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
owner_type = st.sidebar.selectbox("Owner Type", ['First', 'Second', 'Third'])
mileage = st.sidebar.number_input("Mileage (km/l)", min_value=5.0, max_value=35.0, value=18.0)
car_age = st.sidebar.slider("Car Age (Years)", 0, 25, 5)

st.markdown("---")

if st.button("Predict Sale Price"):
    input_dict = {col: 0 for col in encoder_columns}

    for prefix, value in {
        'brand': brand,
        'model': model_name,
        'fuel_type': fuel_type,
        'transmission': transmission,
        'owner_type': owner_type
    }.items():
        col_name = f"{prefix}_{value}"
        if col_name in input_dict:
            input_dict[col_name] = 1

    input_dict['mileage_kmpl'] = mileage
    input_dict['car_age'] = car_age

    input_df = pd.DataFrame([input_dict])
    input_df[['mileage_kmpl', 'car_age']] = scaler.transform(input_df[['mileage_kmpl', 'car_age']])

    predicted_price = model.predict(input_df)[0]

    st.success("Prediction Complete!")
    st.metric(label=" Estimated Sale Price", value=f"${predicted_price:,.2f}")

    with st.expander("View Input Summary"):
        st.json({
            "Brand": brand,
            "Model": model_name,
            "Fuel": fuel_type,
            "Transmission": transmission,
            "Owner Type": owner_type,
            "Mileage (km/l)": mileage,
            "Car Age": car_age
        })

else:
    st.info("Fill in the details and click **Predict Sale Price**.")

st.markdown("---")
st.caption("Built with love using Streamlit")
