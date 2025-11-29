import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_model.pkl")

st.title("🏨 Hotel Booking Cancellation Prediction App")
st.write("Enter booking details below to predict if the booking will be cancelled.")

# ------- USER INPUT FORM -------
col1, col2 = st.columns(2)

with col1:
    no_of_adults = st.number_input("Number of Adults", 1, 10, 2)
    no_of_children = st.number_input("Number of Children", 0, 10, 1)
    no_of_weekend_nights = st.number_input("Weekend Nights", 0, 10, 1)
    no_of_week_nights = st.number_input("Week Nights", 0, 30, 3)
    lead_time = st.number_input("Lead Time (days before arrival)", 0, 500, 45)
    avg_price_per_room = st.number_input("Average Price per Room", 1.0, 10000.0, 120.0)

    # ADD MISSING REQUIRED FIELDS
    arrival_year = st.number_input("Arrival Year", 2015, 2030, 2022)
    arrival_month = st.number_input("Arrival Month (1-12)", 1, 12, 5)
    arrival_date = st.number_input("Arrival Day (1-31)", 1, 31, 15)

with col2:
    type_of_meal_plan = st.selectbox(
        "Meal Plan",
        ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]
    )

    room_type_reserved = st.selectbox(
        "Room Type",
        ["Room_Type 1", "Room_Type 2", "Room_Type 3", 
         "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]
    )

    market_segment_type = st.selectbox(
        "Market Segment",
        ["Online", "Offline", "Corporate", "Aviation", "Complementary"]
    )

    required_car_parking_space = st.selectbox(
        "Car Parking Required?",
        [0, 1]
    )

    repeated_guest = st.selectbox(
        "Is Repeated Guest?",
        [0, 1]
    )

    no_of_previous_cancellations = st.number_input(
        "Previous Cancellations", 0, 20, 0
    )

    no_of_previous_bookings_not_canceled = st.number_input(
        "Previous Successful Bookings", 0, 20, 0
    )

    no_of_special_requests = st.number_input(
        "Special Requests", 0, 5, 1
    )

# -------- FEATURE ENGINEERING (MUST MATCH TRAINING) --------
total_stay_nights = no_of_weekend_nights + no_of_week_nights
total_guests = no_of_adults + no_of_children
if total_guests == 0:
    total_guests = 1

avg_price_per_person = avg_price_per_room / total_guests
weekend_booking_flag = 1 if no_of_weekend_nights > 0 else 0

def lead_time_category(x):
    if x <= 30:
        return "short"
    elif x <= 90:
        return "medium"
    return "long"

lead_time_category_value = lead_time_category(lead_time)

# -------- CREATE INPUT DATA FRAME --------
input_data = pd.DataFrame([{
    "no_of_adults": no_of_adults,
    "no_of_children": no_of_children,
    "no_of_weekend_nights": no_of_weekend_nights,
    "no_of_week_nights": no_of_week_nights,
    "lead_time": lead_time,
    "avg_price_per_room": avg_price_per_room,
    "type_of_meal_plan": type_of_meal_plan,
    "room_type_reserved": room_type_reserved,
    "market_segment_type": market_segment_type,
    "required_car_parking_space": required_car_parking_space,
    "repeated_guest": repeated_guest,
    "no_of_previous_cancellations": no_of_previous_cancellations,
    "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
    "no_of_special_requests": no_of_special_requests,

    # engineered features
    "total_stay_nights": total_stay_nights,
    "total_guests": total_guests,
    "avg_price_per_person": avg_price_per_person,
    "lead_time_category": lead_time_category_value,
    "weekend_booking_flag": weekend_booking_flag,

    # REQUIRED → Missing earlier
    "arrival_year": arrival_year,
    "arrival_month": arrival_month,
    "arrival_date": arrival_date
}])

st.write("### 🔍 Input Summary")
st.write(input_data)

# ------------ PREDICT BUTTON ------------
if st.button("Predict Cancellation"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.error(f"❗ Booking is likely to be **CANCELLED**")
    else:
        st.success(f"✅ Booking is likely to **NOT be cancelled**")

    st.write(f"**Cancellation Probability:** `{probability:.2f}`")
