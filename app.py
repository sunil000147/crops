import streamlit as st
import joblib
import os
import pandas as pd

# Load the saved models and encoders
@st.cache_resource
def load_models():
    try:
        # Construct the full path to the models directory
        models_dir = 'C:/crops'  # Adjust the path based on your directory structure
        
        # Load models
        yield_predictor = joblib.load(os.path.join(models_dir, 'yield_predictor.joblib'))
        label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.joblib'))
        
        return yield_predictor, label_encoders
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def predict_yield(farm_data, yield_predictor, label_encoders):
    # Prepare input data
    df = pd.DataFrame([farm_data])
    
    # Encode categorical columns
    for col, le in label_encoders.items():
        df[f'{col}_encoded'] = le.transform(df[col])
    
    # Select features
    features = [
        'Farm_Area(acres)', 
        'Crop_Type_encoded', 
        'Irrigation_Type_encoded', 
        'Fertilizer_Used(tons)', 
        'Pesticide_Used(kg)', 
        'Soil_Type_encoded', 
        'Season_encoded', 
        'Water_Usage(cubic meters)'
    ]
    
    X_input = df[features]
    
    # Predict yield
    return yield_predictor.predict(X_input)[0]

def main():
    # Set page title and favicon
    st.set_page_config(page_title="Crop Yield Predictor", page_icon=":corn:")
    
    # Load models
    yield_predictor, label_encoders = load_models()
    
    if yield_predictor is None or label_encoders is None:
        st.error("Failed to load models. Please check your model files.")
        return

    # Title and description
    st.title("🌾 Crop Yield Prediction System")
    st.write("Predict crop yield based on farm conditions")

    # Sidebar for input
    st.sidebar.header("Farm Conditions Input")

    # Input fields
    farm_conditions = {}

    # Numeric inputs
    farm_conditions['Farm_Area(acres)'] = st.sidebar.number_input(
        "Farm Area (acres)", 
        min_value=0.0, 
        max_value=1000.0, 
        value=100.0,
        step=10.0
    )

    farm_conditions['Fertilizer_Used(tons)'] = st.sidebar.number_input(
        "Fertilizer Used (tons)", 
        min_value=0.0, 
        max_value=50.0, 
        value=5.0,
        step=1.0
    )

    farm_conditions['Pesticide_Used(kg)'] = st.sidebar.number_input(
        "Pesticide Used (kg)", 
        min_value=0.0, 
        max_value=20.0, 
        value=2.0,
        step=0.5
    )

    farm_conditions['Water_Usage(cubic meters)'] = st.sidebar.number_input(
        "Water Usage (cubic meters)", 
        min_value=0, 
        max_value=100000, 
        value=50000,
        step=1000
    )

    # Categorical inputs using the existing label encoders
    farm_conditions['Crop_Type'] = st.sidebar.selectbox(
        "Crop Type", 
        options=list(label_encoders['Crop_Type'].classes_)
    )

    farm_conditions['Irrigation_Type'] = st.sidebar.selectbox(
        "Irrigation Type", 
        options=list(label_encoders['Irrigation_Type'].classes_)
    )

    farm_conditions['Soil_Type'] = st.sidebar.selectbox(
        "Soil Type", 
        options=list(label_encoders['Soil_Type'].classes_)
    )

    farm_conditions['Season'] = st.sidebar.selectbox(
        "Season", 
        options=list(label_encoders['Season'].classes_)
    )

    # Prediction button
    if st.sidebar.button("Predict Crop Yield"):
        try:
            # Make prediction
            predicted_yield = predict_yield(farm_conditions, yield_predictor, label_encoders)
            
            # Display result
            st.success(f"Predicted Crop Yield: {predicted_yield:.2f} tons")
            
            # Additional visualizations or insights
            st.subheader("Prediction Breakdown")
            
            # Create two columns for additional insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="Farm Area", value=f"{farm_conditions['Farm_Area(acres)']} acres")
                st.metric(label="Fertilizer Used", value=f"{farm_conditions['Fertilizer_Used(tons)']} tons")
            
            with col2:
                st.metric(label="Water Usage", value=f"{farm_conditions['Water_Usage(cubic meters)']} cubic meters")
                st.metric(label="Crop Type", value=farm_conditions['Crop_Type'])
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

    # About section
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app predicts crop yield based on various farm conditions. "
        "Input your farm details and get an estimated crop yield."
    )

if __name__ == "__main__":
    main()
