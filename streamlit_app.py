import requests
import base64
import streamlit as st
from PIL import Image
from inference import get_flower_name

st.title("Food Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Classify'):
        try:
            image_bytes = uploaded_file.getvalue()
            food_name = get_flower_name(image_bytes=image_bytes)
            food_name = food_name.replace('_', ' ')

            st.write("Food Category:", food_name)

            api_key = "X9PChaFX2FvOzlTiUXYUNGU5IfVcPeNyM3Pf5Ftc"
            url = "https://api.nal.usda.gov/fdc/v1/foods/search"
            params = {"query": food_name, "api_key": api_key}
            response = requests.get(url, params=params)
            data = response.json()

            if 'foods' in data and len(data['foods']) > 0:
                st.write("Nutritional Information:")
                for item in data['foods']:
                    st.write("Name:", item['description'])
                    st.write("Brand Owner:", item.get('brandOwner', 'N/A'))
                    st.write("Ingredients:", item.get('ingredients', 'N/A'))
                    st.write("Serving Size:", item.get('servingSize', 'N/A'))
                    st.write("Serving Size Unit:", item.get('servingSizeUnit', 'N/A'))
                    if 'foodNutrients' in item:
                        st.write("Nutrients:")
                        for nutrient in item['foodNutrients']:
                            st.write(nutrient['nutrientName'], ":", nutrient['value'], nutrient['unitName'])
                    st.write("---")
            else:
                st.write("Nutritional Information not available.")

        except Exception as e:
            st.write("Error occurred:", e)
