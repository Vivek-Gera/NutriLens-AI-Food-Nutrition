import requests

def get_nutrition_details(api_key, food_name):
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": food_name,
        "api_key": api_key
    }
    response = requests.get(url, params=params)
    data = response.json()

    if 'foods' in data and len(data['foods']) > 0:
        food_item = data['foods'][0]
        nutrition_details = {
            "Name": food_item['description'],
            "Nutrients": {}  # Dictionary to store all nutrients
        }

        for nutrient in food_item['foodNutrients']:
            nutrient_name = nutrient['nutrientName']
            nutrient_value = nutrient['value']
            nutrient_unit = nutrient['unitName']
            nutrition_details["Nutrients"][nutrient_name] = f"{nutrient_value} {nutrient_unit}"

        return nutrition_details
    else:
        return None

def generate_text(nutrition_details):
    text_output = f"Nutrition Details for {nutrition_details['Name']}:\n"
    for nutrient_name, nutrient_value in nutrition_details["Nutrients"].items():
        text_output += f"{nutrient_name}: {nutrient_value}\n"
    return text_output

def main():
    api_key = "X9PChaFX2FvOzlTiUXYUNGU5IfVcPeNyM3Pf5Ftc"
    food_name = input("Enter the food name: ")
    nutrition_details = get_nutrition_details(api_key, food_name)

    if nutrition_details:
        text_output = generate_text(nutrition_details)
        print(text_output)
    else:
        print("Food item not found.")

if __name__ == "__main__":
    main()
