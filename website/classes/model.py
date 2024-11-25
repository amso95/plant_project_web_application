import pandas as pd
import numpy as np
import requests
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup as bs


class Model:

    def __init__(self):
        self.plants_csv = pd.read_csv(r'plantnet300K-water-frequency-air-more-less-sun-hours-location-50-plants.csv')
        self.model_reg = load_model('watering_frequency.keras', compile=False)
        self.model_class = load_model('transfer_learning_EfficientNetB0_plantnet300K_50_classes_tuned_over100mb.keras', compile=False)

    def get_weather_plant_data(self, temperature_data, full_sun_data, partial_sun_data, full_shade_data, sun_hours_data, plant_name):
        self.plants_csv = self.plants_csv.dropna()
        plant_name_matrix = pd.get_dummies(pd.Series(self.plants_csv['Plant Name']), dtype=float)
        sunlight_loc_matrix = pd.get_dummies(pd.Series(self.plants_csv['Sunlight']), dtype=float)
        names = [x for x in plant_name_matrix.columns]
        sunlight_loc = [x for x in sunlight_loc_matrix.columns]
        names_df = pd.DataFrame(columns=names)
        weather_df = pd.DataFrame(temperature_data, columns=["Air temperature in Celicius"])
        sun_hours_df = pd.DataFrame(sun_hours_data, columns=["Sun Hours"])
        sunlight_loc_df = pd.DataFrame(columns=sunlight_loc)
        full_sun_df = pd.DataFrame(full_sun_data, columns=["Full sun"])
        partial_sun_df = pd.DataFrame(partial_sun_data, columns=["Partial sun"])
        full_shade_df = pd.DataFrame(full_shade_data, columns=["Full shade"])
        min_temp_df = pd.DataFrame(columns=['Min Ideal Temperature in Celicius'])
        max_temp_df = pd.DataFrame(columns=['Max Ideal Temperature in Celicius'])
        data = pd.concat([names_df, min_temp_df, max_temp_df, sunlight_loc_df, weather_df, 
                          sun_hours_df, full_shade_df, full_sun_df, partial_sun_df], axis=1)
        min_temp = 0
        max_temp = 0
        full_sun_loc = 0
        partial_sun_loc = 0
        for index, row in self.plants_csv.iterrows():
            if row['Plant Name'] == plant_name:
                min_temp = row['Min Ideal Temperature in Celicius']
                max_temp = row['Max Ideal Temperature in Celicius']
                if row['Sunlight'] == "Full sun location":
                    full_sun_loc = 1
                else:
                    partial_sun_loc = 1
                break
            
        # Fill in zeroes on the plant names except the predicted one
        for col in data.columns:
            if not (col == "Air temperature in Celicius" or col == "Min Ideal Temperature in Celicius" or col == "Max Ideal Temperature in Celicius" or col == "Full sun" or col == "Partial sun" or col == "Full shade" or col == "Sun Hours" or col == "Full sun location" or col == "Partial sun location"):
                data[col].values[:] = 0
            if col == "Min Ideal Temperature in Celicius":
                data[col].values[:] = min_temp
            if col == "Max Ideal Temperature in Celicius":
                data[col].values[:] = max_temp
            if col == "Full sun location":
                data[col].values[:] = full_sun_loc
            if col == "Partial sun location":
                data[col].values[:] = partial_sun_loc
            if col == plant_name:
                data[col].values[:] = 1
        return data
    
    def get_water_frequency(self, data):
        self.plants_csv = self.plants_csv.dropna()
        watering_frequency =  pd.get_dummies(pd.Series(self.plants_csv['Watering Frequency']), dtype=float)
        watering_frequency_names = [x for x in watering_frequency.columns]

        scaler = StandardScaler()
        data_to_predict = scaler.fit_transform(data)

        predictions = self.model_reg.predict(data_to_predict)
        predicted_watering_index= []
        predict_watering_week = []
        predict_hash_map = {}
        for x in range(0, len(predictions) - 1):
            # Get the index of the highest probability
            predicted_watering_index.append(np.argmax(predictions[x], axis=-1))  
        for x in range(0, len(predictions) - 1):
            predict_watering_week.append(watering_frequency_names[predicted_watering_index[x]])
        for x in range(0, len(watering_frequency_names) - 1):
            predict_hash_map[watering_frequency_names[x]] = predict_watering_week.count(watering_frequency_names[x])
        # Get wich key (watering frequency name) has the higets values
        return max(predict_hash_map, key=predict_hash_map.get)

    def get_weather(self, url, USER_AGENT, LANGUAGE):
        session = requests.Session()
        session.headers['User-Agent'] = USER_AGENT
        session.headers['Accept-Language'] = LANGUAGE
        session.headers['Content-Language'] = LANGUAGE
        html = session.get(url)
        # create a new soup
        soup = bs(html.text, "html.parser")
        # store all results on this dictionary
        result = {}
        next_days = []
        days = soup.find("div", attrs={"id": "wob_dp"})
        # print(days)
        for day in days.findAll("div", attrs={"class": "wob_df"}):
            # extract the name of the day
            day_name = day.findAll("div")[0].attrs['aria-label']
            # get weather status for that day
            weather = day.find("img").attrs["alt"]
            temp = day.findAll("span", {"class": "wob_t"})
            # maximum temparature in Celsius, use temp[1].text if you want fahrenheit
            max_temp = temp[0].text
            # minimum temparature in Celsius, use temp[3].text if you want fahrenheit
            min_temp = temp[2].text
            # hum = day.find("span", attrs={"id": "wob_hm"})
            # print(hum)
            next_days.append({"name": day_name, "weather": weather, "max_temp": max_temp, "min_temp": min_temp})
        # append to result
        result['next_days'] = next_days
        return result