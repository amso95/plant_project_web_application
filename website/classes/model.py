import pandas as pd
import numpy as np
import requests
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup as bs


class Model:

    def __init__(self):
        self.plants_csv = pd.read_csv(r'plantnet300K-water-frequency-air-more-less.csv')
        self.model_reg = load_model('watering_frequency.keras', compile=False)
        self.model_class = load_model('transfer_learning_EfficientNetB0_plantnet300K_50_classes_tuned.keras', compile=False)

    def get_weather_plant_data(self, temperature_data, full_sun_data, partial_sun_data, full_shade_data, plant_name):
        #plants_csv = pd.read_csv(r'plantnet300K-water-frequency-air-more-less.csv')
        self.plants_csv = self.plants_csv.dropna()
        plant_name_matrix = pd.get_dummies(pd.Series(self.plants_csv['Plant Name']), dtype=float)
        names = [x for x in plant_name_matrix.columns]
        names_df = pd.DataFrame(columns=names)
        weather_df = pd.DataFrame(temperature_data, columns=["Air temperature in Celicius"])
        full_sun_df = pd.DataFrame(full_sun_data, columns=["Full sun"])
        partial_sun_df = pd.DataFrame(partial_sun_data, columns=["Partial sun"])
        full_shade_df = pd.DataFrame(full_shade_data, columns=["Full shade"])
        data = pd.concat([names_df, weather_df, full_sun_df, partial_sun_df, full_shade_df], axis=1)
        for col in data.columns:
            if col != "Air temperature in Celicius":
                data[col].values[:] = 0
            if col != "Full sun":
                data[col].values[:] = 0
            if col != "Partial sun":
                data[col].values[:] = 0
            if col != "Full shade":
                data[col].values[:] = 0
            if col == plant_name:
                data[col].values[:] = 1
        return data
    
    def get_water_frequency(self, data):
        #plants_csv = pd.read_csv(r'plantnet300K-water-frequency-air-more-less.csv')
        self.plants_csv = self.plants_csv.dropna()
        watering_frequency =  pd.get_dummies(pd.Series(self.plants_csv['Watering Frequency']), dtype=float)
        watering_frequency_names = [x for x in watering_frequency.columns]

        #model = load_model('watering_frequency.keras', compile=False)
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
            next_days.append({"name": day_name, "weather": weather, "max_temp": max_temp, "min_temp": min_temp})
        # append to result
        result['next_days'] = next_days
        return result