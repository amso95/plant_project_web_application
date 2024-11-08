import os 
from flask import Flask, render_template, flash, Response, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import datetime, time
import glob
from classes.model import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory

# sess = requests.Session()

UPLOAD_FOLDER = 'C:/Users/Amand/Documents/OPA23HA/Lia/Sigma/code/website/static/images'

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
# US english
LANGUAGE = "en-US,en;q=0.5"

TARGET_SIZE = (224, 224)

CLASS_NAMES = ["Lactuca virosa L.", "Pelargonium graveolens L'Hér.", "Cirsium arvense (L.) Scop.", 
               "Cirsium vulgare (Savi) Ten.", "Pelargonium zonale (L.) L'Hér.", "Mercurialis annua L.",
               "Hypericum perforatum L.", "Tradescantia fluminensis Vell.", "Lamium amplexicaule L.", 
               "Lavandula dentata L.", "Melilotus albus Medik.", "Dryopteris filix-mas (L.) Schott", 
               "Nephrolepis cordifolia (L.) C. Presl", "Nephrolepis exaltata (L.) Schott", "Osmunda regalis L.", 
               "Lithodora fruticosa (L.) Griseb.", "Humulus lupulus L.", "Vaccaria hispanica (Mill.) Rauschert", 
               "Calendula officinalis L.", "Carthamus lanatus L.", "Helminthotheca echioides (L.) Holub", 
               "Lactuca muralis (L.) Gaertn.", "Limbarda crithmoides (L.) Dumort.", "Sedum acre L.", 
               "Sedum album L.", "Sedum dasyphyllum L.", "Sedum sediforme (Jacq.) Pau", 
               "Alliaria petiolata (M. Bieb.) Cavara & Grande", "Mercurialis perennis L.", "Hypericum androsaemum L.", 
               "Hypericum hirsutum L.", "Hypericum tetrapterum Fr.", "Lamium hybridum Vill.", 
               "Lamium purpureum L.", "Lavandula stoechas L.", "Galega officinalis L.", 
               "Trifolium angustifolium L.", "Trifolium arvense L.", "Trifolium aureum Pollich", 
               "Trifolium campestre Schreb.", "Trifolium hybridum L.", "Trifolium incarnatum L.", 
               "Trifolium montanum L.", "Trifolium pratense L.", "Trifolium resupinatum L.", 
               "Trifolium stellatum L.", "Punica granatum L.", "Alcea rosea L.", 
               "Althaea cannabina L.", "Althaea officinalis L."]

SUNLIGHT = {"Molnigt": "Full shade", "Soligt": "Full sun", "Övervägande molnigt": "Full shade", "Växlande molnighet": "Partial sun"}

global capture,rec_frame, rec, out, camera
capture=0
rec=0

camera = cv2.VideoCapture(0)
model = Model()

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
            return uploaded('Picture successfully taken')
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return uploaded('File uploaded successfully')

    if request.method == 'GET':
        global camera
        camera = cv2.VideoCapture(0)
        # Remove predicted image from folder
        removing_image = glob.glob(UPLOAD_FOLDER + '/*.jpg')
        for i in removing_image:
            os.remove(i)
    return render_template('index.html')

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

@app.route('/predictions/')
def run_model():
    global camera
    camera.release()
    cv2.destroyAllWindows()

    img_path = len(os.listdir(UPLOAD_FOLDER))
    if img_path != 0:
        predict_folder = "C:/Users/Amand/Documents/OPA23HA/Lia/Sigma/code/website/static"
        image_to_predict = image_dataset_from_directory(
            predict_folder,
            batch_size=32,
            image_size=TARGET_SIZE,
            shuffle=False
        )
        
        # Prefetch
        AUTOTUNE = tf.data.AUTOTUNE
        image_to_predict = image_to_predict.prefetch(buffer_size=AUTOTUNE)
        
        # Classify the image
        predictions = model.model_class.predict(image_to_predict)
        predicted_class_index = np.argmax(predictions, axis=-1)[0]  # Get the index of the highest probability
        predicted_class = CLASS_NAMES[predicted_class_index]  # Map index to class name

        URL = "https://www.google.com/search?lr=lang_en&ie=UTF-8&q=weather"
        parser = argparse.ArgumentParser(description="Quick Script for Extracting Weather data using Google Weather")
        parser.add_argument("region", nargs="?", help="""Region to get weather for, must be available region.
                                            Default is your current location determined by your IP Address""", default="")
        # parse arguments
        args = parser.parse_args()
        region = args.region
        URL += region
        # get data
        data = model.get_weather(URL, USER_AGENT, LANGUAGE)
        temperature_data = []
        sunlight_data = []
        full_sun_data = []
        partial_sun_data = []
        full_shade_data = []
        for dayweather in data["next_days"]:
            temperature_data.append(dayweather['max_temp'])
            temperature_data.append(dayweather['min_temp'])
            if dayweather['weather'] == "Soligt":
                full_sun_data.append(1)
                full_sun_data.append(1)
            else:
                full_sun_data.append(0)
                full_sun_data.append(0)
            if dayweather['weather'] == "Växlande molnighet":
                partial_sun_data.append(1)
                partial_sun_data.append(1)
            else:
                partial_sun_data.append(0)
                partial_sun_data.append(0)
            if dayweather['weather'] == "Molnigt" or dayweather['weather'] == "Övervägande molnigt":
                full_shade_data.append(1)
                full_shade_data.append(1)
            else:
                full_shade_data.append(0)
                full_shade_data.append(0)

        plant_weather_data = model.get_weather_plant_data(temperature_data, full_sun_data, partial_sun_data, full_shade_data, predicted_class)
        predicted_water_frequency = model.get_water_frequency(plant_weather_data)
        image_name = os.listdir(UPLOAD_FOLDER)

        if len(predicted_class) != 0:
            return render_template('prediction.html', predicted_plant=predicted_class, predicted_water_frequency=predicted_water_frequency, image_name=image_name)
    return render_template('prediction.html')

@app.route('/uploaded/')
def uploaded(msg):
    return render_template('uploaded.html', msg=msg)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frame():
    global capture
    while True:
        success, frame = camera.read() 
        if success:  
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['C:/Users/Amand/Documents/OPA23HA/Lia/Sigma/code/website/static/images', "shot_{}.jpg".format(str(now).replace(":",''))]) #uploaded-image/to-predict
                cv2.imwrite(p, frame)
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)
    
if __name__ == '__main__':
  app.secret_key = 'super secret key'
  app.config['SESSION_TYPE'] = 'filesystem'

  app.run(debug=True)