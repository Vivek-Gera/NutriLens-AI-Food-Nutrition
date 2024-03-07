import os
import requests
import base64
from flask import Flask, render_template, request
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

from inference import get_flower_name

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        try:
            file = request.files['file']
            image = file.read()
            encoded_image = base64.b64encode(image).decode('utf-8')  # Encode image to base64
            food_name = get_flower_name(image_bytes=image)
            url = "https://api.nal.usda.gov/fdc/v1/foods/search"
            food_name = food_name.replace('_', ' ')
            params = {"query": food_name, "api_key": "X9PChaFX2FvOzlTiUXYUNGU5IfVcPeNyM3Pf5Ftc"}
            print(params)
            response = requests.get(url, params=params)
            data = response.json()

            print(file)
            socketio.emit('result', {'food': food_name, 'data': data, 'image': encoded_image})
            return render_template('result.html', food=food_name, data=data, image=encoded_image)
        except:
            return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/architecture')
def architecture():
    return render_template('architecture.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=os.getenv('PORT', 5000))
