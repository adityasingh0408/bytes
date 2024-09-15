from flask import Flask, render_template, request, redirect, url_for, send_file, Response
from flask_pymongo import PyMongo
import cv2
import time
import numpy as np
from keras.models import load_model, save_model
from deepfake_detection import detect_deepfake
import os
from apscheduler.schedulers.background import BackgroundScheduler
import logging
from sklearn.model_selection import train_test_split
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['deepfake_db']
feedback_collection = db['feedback']

for doc in feedback_collection.find():
    print(doc)

app = Flask(__name__)

# MongoDB configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/deepfake_db"
mongo = PyMongo(app)

# Load the model
model1 = load_model(r'C:\projects\deepfake_detection_app\deepfake\25_3.h5')

# Set up logging
logging.basicConfig(filename='retraining.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Scheduler setup
scheduler = BackgroundScheduler()
scheduler.start()

# Global flag to track if detection is running
is_detection_running = False

def load_image(filename):
    if filename is None or filename.strip() == "":
        logging.warning(f"Filename is None or empty.")
        return None
    
    image_path = os.path.join(r'C:\projects\deepfake_detection_app\images', filename)
    logging.info(f"Loading image from path: {image_path}")
    
    if not os.path.exists(image_path):
        logging.warning(f"Image path {image_path} does not exist.")
        return None

    image = cv2.imread(image_path)
    
    if image is None:
        logging.warning(f"Image {image_path} could not be loaded.")
    return image

def preprocess_image(image):
    if image is None:
        logging.warning("Preprocessing failed: Image is None.")
        return None
    
    resized_image = cv2.resize(image, (128, 128))  # Example resizing
    return resized_image

def preprocess_data():
    feedback_data = mongo.db.feedback.find()
    
    images = []
    labels = []
    
    logging.info("Processing feedback data...")
    
    for feedback in feedback_data:
        filename = feedback.get('filename')
        feedback_label = feedback.get('feedback')
        
        if not filename or not feedback_label:
            logging.warning(f"Missing filename or feedback label: {feedback}")
            continue
        
        logging.info(f"Processing file: {filename}")
        image = load_image(filename)
        
        if image is None:
            logging.warning(f"Image {filename} not found or couldn't be loaded.")
            continue
        
        preprocessed_image = preprocess_image(image)
        
        if preprocessed_image is not None:
            images.append(preprocessed_image)
            labels.append(1 if feedback_label == 'correct' else 0)
    
    images = np.array(images)
    labels = np.array(labels)
    
    logging.info(f"Total images processed: {len(images)}")
    logging.info(f"Total labels processed: {len(labels)}")
    
    if len(images) == 0 or len(labels) == 0:
        logging.info("data missing  for training will take time to bring data from database.")
        return None, None, None, None
    
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2)
    
    return X_train, X_val, y_train, y_val

def test_preprocess_data():
    feedback_data = mongo.db.feedback.find()
    
    for feedback in feedback_data:
        filename = feedback.get('filename')
        feedback_label = feedback.get('feedback')
        
        if filename:
            print(f"Processing file: {filename}")
            image = load_image(filename)
            
            if image is not None:
                preprocessed_image = preprocess_image(image)
                print(f"Image processed: {preprocessed_image.shape}")
            else:
                print(f"Image {filename} not loaded.")

def retrain_model():
    global model1
    logging.info('Retraining started')
    
    X_train, X_val, y_train, y_val = preprocess_data()
    
    if X_train is None or X_val is None or y_train is None or y_val is None:
        logging.warning(" Training  not started ")
        return
    
    try:
        model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5)
        save_model(model1, r'C:\projects\deepfake_detection_app\deepfake\25_3.h5')
        logging.info('Retraining completed')
    except Exception as e:
        logging.error(f"Error during retraining: {e}")

# Scheduler job to retrain the model daily at midnight
scheduler.add_job(retrain_model, 'cron', hour=0, minute=0)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        detection_result, accuracy = detect_deepfake(file)
        
        mongo.db.detections.insert_one({
            'filename': file.filename,
            'detection_result': detection_result,
            'accuracy': accuracy,
            'timestamp': time.time()
        })
        
        return render_template('result.html', detection_result=detection_result, accuracy=accuracy)

def detect_barcodes(camera):
    global is_detection_running
    model_images = load_model(r'C:\projects\deepfake_detection_app\deepfake\25_3.h5')

    while is_detection_running:
        success, frame = camera.read()
        if not success:
            break
        
        resized_img = cv2.resize(frame, (224, 224))  # Adjust the resize dimensions to (224, 224)
        input_img = resized_img / 255.0  # Normalize input images to [0, 1]
        input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
            
        predictions1 = model1.predict(input_img)
        pred = "FAKE" if predictions1 >= 0.7 else "REAL"
        print(f'The predicted class of the video is {pred}')

        cv2.putText(frame, f"PREDICTION: {pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate(camera):
    global is_detection_running
    while is_detection_running:
        yield next(detect_barcodes(camera))

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback = request.form.get('feedback')
    filename = request.form.get('filename')
    
    if feedback and filename:
        mongo.db.feedback.insert_one({
            'filename': filename,
            'feedback': feedback,
            'timestamp': time.time()
        })
        
        return redirect(url_for('index'))
    return 'Feedback not submitted.', 400

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/start_detection')
def start_detection():
    global is_detection_running
    if not is_detection_running:
        is_detection_running = True
        camera = cv2.VideoCapture(0)
        return Response(detect_barcodes(camera), mimetype='multipart/x-mixed-replace; boundary=frame')
    return 'Deepfake detection is already running.'

@app.route('/stop_detection')
def stop_detection():
    global is_detection_running
    is_detection_running = False
    time.sleep(2) 
    return 'Deepfake detection stopped.'

@app.route('/retrain_model', methods=['POST'])
def retrain_model_route():
    retrain_model()
    return redirect(url_for('index'))

@app.route('/view_logs')
def view_logs():
    log_file_path = 'retraining.log'
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as file:
            log_content = file.read()
    else:
        log_content = 'Log file not found.'
    
    return render_template('view_logs.html', log_content=log_content)

if __name__ == '__main__':
    app.run(debug=True)
