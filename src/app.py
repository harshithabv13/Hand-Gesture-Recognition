import os
import sqlite3
from datetime import timedelta
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, redirect, url_for, render_template, session, flash, send_from_directory
import numpy as np
import cv2
import tensorflow as tf

# ==========================================
# Core Configuration & Paths
# ==========================================
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.abspath(os.path.join(APP_ROOT, '..', 'uploads'))
DATABASE_FILE = os.path.abspath(os.path.join(APP_ROOT, '..', 'users.db'))
WEIGHTS_FILE = os.path.abspath(os.path.join(APP_ROOT, '..', 'models', 'mobilenet.h5'))

VALID_FORMATS = {'png', 'jpg', 'jpeg', 'bmp'}
TARGET_HEIGHT = 128
TARGET_WIDTH = 128
TARGET_LABELS = ['FIST', 'ONE', 'PALM', 'SUPER']

os.makedirs(STORAGE_DIR, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(APP_ROOT, 'templates'))
app.secret_key = os.urandom(32)
app.config['UPLOAD_FOLDER'] = STORAGE_DIR
app.permanent_session_lifetime = timedelta(days=7)

# ==========================================
# Database persistence subroutines
# ==========================================
def establish_database():
    connection = sqlite3.connect(DATABASE_FILE)
    db_cursor = connection.cursor()
    db_cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    connection.commit()
    connection.close()

def register_new_profile(username, password):
    connection = sqlite3.connect(DATABASE_FILE)
    db_cursor = connection.cursor()
    secure_hash = generate_password_hash(password)
    try:
        db_cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, secure_hash))
        connection.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        connection.close()

def authenticate_credentials(username, password):
    connection = sqlite3.connect(DATABASE_FILE)
    db_cursor = connection.cursor()
    db_cursor.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
    record = db_cursor.fetchone()
    connection.close()
    if record:
        record_id, storage_hash = record
        if check_password_hash(storage_hash, password):
            return record_id
    return None

# ==========================================
# True Neural Network Model Load
# ==========================================
print('Loading Keras Deep Learning Engine...')
if not os.path.exists(WEIGHTS_FILE):
    print(f'Error: Neural network file not found at {WEIGHTS_FILE}.')
    neural_model = None
else:
    try:
        # Load the real trained MobileNet model natively using TensorFlow Keras
        neural_model = tf.keras.models.load_model(WEIGHTS_FILE, compile=False)
        print('MobileNet H5 Network loaded successfully!')
    except Exception as e:
        print(f'Error loading deep learning framework: {e}')
        neural_model = None

# ==========================================
# True Machine Learning Inference Engine
# ==========================================
def verify_file_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VALID_FORMATS

def execute_inference_pipeline(target_path):
    if neural_model is None:
        raise RuntimeError("Model engine is uninitialized or missing.")
        
    matrix_data = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    if matrix_data is None:
        raise ValueError('Targeted file stream cannot be parsed successfully.')
        
    # Apply standard preprocessing matrices matching model training parameters
    matrix_data = cv2.resize(matrix_data, (TARGET_WIDTH, TARGET_HEIGHT))
    matrix_data = cv2.equalizeHist(matrix_data)
    matrix_data = cv2.GaussianBlur(matrix_data, (3, 3), 0)
    
    # Normalize input data to 0.0 - 1.0 array range
    normalized_data = matrix_data.astype(np.float32) / 255.0
    
    # Reshape to match model input requirements: (1, 128, 128, 1)
    input_tensor = np.expand_dims(normalized_data, axis=-1)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    # Run the raw mathematical prediction pass through the model layers
    raw_predictions = neural_model.predict(input_tensor)
    
    # Extract the true prediction calculations
    predicted_index = int(np.argmax(raw_predictions, axis=1)[0])
    confidence_score = float(np.max(raw_predictions))
    
    return TARGET_LABELS[predicted_index], round(confidence_score, 4)

# ==========================================
# Application Routing Matrix
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_entry = request.form['username'].strip()
        pass_entry = request.form['password']
        if not user_entry or not pass_entry:
            flash('Please provide username and password')
            return redirect(url_for('register'))
        success = register_new_profile(user_entry, pass_entry)
        if not success:
            flash('Username already exists')
            return redirect(url_for('register'))
        flash('Registration successful. Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_entry = request.form['username'].strip()
        pass_entry = request.form['password']
        account_id = authenticate_credentials(user_entry, pass_entry)
        if account_id:
            session.permanent = True
            session['user_id'] = account_id
            session['username'] = user_entry
            flash('Login successful')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out')
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('username'):
        flash('Please login to upload images')
        return redirect(url_for('login'))

    if 'image' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
        
    file_stream = request.files['image']
    if file_stream.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file_stream and verify_file_extension(file_stream.filename):
        cleansed_name = secure_filename(file_stream.filename)
        destination_path = os.path.join(app.config['UPLOAD_FOLDER'], cleansed_name)
        file_stream.save(destination_path)
        
        try:
            predicted_tag, tracking_precision = execute_inference_pipeline(destination_path)
        except Exception as system_error:
            flash(f'Error processing image: {system_error}')
            return redirect(url_for('index'))

        render_url = f'uploads/{cleansed_name}'
        return render_template('result.html', pred_class=predicted_tag, confidence=tracking_precision, img_url=render_url)
    else:
        flash('File type not allowed')
        return redirect(url_for('index'))

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ==========================================
# Core Engine Launch
# ==========================================
if __name__ == '__main__':
    establish_database()
    app.run(debug=True)