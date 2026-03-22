import torch
import numpy as np
import cv2
import time
from datetime import datetime, timedelta
import os
import sys
import threading
import pickle
import csv
import random
import bcrypt
from flask import Flask, render_template, Response, jsonify, request, send_from_directory, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
from deepface import DeepFace
from werkzeug.utils import secure_filename

# --- Add yolov7 repository to path ---
YOLOV7_PATH = os.path.join(os.getcwd(), 'yolov7')
if YOLOV7_PATH not in sys.path:
    sys.path.append(YOLOV7_PATH)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox

# --- Flask App Initialization ---
app = Flask(__name__)

# ==============================================================================
# --- 1. CAMERA CONFIGURATION ---
# ==============================================================================
PHONE_CAMERA_URLS = [
    # "http://172.20.10.4:8080/video",
    # "http://172.20.10.2:8080/video",
]

# ==============================================================================
# --- 2. AUTHENTICATION & DATABASE CONFIGURATION ---
# ==============================================================================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = os.urandom(24) 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Email Configuration for OTP ---
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'joyanto2203@gmail.com' 
app.config['MAIL_PASSWORD'] = 'orlj dgob vhhf mkhq'

# --- Upload Configuration ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ==============================================================================

db = SQLAlchemy(app)
mail = Mail(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' 
login_manager.login_message_category = 'info'

# --- User Database Model ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    is_verified = db.Column(db.Boolean, default=False)
    otp = db.Column(db.String(6), nullable=True)
    otp_timestamp = db.Column(db.DateTime, nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Global variable for the selected camera source ---
CAMERA_SOURCE = None
camera_lock = threading.Lock()

# --- Other Global variables ---
APP_START_TIME = datetime.now()
device = None
yolo_model = None
face_database = {}
class_names = []
stride = 0
colors = []

last_frame_lock = threading.Lock()
last_frame = None

PERSON_LOG_FILE = "person_log.csv"
OBJECT_LOG_FILE = "object_log.csv"
log_lock = threading.Lock() 

UNKNOWN_SNAPSHOTS_DIR = "unknown_snapshots"
FACE_DATASET_DIR = "face_dataset"
os.makedirs(UNKNOWN_SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(FACE_DATASET_DIR, exist_ok=True)

selection_lock = threading.Lock()
face_db_lock = threading.Lock()

TARGET_CLASSES_INDICES = []
objects_in_frame = {}
DETECT_ALL = False

def setup_log_files():
    with log_lock:
        for log_file, headers in [
            (PERSON_LOG_FILE, ['Timestamp', 'Event', 'Name', 'Confidence']),
            (OBJECT_LOG_FILE, ['Timestamp', 'Event', 'Object', 'Confidence'])
        ]:
            if os.path.exists(log_file): os.remove(log_file)
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    print("✅ Log files initialized for new session.")

def load_models():
    global device, yolo_model, class_names, stride, colors, face_database
    
    device = select_device('')
    yolo_model = attempt_load("weights/yolov7.pt", map_location=device)
    stride = int(yolo_model.stride.max())
    class_names = yolo_model.module.names if hasattr(yolo_model, 'module') else yolo_model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in class_names]
    print("✅ YOLOv7 model loaded successfully.")

    if os.path.exists("face_database.pkl"):
        with open("face_database.pkl", "rb") as f:
            face_database = pickle.load(f)
        print(f"✅ Face database loaded successfully with {len(face_database)} person(s).")
    else:
        print("⚠️ WARNING: Face database not found. Specific person recognition will be disabled.")

def log_event_to_csv(log_file, event_data):
    try:
        with log_lock:
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [
                    event_data.get('timestamp').strftime("%Y-%m-%d %H:%M:%S"),
                    event_data.get('event'),
                    event_data.get('name') if 'name' in event_data else event_data.get('object'),
                    f"{event_data.get('confidence', ''):.2f}" if 'confidence' in event_data else 'N/A'
                ]
                writer.writerow(row)
    except Exception as e:
        print(f"Error logging to CSV {log_file}: {e}")

def recognize_face(face_image):
    try:
        embedding_objs = DeepFace.represent(img_path=face_image, model_name="VGG-Face", enforce_detection=False)
        embedding = embedding_objs[0]["embedding"]
        best_match_name = None
        min_dist = 0.6 
        
        with face_db_lock:
            for name, known_embeddings in face_database.items():
                for known_embedding in known_embeddings:
                    result = DeepFace.verify(embedding, known_embedding, model_name="VGG-Face", enforce_detection=False, distance_metric='cosine')
                    if result["verified"] and result["distance"] < min_dist:
                        min_dist = result["distance"]
                        best_match_name = name
        return best_match_name
    except Exception:
        return None

def generate_frames():
    global objects_in_frame, last_frame
    
    with camera_lock:
        source = CAMERA_SOURCE
    
    if source is None:
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Please select a camera source", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        while True:
            with camera_lock:
                if CAMERA_SOURCE is not None: return
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1)

    try: cap_source = int(source)
    except ValueError: cap_source = source

    cap = cv2.VideoCapture(cap_source)
    
    while True:
        with camera_lock:
            if CAMERA_SOURCE is None: break
        
        ret, frame = cap.read()
        if not ret: break

        try: frame = cv2.resize(frame, (640, 480))
        except: continue

        with last_frame_lock:
            last_frame = frame.copy()
        
        current_target_indices = []
        is_detecting_all = False
        with selection_lock:
            current_target_indices = list(TARGET_CLASSES_INDICES)
            is_detecting_all = DETECT_ALL
        
        final_frame = frame.copy()

        if is_detecting_all or (current_target_indices and class_names):
            img = letterbox(frame, 640, stride=stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device).float() / 255.0
            if img.ndimension() == 3: img = img.unsqueeze(0)
            
            with torch.no_grad():
                pred = yolo_model(img, augment=False)[0]
            
            nms_classes = None if is_detecting_all else current_target_indices
            pred = non_max_suppression(pred, 0.4, 0.5, classes=nms_classes, agnostic=False)

            objects_detected_this_frame = set()
            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = map(int, xyxy)
                        object_name = class_names[int(cls)]
                        
                        if object_name == 'person':
                            person_crop = frame[y1:y2, x1:x2]
                            recognized_name = recognize_face(person_crop)
                            final_name = recognized_name if recognized_name else "Unknown"
                            
                            objects_detected_this_frame.add(final_name)
                            if final_name not in objects_in_frame:
                                objects_in_frame[final_name] = True
                                log_event_to_csv(PERSON_LOG_FILE, {"timestamp": datetime.now(), "event": "Entry", "name": final_name, "confidence": float(conf)})
                            
                            label = f'{final_name} {conf:.2f}'
                            color = (0, 255, 0) if recognized_name else (0, 0, 255)
                            cv2.rectangle(final_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(final_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        else:
                            objects_detected_this_frame.add(object_name)
                            if object_name not in objects_in_frame:
                                objects_in_frame[object_name] = True
                                log_event_to_csv(OBJECT_LOG_FILE, {"timestamp": datetime.now(), "event": "Entry", "object": object_name, "confidence": float(conf)})

                            label = f'{object_name} {conf:.2f}'
                            cv2.rectangle(final_frame, (x1, y1), (x2, y2), colors[int(cls)], 2)
                            cv2.putText(final_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[int(cls)], 2)

            exited_objects = set(objects_in_frame.keys()) - objects_detected_this_frame
            for obj in exited_objects:
                if obj in face_database or obj == "Unknown":
                    log_event_to_csv(PERSON_LOG_FILE, {"timestamp": datetime.now(), "event": "Exit", "name": obj})
                else:
                    log_event_to_csv(OBJECT_LOG_FILE, {"timestamp": datetime.now(), "event": "Exit", "object": obj})
                del objects_in_frame[obj]
        
        ret, buffer = cv2.imencode('.jpg', final_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()

def generate_camera_preview(source):
    try: cap_source = int(source)
    except ValueError: cap_source = source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        error_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Cannot Open", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        while True:
            ret, buffer = cv2.imencode('.jpg', error_frame)
            if not ret: continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)
    while True:
        ret, frame = cap.read()
        if not ret: break
        try: frame = cv2.resize(frame, (320, 240))
        except: continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# --- Routes ---
@app.route('/')
def home():
    if current_user.is_authenticated: return redirect(url_for('select_camera_page'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('select_camera_page'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash):
            if user.is_verified:
                login_user(user, remember=False)
                return redirect(url_for('select_camera_page'))
            else:
                flash('Please check email for OTP.', 'warning')
                return redirect(url_for('verify_otp', email=email))
        flash('Login failed.', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(email=email).first():
            flash('Email exists.', 'danger')
            return redirect(url_for('signup'))
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        otp = str(random.randint(100000, 999999))
        new_user = User(email=email, password_hash=hashed, otp=otp, otp_timestamp=datetime.now())
        db.session.add(new_user)
        db.session.commit()
        try:
            msg = Message('Verification Code', sender=app.config['MAIL_USERNAME'], recipients=[email])
            msg.body = f'Code: {otp}'
            mail.send(msg)
            return redirect(url_for('verify_otp', email=email))
        except Exception:
            flash('Email failed.', 'danger')
    return render_template('signup.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    email = request.args.get('email')
    if request.method == 'POST':
        user = User.query.filter_by(email=email).first()
        if user and user.otp == request.form.get('otp'):
            user.is_verified = True
            db.session.commit()
            return redirect(url_for('login'))
        flash('Invalid OTP.', 'danger')
    return render_template('verify_otp.html', email=email)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/select_camera')
@login_required
def select_camera_page(): return render_template('select_camera.html')

@app.route('/dashboard')
@login_required
def dashboard(): return render_template('index_interactive_enrollment.html')

@app.route('/video_feed')
@login_required
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def read_logs_from_csv(log_file):
    logs = []
    try:
        with log_lock:
            with open(log_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if datetime.strptime(row['Timestamp'], "%Y-%m-%d %H:%M:%S") >= APP_START_TIME:
                            logs.append(dict(row))
                    except: continue
        logs.reverse()
        return logs
    except: return []

@app.route('/api/get_person_logs')
@login_required
def get_person_logs(): return jsonify(read_logs_from_csv(PERSON_LOG_FILE))

@app.route('/api/get_object_logs')
@login_required
def get_object_logs(): return jsonify(read_logs_from_csv(OBJECT_LOG_FILE))

@app.route('/get_objects', methods=['GET'])
@login_required
def get_objects(): return jsonify(class_names if yolo_model else [])

@app.route('/update_objects', methods=['POST'])
@login_required
def update_objects():
    global TARGET_CLASSES_INDICES, objects_in_frame, DETECT_ALL
    data = request.get_json()
    selected_names = data.get('objects', [])
    with selection_lock:
        if 'all' in selected_names:
            DETECT_ALL = True
            TARGET_CLASSES_INDICES = []
        else:
            DETECT_ALL = False
            TARGET_CLASSES_INDICES = [class_names.index(name) for name in selected_names if name in class_names] if class_names else []
            
            current = set(objects_in_frame.keys())
            for obj in current:
                is_person = (obj in face_database) or (obj == "Unknown") or (obj == "person")
                if (is_person and 'person' not in selected_names) or (not is_person and obj not in selected_names):
                    log_file = PERSON_LOG_FILE if is_person else OBJECT_LOG_FILE
                    log_event_to_csv(log_file, {"timestamp": datetime.now(), "event": "Exit", "name" if is_person else "object": obj})
                    del objects_in_frame[obj]
    return jsonify({'status': 'success'})

@app.route('/api/capture_frame', methods=['POST'])
@login_required
def capture_frame():
    global last_frame
    with last_frame_lock:
        if last_frame is None: return jsonify({"status": "error"}), 500
        try:
            face_objs = DeepFace.extract_faces(last_frame, detector_backend='mtcnn', enforce_detection=True)
            face_crop_float = face_objs[0]['face']
            face_crop_to_save = (face_crop_float * 255).astype(np.uint8)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            snapshot_filename = f"{timestamp_str}.jpg"
            snapshot_path = os.path.join(UNKNOWN_SNAPSHOTS_DIR, snapshot_filename)
            # Save as BGR (OpenCV default) which is what DeepFace expects from file path usually
            face_crop_bgr = cv2.cvtColor(face_crop_to_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(snapshot_path, face_crop_bgr)
            return jsonify({"status": "success", "filename": snapshot_filename})
        except: return jsonify({"status": "error"}), 400

@app.route('/unknown_snapshots/<filename>')
@login_required
def serve_unknown_snapshot(filename):
    return send_from_directory(UNKNOWN_SNAPSHOTS_DIR, filename)

@app.route('/api/enroll_new_person', methods=['POST'])
@login_required
def enroll_new_person():
    data = request.get_json()
    name = data.get('name')
    filename = data.get('filename')
    if not name or not filename:
        return jsonify({"status": "error", "message": "Name and filename are required."}), 400
    snapshot_path = os.path.join(UNKNOWN_SNAPSHOTS_DIR, filename)
    if not os.path.exists(snapshot_path):
        return jsonify({"status": "error", "message": "Snapshot file not found."}), 404
    try:
        embedding_objs = DeepFace.represent(img_path=snapshot_path, model_name="VGG-Face", enforce_detection=True)
        new_embedding = embedding_objs[0]["embedding"]
        with selection_lock:
            if name in face_database:
                face_database[name].append(new_embedding)
            else:
                face_database[name] = [new_embedding]
        with open("face_database.pkl", "wb") as f:
            pickle.dump(face_database, f)
        person_dataset_path = os.path.join(FACE_DATASET_DIR, name)
        os.makedirs(person_dataset_path, exist_ok=True)
        os.replace(snapshot_path, os.path.join(person_dataset_path, filename))
        print(f"✅ Successfully enrolled new person: {name}")
        return jsonify({"status": "success", "message": f"{name} has been enrolled."})
    except Exception as e:
        print(f"❌ Error during enrollment: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/attendance')
@login_required
def attendance_page():
    return render_template('attendance.html')

@app.route('/api/get_attendance_report')
@login_required
def get_attendance_report():
    attendance_report = []
    names_processed = set()
    today_str = datetime.now().strftime('%Y-%m-%d')
    try:
        with log_lock:
            if not os.path.exists(PERSON_LOG_FILE): return jsonify([])
            with open(PERSON_LOG_FILE, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (row.get('Event') == 'Entry' and
                        row.get('Name') != 'person' and 
                        row.get('Name') != 'Unknown' and
                        row.get('Name') not in names_processed and
                        row.get('Timestamp', '').startswith(today_str)):
                        attendance_report.append(dict(row))
                        names_processed.add(row.get('Name'))
        attendance_report.sort(key=lambda x: x['Timestamp'])
        return jsonify(attendance_report)
    except Exception as e:
        print(f"Error reading attendance report: {e}")
        return jsonify([])

@app.route('/api/list_cameras')
@login_required
def list_cameras():
    sources = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            sources.append({"id": str(i), "name": f"Webcam {i}"})
            cap.release()
        else:
            break
    for i, url in enumerate(PHONE_CAMERA_URLS):
        if "YOUR_" not in url:
            sources.append({"id": url, "name": f"Phone Camera {i+1}"})
    return jsonify(sources)

@app.route('/camera_preview/<path:source>')
@login_required
def camera_preview(source):
    return Response(generate_camera_preview(source), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/select_camera', methods=['POST'])
@login_required
def select_camera():
    global CAMERA_SOURCE
    data = request.get_json()
    source = data.get('source')
    if source is not None:
        with camera_lock:
            CAMERA_SOURCE = source
        print(f"✅ Camera source selected: {source}")
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "No source provided"}), 400

@app.route('/api/reset_camera', methods=['POST'])
@login_required
def reset_camera():
    global CAMERA_SOURCE
    with camera_lock:
        CAMERA_SOURCE = None
    print("🔄 Camera source reset. Ready for new selection.")
    return jsonify({"status": "success"})

# --- NEW: Upload and Analysis Routes ---
@app.route('/api/upload_video', methods=['POST'])
@login_required
def upload_video():
    if 'video' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['video']
    if file.filename == '': return jsonify({'error': 'No filename'}), 400
    filename = secure_filename(f"{int(time.time())}_{file.filename}")
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify({'status': 'success', 'filename': filename})

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/video_analysis/<filename>')
@login_required
def video_analysis_page(filename):
    return render_template('video_analysis.html', filename=filename)

@app.route('/api/analyze_video_file', methods=['POST'])
@login_required
def analyze_video_file():
    data = request.get_json()
    filename = data.get('filename')
    target_names = data.get('targets', [])
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath): return jsonify({"error": "File not found"}), 404
    
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps) if fps > 0 else 30 # Process 1 frame per second to save time
    
    detect_all = 'all' in target_names
    target_indices = None
    if not detect_all:
        target_indices = [class_names.index(name) for name in target_names if name in class_names]
        if not target_indices: return jsonify([]) # No valid targets

    results = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            img = letterbox(frame, 640, stride=stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device).float() / 255.0
            if img.ndimension() == 3: img = img.unsqueeze(0)
            
            with torch.no_grad(): pred = yolo_model(img, augment=False)[0]
            pred = non_max_suppression(pred, 0.4, 0.5, classes=target_indices, agnostic=False)
            
            for i, det in enumerate(pred):
                if len(det):
                    # IMPORTANT: Rescale coords to original frame for cropping
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                    
                    for *xyxy, conf, cls in reversed(det):
                        obj_name = class_names[int(cls)]
                        display_name = obj_name
                        
                        # --- INTEGRATE FACE RECOGNITION HERE ---
                        if obj_name == 'person':
                            x1, y1, x2, y2 = map(int, xyxy)
                            person_crop = frame[y1:y2, x1:x2]
                            
                            # Reuse the same recognition function
                            recognized_name = recognize_face(person_crop)
                            if recognized_name:
                                display_name = recognized_name
                            else:
                                display_name = "person" # or "Unknown"

                        results.append({
                            "timestamp": round(timestamp, 2),
                            "object": display_name,
                            "confidence": float(conf)
                        })
        frame_count += 1
    
    cap.release()
    return jsonify(results)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    try:
        file_path = os.path.join(YOLOV7_PATH, 'models', 'experimental.py')
        with open(file_path, 'r') as f: content = f.read()
        if "weights_only=False" not in content:
            new_content = content.replace("ckpt = torch.load(w, map_location=map_location)", "ckpt = torch.load(w, map_location=map_location, weights_only=False)")
            with open(file_path, 'w') as f: f.write(new_content)
            print("✅ Applied compatibility patch to YOLOv7 experimental.py")
    except Exception as e:
        print(f"⚠️ Could not apply YOLOv7 patch: {e}")

    setup_log_files()
    load_models()
    app.run(debug=True, threaded=True, use_reloader=False)