import os
import cv2
import pickle
from collections import deque
from flask import Flask, request, jsonify, send_from_directory
from twilio.rest import Client
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from model_utils import load_activity_model, predict_activity
from time import time

# ========== CONFIG ==========
load_dotenv()

MODEL_PATH = "activity_classifier_final.keras"
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE = (224, 224)
SEQ_LENGTH = 16          # number of frames in a sequence
CONFIDENCE_THRESHOLD = 0.7
ALERT_INTERVAL = 10      # seconds between consecutive alerts
FRAME_SKIP = 5           # process every 5th frame

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
TARGET_PHONE_NUMBER = os.getenv("TARGET_PHONE_NUMBER")

# Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.file']
CREDENTIALS_FILE = "client_secret_1011691015419-17focgg3dkgktfhjo942lp0m3il86anu.apps.googleusercontent.com.json"

# ========== INIT ==========
app = Flask(__name__, static_folder="static")
model, classes = load_activity_model(MODEL_PATH, CLASS_INDICES_PATH)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
last_alert_time = 0

# ========== GOOGLE DRIVE AUTH ==========
def google_drive_service():
    """Authenticate and return Google Drive service object"""
    creds = None
    if os.path.exists("token.pkl"):
        with open("token.pkl", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.pkl", "wb") as token:
            pickle.dump(creds, token)
    service = build("drive", "v3", credentials=creds)
    return service

def upload_to_drive(file_path):
    """Upload file to Google Drive and return public link"""
    service = google_drive_service()
    file_metadata = {"name": os.path.basename(file_path)}
    media = MediaFileUpload(file_path, mimetype="video/mp4")
    file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    service.permissions().create(fileId=file["id"], body={"role": "reader", "type": "anyone"}).execute()
    link = f"https://drive.google.com/file/d/{file['id']}/view?usp=sharing"
    return link

# ========== ALERT ==========
def send_alert(msg):
    """Send SMS/WhatsApp alert through Twilio"""
    global last_alert_time
    if time() - last_alert_time < ALERT_INTERVAL:
        return
    try:
        twilio_client.messages.create(
            body=msg,
            from_=TWILIO_PHONE_NUMBER,
            to=TARGET_PHONE_NUMBER
        )
        last_alert_time = time()
        print(f"[✓] Alert sent: {msg}")
    except Exception as e:
        print(f"[X] Failed to send alert: {e}")

# ========== ROUTES ==========
@app.route("/")
def serve_frontend():
    """Serve static frontend"""
    return send_from_directory(app.static_folder, "index.html")

@app.route("/upload_video", methods=["POST"])
def upload_video():
    """Handle uploaded video and run crime detection"""
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No video uploaded"}), 400

    os.makedirs("saved_clips", exist_ok=True)
    filepath = os.path.join("saved_clips", file.filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    frame_buffer = deque(maxlen=SEQ_LENGTH)
    all_predictions = []
    crime_detected = False
    crime_frame_count = 0
    out = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 encoder

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame_buffer.append(frame.copy())

        # run prediction only when sequence is ready
        if len(frame_buffer) == SEQ_LENGTH:
            pred_class, confidence = predict_activity(model, classes, list(frame_buffer))
            all_predictions.append((pred_class, confidence))

            # detect crime activities
            if pred_class.lower() in ['robbery', 'shoplifting', 'stealing'] and confidence > CONFIDENCE_THRESHOLD:
                crime_frame_count += 1
                if crime_frame_count >= 5 and not crime_detected:
                    crime_detected = True
                    out = cv2.VideoWriter("crime_clip.mp4", fourcc, 20.0,
                                          (frame.shape[1], frame.shape[0]))
                    send_alert(f"🚨 WARNING: Suspicious activity detected! Type: {pred_class.upper()} ⚠️")
                if crime_detected and out:
                    out.write(frame)
            else:
                if crime_frame_count > 0:
                    crime_frame_count = 0
                if crime_detected:
                    crime_detected = False
                    if out:
                        out.release()
                        link = upload_to_drive("crime_clip.mp4")
                        send_alert(f"🚨 CRITICAL ALERT: Crime confirmed! Watch the clip here:\n{link}")
                        out = None

    cap.release()
    if out:
        out.release()

    # return best prediction
    if all_predictions:
        best_pred = max(all_predictions, key=lambda x: x[1])
        pred_class, confidence = best_pred
    else:
        pred_class, confidence = "Unknown", 0.0

    return jsonify({
        "prediction": pred_class,
        "confidence": round(float(confidence), 2)
    })

# ========== MAIN ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
