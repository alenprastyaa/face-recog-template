from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import face_recognition
import numpy as np
import mysql.connector
import pickle
import base64
import io
import logging
import json
import sys
import requests
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('attendance.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'app_att',
    'port': 3306
}

FACE_RECOGNITION_TOLERANCE = 0.5

def fix_b64_padding(b64_string):
    try:
        b64_string = b64_string.strip()
        padding = len(b64_string) % 4
        if padding != 0:
            b64_string += '=' * (4 - padding)
        base64.b64decode(b64_string)
        return b64_string
    except Exception as e:
        logger.error(f"Base64 padding fix failed: {e}")
        raise ValueError("Unable to fix Base64 padding")

def serialize_db_records(records):
    serialized_list = []
    for record in records:
        serialized_record = {}
        for key, value in record.items():
            if isinstance(value, (datetime, date, timedelta)):
                serialized_record[key] = str(value)
            elif isinstance(value, Decimal):
                serialized_record[key] = float(value)
            else:
                serialized_record[key] = value
        serialized_list.append(serialized_record)
    return serialized_list

def download_image_from_url(url, timeout=10):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            raise ValueError(f"URL does not point to an image. Content-Type: {content_type}")
        return response.content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image from URL: {e}")
        raise ValueError(f"Failed to download image from URL: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error downloading image: {e}")
        raise ValueError(f"Failed to process image URL: {str(e)}")

def process_image_data(image_data):
    try:
        if image_data.startswith(('http://', 'https://')):
            logger.info(f"Processing image from URL: {image_data}")
            image_bytes = download_image_from_url(image_data)
        elif image_data.startswith('data:image'):
            if ',' in image_data:
                b64_data = image_data.split(',', 1)[1]
            else:
                raise ValueError("Invalid data URL format")
            padded_b64 = fix_b64_padding(b64_data)
            image_bytes = base64.b64decode(padded_b64)
        else:
            padded_b64 = fix_b64_padding(image_data)
            image_bytes = base64.b64decode(padded_b64)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)
        logger.info(f"Successfully processed image with shape: {image_array.shape}")
        return image_array
    except Exception as e:
        logger.error(f"Error processing image data: {e}")
        raise

def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as e:
        logger.error(f"Failed to get database connection: {e}")
        return None

def create_tables_if_not_exist():
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            logger.error("Cannot create tables: no database connection.")
            return
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INT AUTO_INCREMENT PRIMARY KEY,
            employee_id VARCHAR(255) NOT NULL UNIQUE,
            fullname VARCHAR(255) NOT NULL,
            face_encoding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            employee_id VARCHAR(255) NOT NULL,
            attendance_date DATE NOT NULL,
            check_in_time TIME NULL,
            check_out_time TIME NULL,
            check_in_timestamp TIMESTAMP NULL,
            check_out_timestamp TIMESTAMP NULL,
            total_hours DECIMAL(4,2) DEFAULT 0.00,
            status ENUM('present', 'partial', 'absent') DEFAULT 'absent',
            notes TEXT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE,
            UNIQUE KEY unique_employee_date (employee_id, attendance_date)
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            employee_id VARCHAR(255) NOT NULL,
            event_type ENUM('check_in', 'check_out') NOT NULL,
            event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ip_address VARCHAR(45) NULL,
            user_agent TEXT NULL,
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
        );
        """)
        conn.commit()
        logger.info("Database tables for attendance system created or verified successfully.")
    except mysql.connector.Error as e:
        logger.error(f"Error creating tables: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

class AttendanceSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_employee_ids = []
        self.known_fullnames = []
        self.load_registered_employees()

    def load_registered_employees(self):
        conn = None
        try:
            conn = get_db_connection()
            if not conn: return
            cursor = conn.cursor()
            cursor.execute("SELECT employee_id, fullname, face_encoding FROM employees")
            results = cursor.fetchall()
            self.known_face_encodings = []
            self.known_employee_ids = []
            self.known_fullnames = []
            for (employee_id, fullname, encoding_bytes) in results:
                try:
                    face_encoding = pickle.loads(encoding_bytes)
                    self.known_face_encodings.append(face_encoding)
                    self.known_employee_ids.append(employee_id)
                    self.known_fullnames.append(fullname)
                except Exception as e:
                    logger.error(f"Error loading encoding for {employee_id}: {e}")
            logger.info(f"Loaded {len(self.known_employee_ids)} registered employees.")
        except mysql.connector.Error as e:
            logger.error(f"Database error while loading employees: {e}")
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()

    def register_employee(self, employee_id, fullname, image_data):
        conn = None
        try:
            image = process_image_data(image_data)
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                return {"success": False, "message": "No face detected in the image."}
            if len(face_locations) > 1:
                return {"success": False, "message": "Multiple faces detected. Please use an image with only one face."}
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            embedding_bytes = pickle.dumps(face_encoding)
            conn = get_db_connection()
            if not conn: return {"success": False, "message": "Database connection failed."}
            cursor = conn.cursor()
            sql = "INSERT INTO employees (employee_id, fullname, face_encoding) VALUES (%s, %s, %s)"
            cursor.execute(sql, (employee_id, fullname, embedding_bytes))
            conn.commit()
            self.load_registered_employees()
            logger.info(f"Successfully registered employee {fullname} ({employee_id})")
            return {"success": True, "message": f"Employee {fullname} ({employee_id}) registered successfully."}
        except mysql.connector.Error as e:
            if e.errno == 1062:
                return {"success": False, "message": f"Employee ID {employee_id} already exists."}
            logger.error(f"Database error during registration: {e}")
            return {"success": False, "message": "A database error occurred."}
        except ValueError as e:
            logger.error(f"Image processing error: {e}")
            return {"success": False, "message": f"Image processing error: {str(e)}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error downloading image: {e}")
            return {"success": False, "message": "Failed to download image from URL. Please check the URL and try again."}
        except Exception as e:
            logger.error(f"Server error during registration: {e}")
            return {"success": False, "message": "A server error occurred."}
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()

    def get_today_attendance(self, employee_id):
        conn = None
        try:
            conn = get_db_connection()
            if not conn: return None
            cursor = conn.cursor(dictionary=True)
            today = date.today()
            cursor.execute("SELECT * FROM daily_attendance WHERE employee_id = %s AND attendance_date = %s", (employee_id, today))
            return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error getting today's attendance: {e}")
            return None
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()

    def calculate_total_hours(self, check_in_time, check_out_time):
        if not check_in_time or not check_out_time:
            return 0.00
        today = date.today()
        check_in_dt = datetime.combine(today, time()) + check_in_time
        check_out_dt = datetime.combine(today, check_out_time)
        if check_out_dt < check_in_dt:
            check_out_dt += timedelta(days=1)
        total_seconds = (check_out_dt - check_in_dt).total_seconds()
        total_hours = round(total_seconds / 3600, 2)
        return total_hours

    def recognize_and_log_attendance(self, image_data, event_type, ip_address=None, user_agent=None):
        conn = None
        try:
            image = process_image_data(image_data)
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                return {"success": False, "message": "No face detected."}
            face_encodings = face_recognition.face_encodings(image, face_locations)
            recognized_faces = []
            for face_encoding in face_encodings:
                if not self.known_face_encodings:
                    continue
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=FACE_RECOGNITION_TOLERANCE)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        employee_id = self.known_employee_ids[best_match_index]
                        fullname = self.known_fullnames[best_match_index]
                        recognized_faces.append({'employee_id': employee_id, 'fullname': fullname})
            if not recognized_faces:
                return {"success": False, "message": "Wajah tidak dikenali."}
            recognized_employee = recognized_faces[0]
            emp_id = recognized_employee['employee_id']
            emp_name = recognized_employee['fullname']
            today_attendance = self.get_today_attendance(emp_id)
            today = date.today()
            current_time = datetime.now().time()
            current_timestamp = datetime.now()
            conn = get_db_connection()
            if not conn: 
                return {"success": False, "message": "Database connection failed."}
            cursor = conn.cursor()
            if event_type == 'check_in':
                if today_attendance and today_attendance['check_in_time']:
                    return {"success": False, "message": f"{emp_name} sudah melakukan check-in hari ini pada {today_attendance['check_in_time']}"}
                if today_attendance:
                    cursor.execute("UPDATE daily_attendance SET check_in_time = %s, check_in_timestamp = %s, status = 'partial', updated_at = NOW() WHERE employee_id = %s AND attendance_date = %s", (current_time, current_timestamp, emp_id, today))
                else:
                    cursor.execute("INSERT INTO daily_attendance (employee_id, attendance_date, check_in_time, check_in_timestamp, status) VALUES (%s, %s, %s, %s, 'partial')", (emp_id, today, current_time, current_timestamp))
                message = f"Check-in berhasil: {emp_name} ({emp_id}) pada {current_time.strftime('%H:%M:%S')}"
            elif event_type == 'check_out':
                if not today_attendance or not today_attendance['check_in_time']:
                    return {"success": False, "message": f"{emp_name} belum melakukan check-in hari ini"}
                if today_attendance['check_out_time']:
                    return {"success": False, "message": f"{emp_name} sudah melakukan check-out hari ini pada {today_attendance['check_out_time']}"}
                total_hours = self.calculate_total_hours(today_attendance['check_in_time'], current_time)
                cursor.execute("UPDATE daily_attendance SET check_out_time = %s, check_out_timestamp = %s, total_hours = %s, status = 'present', updated_at = NOW() WHERE employee_id = %s AND attendance_date = %s", (current_time, current_timestamp, total_hours, emp_id, today))
                message = f"Check-out berhasil: {emp_name} ({emp_id}) pada {current_time.strftime('%H:%M:%S')}. Total jam kerja: {total_hours} jam"
            cursor.execute("INSERT INTO attendance_log (employee_id, event_type, ip_address, user_agent) VALUES (%s, %s, %s, %s)", (emp_id, event_type, ip_address, user_agent))
            conn.commit()
            logger.info(f"Attendance logged: {emp_name} ({emp_id}) - {event_type}")
            return {"success": True, "message": message, "employee_info": recognized_employee, "attendance_info": {"date": today.isoformat(), "event_type": event_type, "time": current_time.strftime('%H:%M:%S')}}
        except ValueError as e:
            logger.error(f"Image processing error: {e}")
            return {"success": False, "message": f"Image processing error: {str(e)}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error downloading image: {e}")
            return {"success": False, "message": "Failed to download image from URL. Please check the URL and try again."}
        except Exception as e:
            logger.error(f"Error during attendance logging: {e}")
            return {"success": False, "message": "An error occurred during attendance process."}
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()

system = AttendanceSystem()

@app.route('/api/register', methods=['POST'])
def register_employee_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No JSON data provided."}), 400
        employee_id = data.get('employee_id')
        fullname = data.get('fullname')
        image_data = data.get('image')
        if not all([employee_id, fullname, image_data]):
            return jsonify({"success": False, "message": "employee_id, fullname, and image are required."}), 400
        result = system.register_employee(employee_id, fullname, image_data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in register endpoint: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@app.route('/api/attendance', methods=['POST'])
def attendance_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No JSON data provided."}), 400
        image_data = data.get('image')
        event_type = data.get('event_type')
        if not image_data or event_type not in ['check_in', 'check_out']:
            return jsonify({"success": False, "message": "Image data and a valid event_type ('check_in' or 'check_out') are required."}), 400
        ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        user_agent = request.environ.get('HTTP_USER_AGENT', '')
        result = system.recognize_and_log_attendance(image_data, event_type, ip_address, user_agent)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in attendance endpoint: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@app.route('/api/employees', methods=['GET'])
def get_employees_endpoint():
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, employee_id, fullname, created_at FROM employees ORDER BY fullname")
        employees = serialize_db_records(cursor.fetchall())
        return jsonify({"success": True, "employees": employees})
    except Exception as e:
        logger.error(f"Error getting employees: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/api/attendance/daily', methods=['GET'])
def get_daily_attendance_endpoint():
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        cursor = conn.cursor(dictionary=True)
        attendance_date = request.args.get('date', date.today().isoformat())
        query = "SELECT da.*, e.fullname FROM daily_attendance da JOIN employees e ON da.employee_id = e.employee_id WHERE da.attendance_date = %s ORDER BY e.fullname"
        cursor.execute(query, (attendance_date,))
        attendance = serialize_db_records(cursor.fetchall())
        return jsonify({"success": True, "date": attendance_date, "attendance": attendance})
    except Exception as e:
        logger.error(f"Error getting daily attendance: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/api/attendance/report', methods=['GET'])
def get_attendance_report_endpoint():
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        cursor = conn.cursor(dictionary=True)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        employee_id = request.args.get('employee_id')
        query = "SELECT da.*, e.fullname FROM daily_attendance da JOIN employees e ON da.employee_id = e.employee_id WHERE 1=1"
        params = []
        if start_date:
            query += " AND da.attendance_date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND da.attendance_date <= %s"
            params.append(end_date)
        if employee_id:
            query += " AND da.employee_id = %s"
            params.append(employee_id)
        query += " ORDER BY da.attendance_date DESC, e.fullname"
        cursor.execute(query, params)
        report = serialize_db_records(cursor.fetchall())
        return jsonify({"success": True, "filters": {"start_date": start_date, "end_date": end_date, "employee_id": employee_id}, "report": report})
    except Exception as e:
        logger.error(f"Error getting attendance report: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/api/attendance/log', methods=['GET'])
def get_attendance_log_endpoint():
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        cursor = conn.cursor(dictionary=True)
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        query = "SELECT a.*, e.fullname FROM attendance_log a JOIN employees e ON a.employee_id = e.employee_id ORDER BY a.event_time DESC LIMIT %s OFFSET %s"
        cursor.execute(query, (limit, offset))
        logs = serialize_db_records(cursor.fetchall())
        return jsonify({"success": True, "logs": logs, "limit": limit, "offset": offset})
    except Exception as e:
        logger.error(f"Error getting attendance log: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/api/attendance/status/<employee_id>', methods=['GET'])
def get_employee_status_endpoint(employee_id):
    try:
        today_attendance = system.get_today_attendance(employee_id)
        if not today_attendance:
            status = {"employee_id": employee_id, "date": date.today().isoformat(), "status": "not_started", "can_check_in": True, "can_check_out": False, "message": "Belum melakukan absensi hari ini"}
        else:
            can_check_in = not today_attendance['check_in_time']
            can_check_out = today_attendance['check_in_time'] and not today_attendance['check_out_time']
            if today_attendance['status'] == 'present':
                message = "Absensi hari ini sudah lengkap"
            elif today_attendance['check_in_time'] and not today_attendance['check_out_time']:
                message = f"Sudah check-in pada {today_attendance['check_in_time']}, belum check-out"
            else:
                message = "Status absensi tidak normal"
            status = {"employee_id": employee_id, "date": str(today_attendance['attendance_date']), "status": today_attendance['status'], "check_in_time": str(today_attendance['check_in_time']) if today_attendance['check_in_time'] else None, "check_out_time": str(today_attendance['check_out_time']) if today_attendance['check_out_time'] else None, "total_hours": float(today_attendance['total_hours']) if today_attendance['total_hours'] else 0, "can_check_in": can_check_in, "can_check_out": can_check_out, "message": message}
        return jsonify({"success": True, "status": status})
    except Exception as e:
        logger.error(f"Error getting employee status: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@app.route('/api/detect-faces', methods=['POST'])
def detect_faces_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No JSON data provided."}), 400
        image_data = data.get('image')
        if not image_data:
            return jsonify({"success": False, "message": "Image data is required."}), 400
        image = process_image_data(image_data)
        face_locations = face_recognition.face_locations(image)
        faces = []
        height, width = image.shape[:2]
        for (top, right, bottom, left) in face_locations:
            faces.append({'x': left, 'y': top, 'width': right - left, 'height': bottom - top})
        return jsonify({"success": True, "faces": faces, "count": len(faces), "image_size": {"width": width, "height": height}})
    except ValueError as e:
        logger.error(f"Image processing error in face detection: {e}")
        return jsonify({"success": False, "message": f"Image processing error: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error in face detection endpoint: {e}")
        return jsonify({"success": False, "message": "Face detection error"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"success": True, "message": "Server is running", "registered_employees": len(system.known_employee_ids), "current_date": date.today().isoformat(), "current_time": datetime.now().strftime('%H:%M:%S')})

if __name__ == '__main__':
    logger.info("Starting Employee Attendance System...")
    create_tables_if_not_exist()
    socketio.run(app, host='0.0.0.0', port=5959)