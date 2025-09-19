from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import face_recognition
import numpy as np
import psycopg2
import psycopg2.extras
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
    'user': 'postgres',
    'password': 'alen',
    'database': 'erp',
    'port': 5432
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
            if isinstance(value, (datetime, date, timedelta, time)):
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
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
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
        
        # Roles table with role_name as primary key
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS roles (
            role_name VARCHAR(255) PRIMARY KEY,
            expected_check_in_time TIME NULL
        );
        """)
        
        # Updated employees table with role_name as foreign key
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id SERIAL PRIMARY KEY,
            employee_id VARCHAR(255) NOT NULL UNIQUE,
            name VARCHAR(255) NOT NULL,
            address TEXT NULL,
            date_of_birth DATE NULL,
            email VARCHAR(255) NULL,
            phone VARCHAR(20) NULL,
            basic_salary DECIMAL(10,2) NULL,
            face_encoding BYTEA NOT NULL,
            role_name VARCHAR(255) NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (role_name) REFERENCES roles(role_name) ON DELETE SET NULL
        );
        """)
        
        # Updated daily_attendance table with new status values
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_attendance (
            id SERIAL PRIMARY KEY,
            employee_id VARCHAR(255) NOT NULL,
            attendance_date DATE NOT NULL,
            check_in_time TIME NULL,
            check_out_time TIME NULL,
            check_in_timestamp TIMESTAMP NULL,
            check_out_timestamp TIMESTAMP NULL,
            total_hours DECIMAL(4,2) DEFAULT 0.00,
            status VARCHAR(20) DEFAULT 'absent' CHECK (status IN ('present', 'partial', 'absent', 'on_time', 'late')),
            notes TEXT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE,
            UNIQUE (employee_id, attendance_date)
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_log (
            id SERIAL PRIMARY KEY,
            employee_id VARCHAR(255) NOT NULL,
            event_type VARCHAR(20) NOT NULL CHECK (event_type IN ('check_in', 'check_out')),
            event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ip_address VARCHAR(45) NULL,
            user_agent TEXT NULL,
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
        );
        """)
        
        conn.commit()
        logger.info("Database tables for attendance system created or verified successfully.")
    except psycopg2.Error as e:
        logger.error(f"Error creating tables: {e}")
    finally:
        if conn and not conn.closed:
            cursor.close()
            conn.close()

class AttendanceSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_employee_ids = []
        self.known_names = []
        self.load_registered_employees()

    def load_registered_employees(self):
        conn = None
        try:
            conn = get_db_connection()
            if not conn: return
            cursor = conn.cursor()
            cursor.execute("SELECT employee_id, name, face_encoding FROM employees")
            results = cursor.fetchall()
            self.known_face_encodings = []
            self.known_employee_ids = []
            self.known_names = []
            for (employee_id, name, encoding_bytes) in results:
                try:
                    face_encoding = pickle.loads(encoding_bytes)
                    self.known_face_encodings.append(face_encoding)
                    self.known_employee_ids.append(employee_id)
                    self.known_names.append(name)
                except Exception as e:
                    logger.error(f"Error loading encoding for {employee_id}: {e}")
            logger.info(f"Loaded {len(self.known_employee_ids)} registered employees.")
        except psycopg2.Error as e:
            logger.error(f"Database error while loading employees: {e}")
        finally:
            if conn and not conn.closed:
                cursor.close()
                conn.close()

    def register_employee(self, employee_id, name, address, date_of_birth, email, phone, basic_salary, image_data, role_name=None):
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
            
            sql = """INSERT INTO employees (employee_id, name, address, date_of_birth, email, phone, basic_salary, face_encoding, role_name) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (employee_id, name, address, date_of_birth, email, phone, basic_salary, embedding_bytes, role_name))
            conn.commit()
            
            self.load_registered_employees()
            logger.info(f"Successfully registered employee {name} ({employee_id})")
            return {"success": True, "message": f"Employee {name} ({employee_id}) registered successfully."}
            
        except psycopg2.IntegrityError as e:
            return {"success": False, "message": f"Employee ID {employee_id} already exists."}
        except psycopg2.Error as e:
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
            if conn and not conn.closed:
                cursor.close()
                conn.close()

    def get_today_attendance(self, employee_id):
        conn = None
        try:
            conn = get_db_connection()
            if not conn: return None
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            today = date.today()
            cursor.execute("SELECT * FROM daily_attendance WHERE employee_id = %s AND attendance_date = %s", (employee_id, today))
            return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error getting today's attendance: {e}")
            return None
        finally:
            if conn and not conn.closed:
                cursor.close()
                conn.close()

    def calculate_total_hours(self, check_in_time, check_out_time):
        if not check_in_time or not check_out_time:
            return 0.00
        today = date.today()
        check_in_dt = datetime.combine(today, check_in_time)
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
                        name = self.known_names[best_match_index]
                        recognized_faces.append({'employee_id': employee_id, 'name': name})
            
            if not recognized_faces:
                return {"success": False, "message": "Wajah tidak dikenali."}
            
            recognized_employee = recognized_faces[0]
            emp_id = recognized_employee['employee_id']
            emp_name = recognized_employee['name']
            
            today_attendance = self.get_today_attendance(emp_id)
            today = date.today()
            current_time = datetime.now().time()
            current_timestamp = datetime.now()
            
            conn = get_db_connection()
            if not conn: 
                return {"success": False, "message": "Database connection failed."}
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if event_type == 'check_in':
                if today_attendance and today_attendance['check_in_time']:
                    return {"success": False, "message": f"{emp_name} sudah melakukan check-in hari ini pada {today_attendance['check_in_time']}"}
                
                # Get employee's role and expected check-in time
                cursor.execute("""SELECT r.expected_check_in_time 
                                 FROM employees e 
                                 JOIN roles r ON e.role_name = r.role_name 
                                 WHERE e.employee_id = %s""", (emp_id,))
                role_data = cursor.fetchone()
                
                status = 'partial' # Default status
                if role_data and role_data['expected_check_in_time']:
                    if current_time <= role_data['expected_check_in_time']:
                        status = 'on_time'
                    else:
                        status = 'late'

                if today_attendance:
                    cursor.execute("UPDATE daily_attendance SET check_in_time = %s, check_in_timestamp = %s, status = %s, updated_at = CURRENT_TIMESTAMP WHERE employee_id = %s AND attendance_date = %s", 
                                 (current_time, current_timestamp, status, emp_id, today))
                else:
                    cursor.execute("INSERT INTO daily_attendance (employee_id, attendance_date, check_in_time, check_in_timestamp, status) VALUES (%s, %s, %s, %s, %s)", 
                                 (emp_id, today, current_time, current_timestamp, status))
                
                message = f"Check-in berhasil: {emp_name} ({emp_id}) pada {current_time.strftime('%H:%M:%S')}"
                
            elif event_type == 'check_out':
                if not today_attendance or not today_attendance['check_in_time']:
                    return {"success": False, "message": f"{emp_name} belum melakukan check-in hari ini"}
                
                if today_attendance['check_out_time']:
                    return {"success": False, "message": f"{emp_name} sudah melakukan check-out hari ini pada {today_attendance['check_out_time']}"}
                
                total_hours = self.calculate_total_hours(today_attendance['check_in_time'], current_time)
                cursor.execute("UPDATE daily_attendance SET check_out_time = %s, check_out_timestamp = %s, total_hours = %s, status = 'present', updated_at = CURRENT_TIMESTAMP WHERE employee_id = %s AND attendance_date = %s", 
                             (current_time, current_timestamp, total_hours, emp_id, today))
                
                message = f"Check-out berhasil: {emp_name} ({emp_id}) pada {current_time.strftime('%H:%M:%S')}. Total jam kerja: {total_hours} jam"
            
            cursor.execute("INSERT INTO attendance_log (employee_id, event_type, ip_address, user_agent) VALUES (%s, %s, %s, %s)", 
                         (emp_id, event_type, ip_address, user_agent))
            conn.commit()
            
            logger.info(f"Attendance logged: {emp_name} ({emp_id}) - {event_type}")
            return {
                "success": True, 
                "message": message, 
                "employee_info": recognized_employee, 
                "attendance_info": {
                    "date": today.isoformat(), 
                    "event_type": event_type, 
                    "time": current_time.strftime('%H:%M:%S')
                }
            }
            
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
            if conn and not conn.closed:
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
        name = data.get('name')
        address = data.get('address')
        date_of_birth = data.get('date_of_birth')
        email = data.get('email')
        phone = data.get('phone')
        basic_salary = data.get('basic_salary')
        image_data = data.get('image')
        role_name = data.get('role_name')
        
        if not all([employee_id, name, image_data]):
            return jsonify({"success": False, "message": "employee_id, name, and image are required."}), 400
        
        result = system.register_employee(employee_id, name, address, date_of_birth, email, phone, basic_salary, image_data, role_name)
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
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("""SELECT e.id, e.employee_id, e.name, e.address, e.date_of_birth, e.email, e.phone, e.basic_salary, e.created_at, e.role_name 
                         FROM employees e ORDER BY e.name""")
        employees = serialize_db_records(cursor.fetchall())
        
        return jsonify({"success": True, "employees": employees})
        
    except Exception as e:
        logger.error(f"Error getting employees: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and not conn.closed:
            cursor.close()
            conn.close()

@app.route('/api/employees/<employee_id>', methods=['GET'])
def get_employee_details_endpoint(employee_id):
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("""SELECT e.id, e.employee_id, e.name, e.address, e.date_of_birth, e.email, e.phone, e.basic_salary, e.created_at, e.role_name 
                         FROM employees e WHERE e.employee_id = %s""", (employee_id,))
        employee = cursor.fetchone()
        
        if not employee:
            return jsonify({"success": False, "message": "Employee not found"}), 404
        
        return jsonify({"success": True, "employee": serialize_db_records([employee])[0]})
        
    except Exception as e:
        logger.error(f"Error getting employee details: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and not conn.closed:
            cursor.close()
            conn.close()

@app.route('/api/employees/<employee_id>', methods=['PUT'])
def update_employee_endpoint(employee_id):
    conn = None
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No JSON data provided."}), 400
        
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        
        cursor = conn.cursor()
        
        # Build dynamic update query
        update_fields = []
        update_values = []
        
        allowed_fields = ['name', 'address', 'date_of_birth', 'email', 'phone', 'basic_salary']
        for field in allowed_fields:
            if field in data:
                update_fields.append(f"{field} = %s")
                update_values.append(data[field])
        
        if not update_fields:
            return jsonify({"success": False, "message": "No valid fields to update"}), 400
        
        update_values.append(employee_id)
        query = f"UPDATE employees SET {', '.join(update_fields)} WHERE employee_id = %s"
        
        cursor.execute(query, update_values)
        
        if cursor.rowcount == 0:
            return jsonify({"success": False, "message": "Employee not found"}), 404
        
        conn.commit()
        system.load_registered_employees()  # Reload employees data
        
        return jsonify({"success": True, "message": "Employee updated successfully"})
        
    except Exception as e:
        logger.error(f"Error updating employee: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and not conn.closed:
            cursor.close()
            conn.close()

@app.route('/api/attendance/daily', methods=['GET'])
def get_daily_attendance_endpoint():
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        attendance_date = request.args.get('date', date.today().isoformat())
        
        query = """SELECT da.*, e.name 
                   FROM daily_attendance da 
                   JOIN employees e ON da.employee_id = e.employee_id 
                   WHERE da.attendance_date = %s 
                   ORDER BY e.name"""
        cursor.execute(query, (attendance_date,))
        attendance = serialize_db_records(cursor.fetchall())
        
        return jsonify({"success": True, "date": attendance_date, "attendance": attendance})
        
    except Exception as e:
        logger.error(f"Error getting daily attendance: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and not conn.closed:
            cursor.close()
            conn.close()

@app.route('/api/attendance/report', methods=['GET'])
def get_attendance_report_endpoint():
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        employee_id = request.args.get('employee_id')
        
        query = """SELECT da.*, e.name, e.basic_salary 
                   FROM daily_attendance da 
                   JOIN employees e ON da.employee_id = e.employee_id 
                   WHERE 1=1"""
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
        
        query += " ORDER BY da.attendance_date DESC, e.name"
        cursor.execute(query, params)
        report = serialize_db_records(cursor.fetchall())
        
        return jsonify({
            "success": True, 
            "filters": {"start_date": start_date, "end_date": end_date, "employee_id": employee_id}, 
            "report": report
        })
        
    except Exception as e:
        logger.error(f"Error getting attendance report: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and not conn.closed:
            cursor.close()
            conn.close()

@app.route('/api/attendance/log', methods=['GET'])
def get_attendance_log_endpoint():
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        query = """SELECT a.*, e.name 
                   FROM attendance_log a 
                   JOIN employees e ON a.employee_id = e.employee_id 
                   ORDER BY a.event_time DESC 
                   LIMIT %s OFFSET %s"""
        cursor.execute(query, (limit, offset))
        logs = serialize_db_records(cursor.fetchall())
        
        return jsonify({"success": True, "logs": logs, "limit": limit, "offset": offset})
        
    except Exception as e:
        logger.error(f"Error getting attendance log: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and not conn.closed:
            cursor.close()
            conn.close()

@app.route('/api/attendance/status/<employee_id>', methods=['GET'])
def get_employee_status_endpoint(employee_id):
    try:
        today_attendance = system.get_today_attendance(employee_id)
        
        if not today_attendance:
            status = {
                "employee_id": employee_id,
                "date": date.today().isoformat(),
                "status": "not_started",
                "can_check_in": True,
                "can_check_out": False,
                "message": "Belum melakukan absensi hari ini"
            }
        else:
            can_check_in = not today_attendance['check_in_time']
            can_check_out = today_attendance['check_in_time'] and not today_attendance['check_out_time']
            
            if today_attendance['status'] == 'present':
                message = "Absensi hari ini sudah lengkap"
            elif today_attendance['check_in_time'] and not today_attendance['check_out_time']:
                message = f"Sudah check-in pada {today_attendance['check_in_time']}, belum check-out"
            else:
                message = "Status absensi tidak normal"
            
            status = {
                "employee_id": employee_id,
                "date": str(today_attendance['attendance_date']),
                "status": today_attendance['status'],
                "check_in_time": str(today_attendance['check_in_time']) if today_attendance['check_in_time'] else None,
                "check_out_time": str(today_attendance['check_out_time']) if today_attendance['check_out_time'] else None,
                "total_hours": float(today_attendance['total_hours']) if today_attendance['total_hours'] else 0,
                "can_check_in": can_check_in,
                "can_check_out": can_check_out,
                "message": message
            }
        
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
            faces.append({
                'x': left,
                'y': top,
                'width': right - left,
                'height': bottom - top
            })
        
        return jsonify({
            "success": True,
            "faces": faces,
            "count": len(faces),
            "image_size": {"width": width, "height": height}
        })
        
    except ValueError as e:
        logger.error(f"Image processing error in face detection: {e}")
        return jsonify({"success": False, "message": f"Image processing error: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error in face detection endpoint: {e}")
        return jsonify({"success": False, "message": "Face detection error"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "success": True,
        "message": "Server is running",
        "registered_employees": len(system.known_employee_ids),
        "current_date": date.today().isoformat(),
        "current_time": datetime.now().strftime('%H:%M:%S')
    })

@app.route('/api/employees/<employee_id>', methods=['DELETE'])
def delete_employee_endpoint(employee_id):
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        
        cursor = conn.cursor()
        cursor.execute("DELETE FROM employees WHERE employee_id = %s", (employee_id,))
        
        if cursor.rowcount == 0:
            return jsonify({"success": False, "message": "Employee not found"}), 404
        
        conn.commit()
        system.load_registered_employees()  # Reload employees data
        
        return jsonify({"success": True, "message": "Employee deleted successfully"})
        
    except Exception as e:
        logger.error(f"Error deleting employee: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and not conn.closed:
            cursor.close()
            conn.close()

@app.route('/api/employees/search', methods=['GET'])
def search_employees_endpoint():
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        search_term = request.args.get('q', '').strip()
        
        if not search_term:
            return jsonify({"success": False, "message": "Search term is required"}), 400
        
        query = """SELECT id, employee_id, name, address, date_of_birth, email, phone, basic_salary, created_at 
                   FROM employees 
                   WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(employee_id) LIKE LOWER(%s) OR LOWER(email) LIKE LOWER(%s)
                   ORDER BY name"""
        search_pattern = f"%{search_term}%"
        cursor.execute(query, (search_pattern, search_pattern, search_pattern))
        employees = serialize_db_records(cursor.fetchall())
        
        return jsonify({"success": True, "employees": employees, "search_term": search_term})
        
    except Exception as e:
        logger.error(f"Error searching employees: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and not conn.closed:
            cursor.close()
            conn.close()

@app.route('/api/attendance/summary', methods=['GET'])
def get_attendance_summary_endpoint():
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get date range parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        employee_id = request.args.get('employee_id')
        
        # Base query for attendance summary
        query = """
        SELECT 
            e.employee_id,
            e.name,
            e.basic_salary,
            COUNT(CASE WHEN da.status = 'present' THEN 1 END) as total_present,
            COUNT(CASE WHEN da.status = 'partial' THEN 1 END) as total_partial,
            COUNT(CASE WHEN da.status = 'absent' THEN 1 END) as total_absent,
            COALESCE(SUM(da.total_hours), 0) as total_hours,
            COALESCE(AVG(da.total_hours), 0) as avg_hours_per_day
        FROM employees e
        LEFT JOIN daily_attendance da ON e.employee_id = da.employee_id
        WHERE 1=1
        """
        
        params = []
        if start_date:
            query += " AND (da.attendance_date IS NULL OR da.attendance_date >= %s)"
            params.append(start_date)
        if end_date:
            query += " AND (da.attendance_date IS NULL OR da.attendance_date <= %s)"
            params.append(end_date)
        if employee_id:
            query += " AND e.employee_id = %s"
            params.append(employee_id)
        
        query += " GROUP BY e.employee_id, e.name, e.basic_salary ORDER BY e.name"
        
        cursor.execute(query, params)
        summary = serialize_db_records(cursor.fetchall())
        
        return jsonify({
            "success": True,
            "filters": {"start_date": start_date, "end_date": end_date, "employee_id": employee_id},
            "summary": summary
        })
        
    except Exception as e:
        logger.error(f"Error getting attendance summary: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and not conn.closed:
            cursor.close()
            conn.close()

@app.route('/api/roles', methods=['POST'])
def create_role_endpoint():
    conn = None
    try:
        data = request.get_json()
        if not data or 'role_name' not in data:
            return jsonify({"success": False, "message": "role_name is required."}), 400
        
        role_name = data.get('role_name')
        expected_check_in_time = data.get('expected_check_in_time')
        
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        
        cursor = conn.cursor()
        cursor.execute("INSERT INTO roles (role_name, expected_check_in_time) VALUES (%s, %s)", (role_name, expected_check_in_time))
        conn.commit()
        
        return jsonify({"success": True, "message": f"Role '{role_name}' created successfully."}), 201
        
    except psycopg2.IntegrityError:
        return jsonify({"success": False, "message": "Role with this name already exists."}), 409
    except Exception as e:
        logger.error(f"Error creating role: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and not conn.closed:
            cursor.close()
            conn.close()

@app.route('/api/roles', methods=['GET'])
def get_roles_endpoint():
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("SELECT role_name, expected_check_in_time FROM roles ORDER BY role_name")
        roles = serialize_db_records(cursor.fetchall())
        
        return jsonify({"success": True, "roles": roles})
        
    except Exception as e:
        logger.error(f"Error getting roles: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and not conn.closed:
            cursor.close()
            conn.close()

@app.route('/api/roles/<role_name>', methods=['PUT'])
def update_role_endpoint(role_name):
    conn = None
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No JSON data provided."}), 400
        
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        
        cursor = conn.cursor()
        
        update_fields = []
        update_values = []
        
        if 'expected_check_in_time' in data:
            update_fields.append("expected_check_in_time = %s")
            update_values.append(data['expected_check_in_time'])
            
        if not update_fields:
            return jsonify({"success": False, "message": "No valid fields to update"}), 400
            
        update_values.append(role_name)
        query = f"UPDATE roles SET {', '.join(update_fields)} WHERE role_name = %s"
        
        cursor.execute(query, update_values)
        
        if cursor.rowcount == 0:
            return jsonify({"success": False, "message": "Role not found"}), 404
            
        conn.commit()
        
        return jsonify({"success": True, "message": "Role updated successfully"})
        
    except psycopg2.IntegrityError:
        return jsonify({"success": False, "message": "Role with this name already exists."}), 409
    except Exception as e:
        logger.error(f"Error updating role: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and not conn.closed:
            cursor.close()
            conn.close()

@app.route('/api/roles/<role_name>', methods=['DELETE'])
def delete_role_endpoint(role_name):
    conn = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"success": False, "message": "Database not available"}), 503
        
        cursor = conn.cursor()
        cursor.execute("DELETE FROM roles WHERE role_name = %s", (role_name,))
        
        if cursor.rowcount == 0:
            return jsonify({"success": False, "message": "Role not found"}), 404
            
        conn.commit()
        
        return jsonify({"success": True, "message": "Role deleted successfully"})
        
    except Exception as e:
        logger.error(f"Error deleting role: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500
    finally:
        if conn and not conn.closed:
            cursor.close()
            conn.close()

if __name__ == '__main__':
    logger.info("Starting Employee Attendance System...")
    create_tables_if_not_exist()
    socketio.run(app, host='0.0.0.0', port=5959)

if __name__ == '__main__':
    logger.info("Starting Employee Attendance System...")
    create_tables_if_not_exist()
    socketio.run(app, host='0.0.0.0', port=5959)