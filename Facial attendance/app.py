import base64
import io
import os
import csv
import shutil
from datetime import datetime, date

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from flask_sqlalchemy import SQLAlchemy

from recognizer import FaceRecognizerService

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///attendance.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["DATASET_DIR"] = os.path.join(os.path.dirname(__file__), "dataset")
app.config["TRAINER_DIR"] = os.path.join(os.path.dirname(__file__), "trainer")

os.makedirs(app.config["DATASET_DIR"], exist_ok=True)
os.makedirs(app.config["TRAINER_DIR"], exist_ok=True)

db = SQLAlchemy(app)


class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    roll_no = db.Column(db.String(80), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class AttendanceSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    session_date = db.Column(db.Date, default=date.today)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("student.id"), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey("attendance_session.id"), nullable=False)
    present = db.Column(db.Boolean, default=True)
    recognized_at = db.Column(db.DateTime, default=datetime.utcnow)

    student = db.relationship("Student")
    session = db.relationship("AttendanceSession")


with app.app_context():
    db.create_all()


recognizer_service = FaceRecognizerService(
    dataset_dir=app.config["DATASET_DIR"],
    trainer_dir=app.config["TRAINER_DIR"],
)


def decode_base64_image(data_url: str):
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    return base64.b64decode(encoded)


@app.get("/")
def index():
    total_students = Student.query.count()
    total_sessions = AttendanceSession.query.count()
    return render_template("index.html", total_students=total_students, total_sessions=total_sessions)


@app.get("/train")
def train_page():
    return render_template("train.html")


@app.get("/students")
def students_page():
    students = Student.query.order_by(Student.created_at.desc()).all()
    return render_template("students.html", students=students)


@app.get("/students/<int:student_id>/edit")
def edit_student_page(student_id: int):
    student = Student.query.get_or_404(student_id)
    return render_template("edit_student.html", student=student)


@app.get("/register")
def register_page():
    return render_template("register.html")


@app.post("/api/students")
def api_create_student():
    payload = request.get_json(force=True)
    name = payload.get("name", "").strip()
    roll_no = payload.get("roll_no", "").strip()
    if not name or not roll_no:
        return jsonify({"error": "name and roll_no are required"}), 400

    # Check for duplicate roll number
    if Student.query.filter_by(roll_no=roll_no).first():
        return jsonify({"error": "roll_no already exists"}), 409
    
    # Check for duplicate name (case-insensitive)
    existing_name = Student.query.filter(Student.name.ilike(name)).first()
    if existing_name:
        return jsonify({"error": f"Student with name '{name}' already exists"}), 409

    student = Student(name=name, roll_no=roll_no)
    db.session.add(student)
    db.session.commit()

    os.makedirs(os.path.join(app.config["DATASET_DIR"], f"student_{student.id}"), exist_ok=True)
    return jsonify({"id": student.id, "name": student.name, "roll_no": student.roll_no})


@app.put("/api/students/<int:student_id>")
def api_update_student(student_id: int):
    student = Student.query.get_or_404(student_id)
    payload = request.get_json(force=True)
    name = payload.get("name", "").strip()
    roll_no = payload.get("roll_no", "").strip()
    
    if not name or not roll_no:
        return jsonify({"error": "name and roll_no are required"}), 400

    # Check if roll_no already exists for another student
    existing_roll = Student.query.filter_by(roll_no=roll_no).first()
    if existing_roll and existing_roll.id != student_id:
        return jsonify({"error": "roll_no already exists"}), 409
    
    # Check if name already exists for another student (case-insensitive)
    existing_name = Student.query.filter(Student.name.ilike(name), Student.id != student_id).first()
    if existing_name:
        return jsonify({"error": f"Student with name '{name}' already exists"}), 409

    student.name = name
    student.roll_no = roll_no
    db.session.commit()
    
    return jsonify({"id": student.id, "name": student.name, "roll_no": student.roll_no})


@app.delete("/api/students/<int:student_id>")
def api_delete_student(student_id: int):
    student = Student.query.get_or_404(student_id)
    Attendance.query.filter_by(student_id=student_id).delete()
    student_dir = os.path.join(app.config["DATASET_DIR"], f"student_{student_id}")
    if os.path.isdir(student_dir):
        shutil.rmtree(student_dir, ignore_errors=True)
    db.session.delete(student)
    db.session.commit()
    return jsonify({"deleted": True, "id": student_id})


@app.delete("/api/sessions/<int:session_id>")
def api_delete_session(session_id: int):
    session = AttendanceSession.query.get_or_404(session_id)
    Attendance.query.filter_by(session_id=session_id).delete()
    db.session.delete(session)
    db.session.commit()
    return jsonify({"deleted": True, "id": session_id})


@app.post("/api/cleanup-duplicates")
def api_cleanup_duplicates():
    """Remove duplicate students and clean up dataset"""
    duplicates_removed = 0
    
    # Find students with same names (case-insensitive)
    students = Student.query.all()
    seen_names = set()
    to_delete = []
    
    for student in students:
        name_lower = student.name.lower().strip()
        if name_lower in seen_names:
            to_delete.append(student.id)
        else:
            seen_names.add(name_lower)
    
    # Delete duplicates
    for student_id in to_delete:
        student = Student.query.get(student_id)
        if student:
            Attendance.query.filter_by(student_id=student_id).delete()
            student_dir = os.path.join(app.config["DATASET_DIR"], f"student_{student_id}")
            if os.path.isdir(student_dir):
                shutil.rmtree(student_dir, ignore_errors=True)
            db.session.delete(student)
            duplicates_removed += 1
    
    db.session.commit()
    
    # Clean up trainer file to force retraining
    if os.path.exists(app.config["TRAINER_DIR"]):
        shutil.rmtree(app.config["TRAINER_DIR"], ignore_errors=True)
        os.makedirs(app.config["TRAINER_DIR"], exist_ok=True)
    
    return jsonify({
        "duplicates_removed": duplicates_removed,
        "message": f"Removed {duplicates_removed} duplicate students. Please retrain the model."
    })


@app.post("/api/students/<int:student_id>/images")
def api_upload_student_image(student_id: int):
    Student.query.get_or_404(student_id)
    payload = request.get_json(force=True)
    images = payload.get("images")
    if isinstance(images, list):
        saved = 0
        for img_data in images:
            saved += _save_student_image(student_id, img_data)
        return jsonify({"saved": saved})
    elif isinstance(images, str):
        saved = _save_student_image(student_id, images)
        return jsonify({"saved": saved})
    else:
        return jsonify({"error": "images must be a data URL string or list of them"}), 400


def _save_student_image(student_id: int, data_url: str) -> int:
    raw = decode_base64_image(data_url)
    saved = recognizer_service.save_face_image(student_id, raw)
    return 1 if saved else 0


@app.post("/api/train")
def api_train():
    trained, num_faces, num_students = recognizer_service.train()
    status = {
        "trained": trained,
        "faces": num_faces,
        "students": num_students,
    }
    return jsonify(status)


@app.get("/sessions")
@app.post("/sessions")
def sessions_page():
    if request.method == "POST":
        name = request.form.get("name") or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        session = AttendanceSession(name=name)
        db.session.add(session)
        db.session.commit()
        return redirect(url_for("session_detail_page", session_id=session.id))

    sessions = AttendanceSession.query.order_by(AttendanceSession.created_at.desc()).all()
    return render_template("sessions.html", sessions=sessions)


@app.get("/sessions/<int:session_id>")
def session_detail_page(session_id: int):
    session = AttendanceSession.query.get_or_404(session_id)
    attendance_rows = Attendance.query.filter_by(session_id=session_id).all()
    return render_template("session_detail.html", session=session, attendance_rows=attendance_rows)


@app.post("/api/recognize/<int:session_id>")
def api_recognize(session_id: int):
    AttendanceSession.query.get_or_404(session_id)
    payload = request.get_json(force=True)
    data_url = payload.get("image")
    if not data_url:
        return jsonify({"error": "image is required"}), 400

    raw = decode_base64_image(data_url)
    result = recognizer_service.recognize(raw)

    recognized = result.get("recognized", [])
    saved_any = False
    for entry in recognized:
        student_id = entry.get("student_id")
        if student_id is None:
            continue
        existing = Attendance.query.filter_by(session_id=session_id, student_id=student_id).first()
        if existing is None:
            db.session.add(Attendance(student_id=student_id, session_id=session_id, present=True))
            saved_any = True
    if saved_any:
        db.session.commit()

    for entry in recognized:
        sid = entry.get("student_id")
        if sid is None:
            continue
        student = Student.query.get(sid)
        if student:
            entry["student"] = {"id": student.id, "name": student.name, "roll_no": student.roll_no}

    if result.get("student_id") is not None:
        sid = result["student_id"]
        student = Student.query.get(sid)
        if student:
            result["student"] = {"id": student.id, "name": student.name, "roll_no": student.roll_no}

    return jsonify(result)


@app.get("/export/<int:session_id>.csv")
def export_csv(session_id: int):
    session = AttendanceSession.query.get_or_404(session_id)
    rows = (
        db.session.query(Attendance, Student)
        .join(Student, Attendance.student_id == Student.id)
        .filter(Attendance.session_id == session_id)
        .all()
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Session", session.name])
    writer.writerow(["Date", session.session_date.isoformat()])
    writer.writerow([])
    writer.writerow(["Student ID", "Name", "Roll No", "Present", "Recognized At"])
    for attendance, student in rows:
        writer.writerow([
            student.id,
            student.name,
            student.roll_no,
            "Yes" if attendance.present else "No",
            attendance.recognized_at.strftime("%Y-%m-%d %H:%M:%S"),
        ])

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)

    filename = f"attendance_session_{session_id}.csv"
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


@app.context_processor
def inject_now():
    return {"now": datetime.utcnow()}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
