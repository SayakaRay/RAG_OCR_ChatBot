# main.py
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from io import BytesIO

# โหลด Environment Variables จากไฟล์ .env
load_dotenv()

# Import คลาสจากไฟล์โปรเจกต์ของคุณ
from storage import S3Storage
from image_manager import ImageManager

# --- ตั้งค่าการเชื่อมต่อ S3 ---
# อ่านค่าจาก Environment Variables
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY") # 💡 เปลี่ยนชื่อให้ตรงกับที่คุณมี
S3_SECRET_KEY = os.getenv("AWS_SECRET_KEY") # 💡 เปลี่ยนชื่อให้ตรงกับที่คุณมี
S3_BUCKET = os.getenv("AWS_BUCKET_NAME")   # 💡 เปลี่ยนชื่อให้ตรงกับที่คุณมี
# 💡 ลบบรรทัด S3_REGION ออก

# 💡 แก้ไขเงื่อนไขการตรวจสอบ
if not all([S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET]):
    raise ValueError("กรุณาตั้งค่า S3 environment variables ให้ครบถ้วน (AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_BUCKET_NAME)")

# สร้าง instance ของ S3Storage โดยไม่ส่ง region
s3_storage = S3Storage(
    access=S3_ACCESS_KEY,
    secret=S3_SECRET_KEY,
    bucket=S3_BUCKET
)

app = FastAPI(title="Hybrid RAG API")

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.get("/get-presigned-url/{image_id}")
def get_presigned_url(image_id: str, username: str, project_id: str):
    try:
        url = s3_storage.presigned_url(
            image_id=image_id,
            username=username,
            project_id=project_id,
            expires=3600
        )
        return {"url": url}
    except Exception as e:
        print(f"Error generating presigned URL for {image_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Image not found or error generating URL: {e}")

# TODO: เพิ่ม Endpoint สำหรับ /upload และ /search-hybrid ตามที่ ui.py เรียกใช้
# ตัวอย่าง:
@app.post("/upload")
def upload_document(username: str = Form(...), project_id: str = Form(...), file: UploadFile = File(...)):
    # ที่นี่คือส่วนที่คุณจะเรียกใช้ OCR Processor และ ImageManager
    content = file.file.read()
    results = process_uploaded_file(content, file.filename, s3_storage, username, project_id)
    return {"message": "Upload endpoint needs to be fully implemented."}

@app.get("/search-hybrid")
def search_hybrid(user_query: str, top_k: int, top_rerank: int, alpha: float):
    # ที่นี่คือส่วนที่คุณจะเรียกใช้ RAG logic ของคุณ
    return {"generated_answer": "Search endpoint needs to be fully implemented. ![Image](example_image_id.jpg)"}