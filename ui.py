import gradio as gr
import requests
import re
import base64
from io import BytesIO

# ฟังก์ชันเรียก API จริง
def chat_with_api(message, history):
    try:
        # เตรียมพารามิเตอร์สำหรับ API
        params = {
            "user_query": message,
            "top_k": 10,
            "top_rerank": 3,
            "alpha": 0.7
        }
        
        # เรียก API ผ่าน GET request
        response = requests.get(
            "http://localhost:8000/generate-hybrid",
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            # ถ้า API ตอบกลับเป็น JSON
            try:
                result = response.json()
                # ดึงเฉพาะ generated_answer
                reply = result.get("generated_answer", str(result))
                
                # ตรวจหาและดึงรูปภาพจาก S3
                reply_with_images = process_images_in_text(reply)
                
            except:
                # ถ้าไม่ใช่ JSON ให้ใช้ text
                reply_with_images = response.text
        else:
            reply_with_images = f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        reply_with_images = "ไม่สามารถเชื่อมต่อกับ API ได้ กรุณาตรวจสอบว่า API server ทำงานอยู่ที่ port 8000"
    except requests.exceptions.Timeout:
        reply_with_images = "API ใช้เวลานานเกินไป กรุณาลองใหม่อีกครั้ง"
    except Exception as e:
        reply_with_images = f"เกิดข้อผิดพลาด: {str(e)}"
    
    # เพิ่มข้อความและตอบกลับใน history
    history.append([message, reply_with_images])
    return "", history

def process_images_in_text(text):
    """ประมวลผลข้อความและแทนที่ลิงก์รูปภาพด้วยรูปจริงจาก S3"""
    # หา pattern รูปภาพในข้อความ เช่น [Image: filename.jpg](filename.jpg)
    image_pattern = r'\[Image:\s*([^\]]+)\]\(([^)]+)\)'
    
    def replace_image(match):
        image_description = match.group(1)
        image_filename = match.group(2)
        
        try:
            # ดึงรูปภาพจาก S3 ผ่าน API
            image_url = get_image_from_s3(image_filename)
            if image_url:
                # ใช้ HTML img tag สำหรับแสดงรูป
                return f'\n\n<img src="{image_url}" alt="{image_description}" style="max-width: 400px; max-height: 300px; border-radius: 8px; margin: 10px 0;"/>\n\n'
            else:
                return f"\n\n🖼️ รูปภาพ: {image_description} (ไม่สามารถโหลดได้)\n\n"
        except Exception as e:
            return f"\n\n🖼️ รูปภาพ: {image_description} (Error: {str(e)})\n\n"
    
    # แทนที่ pattern รูปภาพทั้งหมด
    processed_text = re.sub(image_pattern, replace_image, text)
    return processed_text

def get_image_from_s3(image_filename):
    """ดึงรูปภาพจาก S3 ผ่าน API endpoint แบบ binary"""
    try:
        # ลบส่วน extension และ path หากมี
        filename = image_filename.split('/')[-1]  # เอาเฉพาะชื่อไฟล์
        filename = filename.rsplit('.', 1)[0]     # ลบ extension
        
        # เรียก API เพื่อขอรูปภาพเป็น binary
        response = requests.get(
            f"http://localhost:8000/get-image-binary/{filename}",
            timeout=30
        )
        
        if response.status_code == 200:
            # แปลง binary เป็น base64 สำหรับแสดงใน HTML
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
        else:
            print(f"Failed to get image binary for {filename}: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error getting image from S3: {e}")
        return None

# สร้าง UI chatbot
with gr.Blocks(title="💬 Hybrid API Chatbot") as demo:
    gr.Markdown("# 💬 Hybrid API Chatbot")
    gr.Markdown("เชื่อมต่อกับ API Hybrid Search บน localhost:8000")
    
    chatbot = gr.Chatbot(height=400, type="tuples")
    msg = gr.Textbox(label="ส่งข้อความ", placeholder="พิมพ์คำถามของคุณที่นี่...")
    clear = gr.Button("เคลียร์ประวัติ")
    
    # สร้างตัวแปร state สำหรับเก็บประวัติการสนทนา
    history_state = gr.State([])
    
    def respond(message, history):
        return chat_with_api(message, history)
    
    msg.submit(respond, [msg, history_state], [msg, chatbot])
    clear.click(lambda: ([], []), None, [history_state, chatbot])

demo.launch(debug=True, share=False)
