# ui.py

import gradio as gr
import requests
import re
import base64
from io import BytesIO

DEFAULT_USERNAME = "test_user"
DEFAULT_PROJECT = "test_project"

def get_image_url_from_api(image_filename, username, project_id):
    try:
        image_id = image_filename.split('/')[-1].split('.')[0]
        url = f"http://localhost:8000/get-presigned-url/{image_id}"
        params = {"username": username, "project_id": project_id}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json().get("url")
        
        print(f"API Error fetching URL for {image_id}: {response.status_code} - {response.text}")
        return None
    except Exception as e:
        print(f"Error fetching presigned URL for {image_filename}: {e}")
        return None

def process_images_in_text(text, username, project_id):
    image_pattern = r'!\[.*?\]\((.*?)\)'

    def replace_image(match):
        image_filename = match.group(1)
        try:
            presigned_url = get_image_url_from_api(image_filename, username, project_id)
            
            if presigned_url:
                return f'<img src="{presigned_url}" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0;"/>'
            else:
                return f"\n\n🖼️ ไม่สามารถโหลดรูปภาพ: {image_filename}\n\n"
        except Exception as e:
            return f"\n\n🖼️ รูปภาพผิดพลาด: {str(e)}\n\n"

    return re.sub(image_pattern, replace_image, text)

def chat_with_api(message, history, username, project_id):
    try:
        params = {
            "user_query": message,
            "top_k": 10,
            "top_rerank": 3,
            "alpha": 0.7
        }
        response = requests.get("http://localhost:8000/search-hybrid", params=params, timeout=30)

        if response.status_code == 200:
            result = response.json()
            reply = result.get("generated_answer", str(result))
            reply_with_images = process_images_in_text(reply, username, project_id)
        else:
            reply_with_images = f"API Error: {response.status_code} - {response.text}"

    except requests.exceptions.ConnectionError:
        reply_with_images = "❌ ไม่สามารถเชื่อมต่อกับ API ได้ กรุณาตรวจสอบว่า API server ทำงานอยู่ที่ port 8000"
    except requests.exceptions.Timeout:
        reply_with_images = "⌛ API ใช้เวลานานเกินไป กรุณาลองใหม่อีกครั้ง"
    except Exception as e:
        reply_with_images = f"เกิดข้อผิดพลาด: {str(e)}"

    history.append([message, reply_with_images])
    return "", history

def upload_file(file, username, project_id):
    if not file:
        return "กรุณาเลือกไฟล์ที่จะอัปโหลด"
    
    yield "🚀 กำลังอัปโหลดไฟล์และประมวลผล OCR..."

    try:
        files = {"file": (file.name, open(file.name, "rb"), "application/octet-stream")}
        params = {"username": username, "project_id": project_id}
        response = requests.post("http://localhost:8000/upload", files=files, params=params, timeout=360)

        if response.status_code == 200:
            result = response.json()
            message = result.get("message", "✅ อัปโหลดสำเร็จ")
        else:
            message = f"❌ API Error: {response.status_code} - {response.text}"
        yield message

    except Exception as e:
        yield f"❌ การอัปโหลดล้มเหลว: {str(e)}"

css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}
.header {
    text-align: center;
    background: linear-gradient(45deg, #00b4db, #0083b0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.chatbot {
    min-height: 500px;
    border: 2px solid #e1e5eb;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    background-color: #ffffff !important;
}
.send-btn {
    background: linear-gradient(45deg, #00b4db, #0083b0) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    margin-left: 10px !important;
    height: 52px !important;
}
.send-btn:hover {
    background: linear-gradient(45deg, #0083b0, #00b4db) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}
.clear-btn {
    background: linear-gradient(45deg, #ff9a9e, #fad0c4) !important;
    color: #333 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    margin-top: 10px !important;
}
.clear-btn:hover {
    background: linear-gradient(45deg, #fad0c4, #ff9a9e) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}
.upload-box {
    border: 2px dashed #00b4db !important;
    border-radius: 12px !important;
    padding: 20px !important;
    background-color: rgba(0, 180, 219, 0.05) !important;
}
.textbox {
    border-radius: 12px !important;
    border: 2px solid #e1e5eb !important;
    padding: 12px !important;
}
.input-container {
    border: 2px solid #e1e5eb !important;
    border-radius: 12px !important;
    padding: 10px !important;
    background: white !important;
}
.label {
    font-weight: bold !important;
    color: #0083b0 !important;
}
"""

with gr.Blocks(title="💬 Hybrid RAG Chatbot", css=css) as demo:
    gr.HTML("""
    <div class="header">
        <h1>💬 Hybrid RAG Chatbot</h1>
        <p>ระบบค้นหาผสม FAISS + BM25 พร้อม OCR และแสดงรูปภาพจาก S3</p>
    </div>
    """)

    with gr.Row():
        username_input = gr.Textbox(
            label="👤 USERNAME", 
            value=DEFAULT_USERNAME,
            elem_classes="textbox"
        )
        project_input = gr.Textbox(
            label="📁 PROJECT ID", 
            value=DEFAULT_PROJECT,
            elem_classes="textbox"
        )

    upload_status = gr.Markdown()

    with gr.Row():
        upload_button = gr.File(
            label="📤 อัปโหลดไฟล์ (PDF, PPTX, JPG, PNG)", 
            file_types=[".pdf", ".pptx", ".jpg", ".jpeg", ".png"],
            elem_classes="upload-box"
        )

    chatbot = gr.Chatbot(
        height=500, 
        label="การสนทนา", 
        show_copy_button=True,
        elem_classes="chatbot"
    )
    
    with gr.Row(elem_classes="input-container"):
        msg = gr.Textbox(
            label="คำถามของคุณ", 
            placeholder="พิมพ์คำถามที่นี่...", 
            scale=8,
            elem_classes="textbox",
            container=False,
            show_label=False
        )
        send_btn = gr.Button(
            "ส่งคำถาม ➡️", 
            elem_classes="send-btn",
            min_width=100
        )
    
    with gr.Row():
        clear = gr.Button("🧹 ล้างประวัติ", elem_classes="clear-btn")

    history_state = gr.State([])
    
    upload_button.upload(
        fn=upload_file,
        inputs=[upload_button, username_input, project_input],
        outputs=[upload_status]
    )
    
    # ฟังก์ชันการทำงานสำหรับทั้งปุ่มส่งและปุ่ม Enter
    def submit_fn(message, history, username, project_id):
        return chat_with_api(message, history, username, project_id)
    
    # เชื่อมต่อทั้งการกด Enter และการคลิกปุ่มส่ง
    msg.submit(
        fn=submit_fn, 
        inputs=[msg, history_state, username_input, project_input], 
        outputs=[msg, chatbot]
    )
    send_btn.click(
        fn=submit_fn, 
        inputs=[msg, history_state, username_input, project_input], 
        outputs=[msg, chatbot]
    )
    
    clear.click(
        lambda: None, 
        None, 
        chatbot, 
        queue=False
    ).then(
        lambda: None, 
        None, 
        upload_status, 
        queue=False
    ).then(
        lambda: [], 
        None, 
        history_state, 
        queue=False
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=False)