import gradio as gr
import requests
import re
import base64
from io import BytesIO

# --------------------------
# Function to get image from S3/local API
# --------------------------
def get_image_from_s3(image_filename):
    try:
        filename = image_filename.split('/')[-1].split('.')[0]  # extract name without extension
        response = requests.get(f"http://localhost:8000/get-image-binary/{filename}", timeout=30)
        if response.status_code == 200:
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
        return None
    except Exception as e:
        print(f"Error fetching image {image_filename}: {e}")
        return None

# --------------------------
# Replace image markdown with actual <img> tag
# --------------------------
def process_images_in_text(text):
    image_pattern = r'!\[.*?\]\((.*?)\)'

    def replace_image(match):
        image_filename = match.group(1)
        try:
            image_url = get_image_from_s3(image_filename)
            if image_url:
                return f'<img src="{image_url}" style="max-width: 400px; max-height: 300px; border-radius: 8px; margin: 10px 0;"/>'
            else:
                return f"\n\n🖼️ ไม่สามารถโหลดรูปภาพ: {image_filename}\n\n"
        except Exception as e:
            return f"\n\n🖼️ รูปภาพผิดพลาด: {str(e)}\n\n"

    return re.sub(image_pattern, replace_image, text)

# --------------------------
# Function to handle chatbot interaction
# --------------------------
def chat_with_api(message, history):
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
            reply_with_images = process_images_in_text(reply)
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

# --------------------------
# Optional: Upload File Handler
# --------------------------
def upload_file(file):
    try:
        files = {"file": (file.name, open(file.name, "rb"), "application/octet-stream")}
        response = requests.post("http://localhost:8000/upload", files=files, timeout=60)
        return response.json().get("message", response.text)
    except Exception as e:
        return f"❌ การอัปโหลดล้มเหลว: {str(e)}"

# --------------------------
# Gradio UI Layout
# --------------------------
with gr.Blocks(title="💬 Hybrid RAG Chatbot") as demo:
    gr.Markdown("# 💬 Hybrid RAG Chatbot")
    gr.Markdown("ระบบค้นหาผสม FAISS + BM25 พร้อม OCR และแสดงรูปภาพจาก S3")

    chatbot = gr.Chatbot(height=500, label="การสนทนา")
    msg = gr.Textbox(label="คำถามของคุณ", placeholder="พิมพ์คำถามที่นี่...", scale=8)
    upload_button = gr.File(label="📤 อัปโหลดไฟล์ (PDF, PPTX, JPG, PNG)", file_types=[".pdf", ".pptx", ".jpg", ".jpeg", ".png"])
    clear = gr.Button("🧹 ล้างประวัติ")
    history_state = gr.State([])

    msg.submit(chat_with_api, [msg, history_state], [msg, chatbot])
    clear.click(lambda: ([], []), None, [history_state, chatbot])
    upload_button.change(fn=upload_file, inputs=upload_button, outputs=None)

if __name__ == "__main__":
    demo.launch(debug=True, share=False)
