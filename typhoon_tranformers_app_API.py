# typhoon_tranformers_app_API.py (เวอร์ชันสุดท้าย: AI-Guided Cropping)

import base64
import json
import os
from typing import Tuple, Any
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import fitz  # PyMuPDF

load_dotenv()
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")
TYPHOON_BASE_URL = os.getenv("TYPHOON_BASE_URL", "api.opentyphoon.ai/v1")
if not TYPHOON_BASE_URL.startswith(("http://", "https://")):
    TYPHOON_BASE_URL = "https://" + TYPHOON_BASE_URL

# --- Prompt ใหม่ล่าสุด: สั่งให้ AI หาพิกัดของรูปภาพและคืนค่าเป็น JSON ---
PROMPT_AI_GUIDED_CROPPING = (
    "You are an expert document analysis system. Your task is to do two things:\n"
    "1. Reconstruct the document in the image into clean Markdown format. All tables must be plain text Markdown.\n"
    "2. Identify all figures or images in the document. For each figure, determine its bounding box [x1, y1, x2, y2] relative to the image dimensions.\n"
    "Your final output MUST be a single JSON object with two keys:\n"
    "- 'markdown_text': A string containing the full reconstructed markdown content. The figure captions should be included in this text.\n"
    "- 'figures': A list of objects, where each object has a 'caption' string (the full text of the figure's caption) and a 'bbox' list of 4 numbers [x1, y1, x2, y2].\n"
    "If no figures are found, return an empty list for the 'figures' key."
)

# --- ฟังก์ชันหลักในการประมวลผล ---
def process_pdf_via_api(pdf_path: str, page_num: int, task_type: str = "structure") -> Tuple[Any, str]:
    print(f"\n{'='*20} Processing Page {page_num} of PDF: {os.path.basename(pdf_path)} {'='*20}")
    
    try:
        # Step 1: เตรียมไฟล์ภาพต้นฉบับสำหรับ AI และสำหรับการ Crop
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap(dpi=300) # ใช้ DPI สูงเพื่อให้ AI เห็นรายละเอียดชัดเจน
            page_image_bytes = pix.tobytes("png")
            page_image_base64 = base64.b64encode(page_image_bytes).decode("utf-8")
            original_pil_image = Image.open(BytesIO(page_image_bytes))

        # Step 2: เรียก API ด้วย Prompt ใหม่เพียงครั้งเดียว
        print("🚀 Sending full page to API for combined analysis (Text + Figure BBoxes)...")
        messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT_AI_GUIDED_CROPPING}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_image_base64}"}}]}]
        
        openai = OpenAI(base_url=TYPHOON_BASE_URL, api_key=TYPHOON_API_KEY)
        response = openai.chat.completions.create(model="typhoon-ocr-preview", messages=messages, max_tokens=4096,
            response_format={"type": "json_object"}) # บังคับให้ AI ตอบกลับเป็น JSON
        
        text_output = response.choices[0].message.content
        
        # Step 3: ประมวลผลผลลัพธ์ JSON
        final_markdown = ""
        if text_output and text_output.strip():
            print("✅ Received structured JSON response from API.")
            try:
                response_data = json.loads(text_output)
                final_markdown = response_data.get('markdown_text', '')
                figures_to_embed = response_data.get('figures', [])
                
                print(f"🔎 AI identified {len(figures_to_embed)} figures to embed.")

                # Step 4: วนลูปเพื่อ Crop และแทนที่รูปภาพตามพิกัดที่ AI ให้มา
                for figure in figures_to_embed:
                    caption = figure.get('caption', 'Extracted Figure')
                    bbox = figure.get('bbox')

                    if not isinstance(bbox, list) or len(bbox) != 4:
                        print(f"⚠️ Skipping figure with invalid BBox: {bbox}")
                        continue
                    
                    # Crop รูปจาก PIL Image ที่เราเก็บไว้
                    cropped_image = original_pil_image.crop(tuple(bbox))
                    
                    # เข้ารหัสเป็น Base64
                    buffered = BytesIO()
                    cropped_image.save(buffered, format="PNG")
                    base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    image_data_url = f"data:image/png;base64,{base64_data}"
                    
                    final_tag = f"![{caption}]({image_data_url})"
                    # แทนที่ข้อความ caption เดิมใน Markdown ด้วยแท็กรูปภาพ
                    if caption in final_markdown:
                        final_markdown = final_markdown.replace(caption, final_tag, 1)
                        print(f"✅ Embedded figure for caption: '{caption[:30]}...'")
                    else:
                        print(f"⚠️ Caption '{caption[:30]}...' not found in markdown_text. Appending image to the end.")
                        final_markdown += "\n\n" + final_tag

            except json.JSONDecodeError:
                print("⚠️ API did not return a valid JSON. Using raw output.")
                final_markdown = text_output
            except Exception as e:
                print(f"❌ Error during embedding process: {e}")
                final_markdown = response_data.get('markdown_text', text_output)
        else:
            print("⚠️ API response is empty.")
            
        return None, final_markdown

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"An error occurred: {e}"