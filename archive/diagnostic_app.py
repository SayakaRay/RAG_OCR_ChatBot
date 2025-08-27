# diagnostic_app.py (เวอร์ชันแก้ไข NameError)

import base64
import json
import os
import re
import tempfile
from dataclasses import dataclass
from io import BytesIO
from typing import Tuple, Any, List
import time

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import fitz  # PyMuPDF
import gradio as gr

# --- โหลด Environment Variables ---
load_dotenv()
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY") # <--- แก้ไข Typo ที่บรรทัดนี้แล้ว
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"

# --- Data Classes and Prompts (เหมือนเดิม) ---
@dataclass
class PageElement:
    bbox: fitz.Rect
    sort_key: tuple
@dataclass
class TextElement(PageElement):
    text_content: str = ""
@dataclass
class ImageElement(PageElement):
    base64_data: str
PROMPT_OCR = "Transcribe the text content from this image. If it is a table, represent it as a plain text Markdown table."

# --- ฟังก์ชันสกัดและจัดเรียง Elements (พร้อม Log เพิ่มเติม) ---
def extract_and_sort_page_elements(page: fitz.Page, log_updates: list) -> List[PageElement]:
    log_updates.append("[DIAGNOSTIC] Starting element extraction...")
    elements = []
    
    images = page.get_images(full=True)
    for img_info in images:
        try:
            bbox = page.get_image_bbox(img_info)
            base_image = page.parent.extract_image(img_info[0])
            elements.append(ImageElement(bbox=bbox, sort_key=(bbox.y0, bbox.x0), base64_data=base64.b64encode(base_image["image"]).decode('utf-8')))
        except ValueError:
            continue
    log_updates.append(f"[DIAGNOSTIC] Found {len(elements)} image elements.")

    text_blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_INHIBIT_SPACES)["blocks"]
    text_element_count = 0
    for block in text_blocks:
        if block['type'] == 0:
            bbox = fitz.Rect(block['bbox'])
            if not any(bbox.intersects(img.bbox) for img in elements if isinstance(img, ImageElement)) and block['lines']:
                elements.append(TextElement(bbox=bbox, sort_key=(bbox.y0, bbox.x0)))
                text_element_count += 1
    log_updates.append(f"[DIAGNOSTIC] Found {text_element_count} text block elements.")

    if not elements:
        log_updates.append("[DIAGNOSTIC] ⚠️ No elements extracted. Creating a single catch-all element for the whole page.")
        page_bbox = page.rect
        catch_all_bbox = fitz.Rect(page_bbox.x0 + 20, page_bbox.y0 + 20, page_bbox.x1 - 20, page_bbox.y1 - 20)
        if not catch_all_bbox.is_empty:
            elements.append(TextElement(bbox=catch_all_bbox, sort_key=(catch_all_bbox.y0, catch_all_bbox.x0)))

    elements.sort(key=lambda e: e.sort_key)
    log_updates.append(f"[DIAGNOSTIC] ✅ Total elements to process: {len(elements)}")
    return elements

# --- ฟังก์ชันหลักในการประมวลผล (พร้อม Yielding Logs) ---
def run_diagnostic(pdf_file_obj):
    if pdf_file_obj is None:
        yield "Please upload a PDF file.", ""
        return

    log_updates = ["--- Starting Diagnostic ---"]
    yield "\n".join(log_updates), ""

    final_md_parts = []
    
    try:
        # ตรวจสอบ API Key อีกครั้ง
        if not TYPHOON_API_KEY:
            log_updates.append("[DIAGNOSTIC] ❌ FATAL ERROR: TYPHOON_API_KEY not found in .env file or environment variables.")
            yield "\n".join(log_updates), ""
            return

        openai = OpenAI(base_url=TYPHOON_BASE_URL, api_key=TYPHOON_API_KEY)
        
        pdf_path = pdf_file_obj.name
        log_updates.append(f"[DIAGNOSTIC] Opening PDF: {os.path.basename(pdf_path)}")
        yield "\n".join(log_updates), ""
        
        doc = fitz.open(pdf_path)
        
        # ประมวลผลหน้าแรกเท่านั้นเพื่อการวินิจฉัย
        page_num = 1
        page = doc.load_page(page_num - 1)
        
        # --- ขั้นตอนที่ 1: สกัดและจัดเรียง Elements ---
        sorted_elements = extract_and_sort_page_elements(page, log_updates)
        yield "\n".join(log_updates), ""
        
        if not sorted_elements:
            log_updates.append("[DIAGNOSTIC] ❌ ERROR: Element extraction resulted in an empty list even after fallback. Halting.")
            yield "\n".join(log_updates), ""
            doc.close()
            return

        # --- ขั้นตอนที่ 2 & 3: วนลูป OCR และร้อยเรียงผลลัพธ์ ---
        log_updates.append(f"[DIAGNOSTIC] 🚀 Starting element-by-element OCR for {len(sorted_elements)} elements...")
        yield "\n".join(log_updates), ""
        
        for i, element in enumerate(sorted_elements):
            log_updates.append(f"\n[DIAGNOSTIC] -> Processing element {i+1}/{len(sorted_elements)}...")
            yield "\n".join(log_updates), ""

            if isinstance(element, ImageElement):
                caption = f"Figure from page {page_num}, element {i+1}"
                final_md_parts.append(f"![{caption}](data:image/png;base64,{element.base64_data})")
                log_updates.append(f"[DIAGNOSTIC]    Type: Image. Appended Base64 tag.")
                yield "\n".join(log_updates), ""
                
            elif isinstance(element, TextElement):
                pix = page.get_pixmap(clip=element.bbox, dpi=300)
                base64_data = base64.b64encode(pix.tobytes("png")).decode('utf-8')
                messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT_OCR}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_data}"}}]}]
                
                try:
                    log_updates.append(f"[DIAGNOSTIC]    Type: Text. Sending {pix.width}x{pix.height}px crop to API...")
                    yield "\n".join(log_updates), ""
                    
                    response = openai.chat.completions.create(model="typhOon-ocr-preview", messages=messages, max_tokens=2048)
                    content = response.choices[0].message.content
                    
                    log_updates.append(f"[DIAGNOSTIC]    [RAW API RESPONSE]: {content}")
                    yield "\n".join(log_updates), ""
                    
                    parsed_content = content
                    if '{"natural_text":' in content:
                        try: parsed_content = json.loads(content).get('natural_text', '')
                        except: pass
                    
                    final_md_parts.append(parsed_content)
                    log_updates.append(f"[DIAGNOSTIC]    Successfully processed and appended content.")
                    yield "\n".join(log_updates), "\n\n".join(final_md_parts)

                except Exception as api_error:
                    log_updates.append(f"[DIAGNOSTIC]    ❌ API Error: {api_error}")
                    yield "\n".join(log_updates), ""
                    final_md_parts.append(f"(Error processing text block at {element.bbox})")

        doc.close()
        
        final_markdown = "\n\n".join(final_md_parts)
        log_updates.append("\n[DIAGNOSTIC] ✅ All elements processed. Final Markdown generated.")
        yield "\n".join(log_updates), final_markdown

    except Exception as e:
        log_updates.append(f"\n[DIAGNOSTIC] ❌ A fatal error occurred: {e}")
        import traceback
        log_updates.append(traceback.format_exc())
        yield "\n".join(log_updates), ""

# --- สร้าง Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 OCR Pipeline Diagnostic App")
    gr.Markdown("อัปโหลดไฟล์ PDF 'ธรรมดา' ที่เคยให้ผลลัพธ์ว่างเปล่า เพื่อดูการทำงานทีละขั้นตอน")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF File", file_types=[".pdf"])
        run_button = gr.Button("▶️ Start Diagnostic", variant="primary")
        
    with gr.Row():
        log_output = gr.Textbox(label="Log Output", lines=20, interactive=False)
        final_output = gr.Markdown(label="Final Markdown Output")
        
    run_button.click(
        fn=run_diagnostic,
        inputs=[pdf_input],
        outputs=[log_output, final_output]
    )

if __name__ == "__main__":
    print("Launching Diagnostic App...")
    print("Please open the URL in your browser.")
    demo.launch()