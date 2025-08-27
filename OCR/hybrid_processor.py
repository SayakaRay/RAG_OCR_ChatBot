# OCR/hybrid_processor.py (เวอร์ชันปรับปรุงใช้ Google Vision API)

import os
import io
import base64
from pathlib import Path
from PIL import Image
import torch
import fitz  # PyMuPDF
from dotenv import load_dotenv
import json
from bs4 import BeautifulSoup
import traceback
import time
import random
import math
import requests

# --- ส่วนที่ 1: Surya (แก้ไขให้ใช้ API ใหม่) ---
DEVICE = "cpu"
LAYOUT_PREDICTOR = None

# ตั้งค่า environment variables สำหรับ Surya
os.environ['TORCH_DEVICE'] = DEVICE
os.environ['LAYOUT_BATCH_SIZE'] = '2'  # ลด batch size สำหรับ CPU
os.environ['SURYA_MODEL_PATH'] = './models'  # path ไปยังโมเดลที่คุณดาวน์โหลด

def _initialize_surya_models():
    """
    โหลดโมเดล Layout ของ Surya (ใช้ LayoutPredictor จากเวอร์ชันใหม่)
    """
    global LAYOUT_PREDICTOR
    if LAYOUT_PREDICTOR is None:
        print(f"[INFO] Initializing Surya Layout Model on {DEVICE} (อาจใช้เวลาสักครู่)...")
        try:
            # ใช้ LayoutPredictor จากเวอร์ชันใหม่
            from surya.layout import LayoutPredictor
            
            LAYOUT_PREDICTOR = LayoutPredictor()
            print("[INFO] ✅ Surya Layout Model loaded successfully.")
        except ImportError as e:
            print(f"[ERROR] ❌ ImportError occurred. Details: {e}")
            # พิมพ์ Path ที่ Python กำลังค้นหาเพื่อช่วยดีบัก
            import sys
            print("Python search paths (sys.path):")
            for p in sys.path:
                print(f"- {p}")
            raise
        except Exception as e:
            print(f"[ERROR] ❌ Failed to load Surya model: {e}")
            traceback.print_exc()
            raise

def _get_reading_order_score(bbox, image_width, image_height):
    """
    คำนวณคะแนนสำหรับการจัดลำดับการอ่าน (ซ้ายไปขวา บนลงล่าง)
    ใช้ grid-based approach เพื่อให้การจัดลำดับแม่นยำมากขึ้น
    """
    x1, y1, x2, y2 = bbox
    
    # ใช้จุดกลางของ bbox สำหรับการคำนวณ
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # แบ่งหน้าเป็น grid เพื่อจัดกลุ่มตามแถว
    # ใช้ row height ประมาณ 1/20 ของความสูงหน้า
    row_height = image_height / 20
    row_index = int(center_y / row_height)
    
    # คะแนนหลัก: แถว (row) มีน้ำหนักมาก
    # คะแนนรอง: ตำแหน่ง x ในแถวเดียวกัน
    score = (row_index * 10000) + center_x
    
    return score

def _merge_overlapping_boxes(elements, overlap_threshold=0.7):
    """
    รวม bounding boxes ที่ทับซ้อนกันมาก เพื่อป้องกันการแยกข้อความ
    """
    if not elements:
        return elements
    
    def calculate_overlap_ratio(bbox1, bbox2):
        """คำนวณอัตราการทับซ้อนระหว่าง 2 bbox"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # หาพื้นที่ทับซ้อน
        x1_overlap = max(x1_1, x1_2)
        y1_overlap = max(y1_1, y1_2)
        x2_overlap = min(x2_1, x2_2)
        y2_overlap = min(y2_1, y2_2)
        
        if x1_overlap >= x2_overlap or y1_overlap >= y2_overlap:
            return 0.0
        
        overlap_area = (x2_overlap - x1_overlap) * (y2_overlap - y1_overlap)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        return overlap_area / min(area1, area2)
    
    merged_elements = []
    used_indices = set()
    
    for i, elem1 in enumerate(elements):
        if i in used_indices:
            continue
            
        merged_bbox = list(elem1['bbox'])
        merged_labels = [elem1['label']]
        merged_confidences = [elem1['confidence']]
        group_indices = [i]
        
        # หาองค์ประกอบที่ทับซ้อนกัน
        for j, elem2 in enumerate(elements[i+1:], i+1):
            if j in used_indices:
                continue
                
            overlap_ratio = calculate_overlap_ratio(elem1['bbox'], elem2['bbox'])
            if overlap_ratio >= overlap_threshold:
                # รวม bbox โดยใช้ union
                merged_bbox[0] = min(merged_bbox[0], elem2['bbox'][0])  # x1
                merged_bbox[1] = min(merged_bbox[1], elem2['bbox'][1])  # y1
                merged_bbox[2] = max(merged_bbox[2], elem2['bbox'][2])  # x2
                merged_bbox[3] = max(merged_bbox[3], elem2['bbox'][3])  # y2
                
                merged_labels.append(elem2['label'])
                merged_confidences.append(elem2['confidence'])
                group_indices.append(j)
        
        # เพิ่มในรายการที่ใช้แล้ว
        used_indices.update(group_indices)
        
        # สร้าง element ใหม่ที่รวมแล้ว
        merged_element = {
            'bbox': merged_bbox,
            'label': max(set(merged_labels), key=merged_labels.count),  # ใช้ label ที่พบบ่อยที่สุด
            'confidence': sum(merged_confidences) / len(merged_confidences),  # ค่าเฉลี่ย confidence
            'merged_count': len(group_indices)
        }
        merged_elements.append(merged_element)
    
    print(f"   -> Merged {len(elements)} elements into {len(merged_elements)} elements")
    return merged_elements

def _expand_bbox_for_context(bbox, image_width, image_height, expansion_ratio=0.05):
    """
    ขยาย bounding box เล็กน้อยเพื่อให้แน่ใจว่าจับข้อความได้ครบถ้วน
    """
    x1, y1, x2, y2 = bbox
    
    width = x2 - x1
    height = y2 - y1
    
    # ขยายตามอัตราส่วนที่กำหนด
    expand_x = width * expansion_ratio
    expand_y = height * expansion_ratio
    
    # คำนวณ bbox ใหม่และจำกัดขอบเขต
    new_x1 = max(0, int(x1 - expand_x))
    new_y1 = max(0, int(y1 - expand_y))
    new_x2 = min(image_width, int(x2 + expand_x))
    new_y2 = min(image_height, int(y2 + expand_y))
    
    return [new_x1, new_y1, new_x2, new_y2]

def _surya_layout_analysis(pil_image):
    """
    ใช้ Surya เพื่อวิเคราะห์ Layout และคืนค่ารายการของ BBoxes ทั้งหมดในหน้า
    พร้อมการจัดลำดับแบบซ้ายไปขวา บนลงล่าง และการรวม bbox ที่ทับซ้อน
    """
    print("   -> Running Surya layout detection...")
    
    try:
        # ใช้ LayoutPredictor ทำการ predict
        layout_predictions = LAYOUT_PREDICTOR([pil_image])
        
        elements = []
        # ดึง bboxes จากผลลัพธ์
        page_prediction = layout_predictions[0]
        
        if hasattr(page_prediction, 'bboxes') and page_prediction.bboxes:
            for bbox_obj in page_prediction.bboxes:
                # แปลง bbox format และเพิ่มข้อมูล
                elements.append({
                    "bbox": bbox_obj.bbox,  # [x1, y1, x2, y2]
                    "label": getattr(bbox_obj, 'label', 'Text'),  # ใช้ label จาก Surya หรือ default เป็น 'Text'
                    "confidence": getattr(bbox_obj, 'confidence', 1.0)  # confidence score
                })
        else:
            print("   -> ⚠️ No bboxes found in layout prediction")
            # fallback: ใช้ทั้งรูปภาพ
            width, height = pil_image.size
            elements.append({
                "bbox": [0, 0, width, height],
                "label": "Text",
                "confidence": 1.0
            })
        
        print(f"   -> Found {len(elements)} initial layout elements")
        
        # รวม bounding boxes ที่ทับซ้อนกันมาก
        elements = _merge_overlapping_boxes(elements, overlap_threshold=0.6)
        
        # ขยาย bbox เล็กน้อยเพื่อให้แน่ใจว่าจับข้อความได้ครบ
        image_width, image_height = pil_image.size
        for elem in elements:
            elem['bbox'] = _expand_bbox_for_context(
                elem['bbox'], image_width, image_height, expansion_ratio=0.02
            )
        
        # จัดลำดับตาม reading order (ซ้ายไปขวา บนลงล่าง)
        elements.sort(key=lambda e: _get_reading_order_score(e['bbox'], image_width, image_height))
        
        print(f"   -> Final: {len(elements)} layout elements (merged and sorted)")
        
        # แสดงข้อมูลการจัดลำดับสำหรับ debug
        for i, elem in enumerate(elements):
            bbox = elem['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            print(f"      Element {i+1}: {elem['label']} at center({center_x:.0f}, {center_y:.0f})")
        
        return elements
        
    except Exception as e:
        print(f"   -> ❌ Error in layout analysis: {e}")
        traceback.print_exc()
        # fallback: ใช้ทั้งรูปภาพ
        width, height = pil_image.size
        return [{
            "bbox": [0, 0, width, height],
            "label": "Text",
            "confidence": 1.0
        }]

# --- ส่วนที่ 2: Google Vision API ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("⚠️ Warning: GOOGLE_API_KEY not found in environment variables")

def convert_html_tables_to_markdown(html_content: str) -> str:
    """แปลง HTML tables เป็น Markdown format"""
    if '<table>' not in html_content:
        return html_content
    soup = BeautifulSoup(html_content, 'lxml')
    for table in soup.find_all('table'):
        markdown_table = ""
        headers = [th.get_text(strip=True).replace("\n", " ") for th in table.find_all('th')]
        tr = table.find('tr')
        if not headers and tr:
             headers = [td.get_text(strip=True).replace("\n", " ") for td in tr.find_all('td')]
        if headers:
            markdown_table += "| " + " | ".join(headers) + " |\n"
            markdown_table += "| " + " | ".join(['---'] * len(headers)) + " |\n"
            start_row = 1
        else:
            start_row = 0
        rows = table.find_all('tr')
        for row in rows[start_row:]:
            columns = [td.get_text(strip=True).replace("\n", " ") for td in row.find_all('td')]
            if len(columns) == len(headers) or not headers:
                markdown_table += "| " + " | ".join(columns) + " |\n"
        if markdown_table:
            table.replace_with(markdown_table)
    return str(soup)

def _detect_text_type(element_info):
    """
    ตรวจสอบประเภทของข้อความเพื่อกำหนดว่าควรใช้ TEXT_DETECTION หรือ DOCUMENT_TEXT_DETECTION
    """
    if not element_info or not element_info.get('label'):
        return 'DOCUMENT_TEXT_DETECTION'  # default
    
    label = element_info['label'].lower()
    
    # ✅ FIX: ใช้ 'DOCUMENT_TEXT_DETECTION' สำหรับลายมือ
    if 'handwriting' in label or 'handwritten' in label or 'hand' in label:
        return 'DOCUMENT_TEXT_DETECTION'
    
    # ใช้ DOCUMENT_TEXT_DETECTION สำหรับเอกสารที่มีโครงสร้าง
    elif any(keyword in label for keyword in ['table', 'document', 'paragraph', 'column']):
        return 'DOCUMENT_TEXT_DETECTION'
    
    # ใช้ TEXT_DETECTION สำหรับข้อความทั่วไป
    else:
        return 'TEXT_DETECTION'

def _format_vision_response(response_data, element_info=None):
    """
    จัดรูปแบบผลลัพธ์จาก Google Vision API ให้เป็น Markdown
    รองรับทั้ง TEXT_DETECTION และ DOCUMENT_TEXT_DETECTION
    """
    try:
        responses = response_data.get('responses', [])
        if not responses:
            return "[No response from Vision API]"
        
        response = responses[0]
        
        # ตรวจสอบ error
        if 'error' in response:
            return f"[Vision API Error: {response['error'].get('message', 'Unknown error')}]"
        
        # ลองดึงข้อความจาก fullTextAnnotation ก่อน (สำหรับ DOCUMENT_TEXT_DETECTION)
        if 'fullTextAnnotation' in response:
            full_text = response['fullTextAnnotation'].get('text', '').strip()
            if full_text:
                # จัดรูปแบบข้อความให้เป็น markdown ตามประเภทของ element
                if element_info and element_info.get('label'):
                    label = element_info['label'].lower()
                    if 'title' in label or 'header' in label:
                        # ทำให้เป็น header
                        lines = full_text.split('\n')
                        if lines:
                            full_text = f"### {lines[0]}\n" + '\n'.join(lines[1:])
                    elif 'table' in label:
                        # พยายามจัดรูปแบบเป็นตาราง (ถ้าเป็นไปได้)
                        full_text = _try_format_as_table(full_text)
                
                return full_text
        
        # ถ้าไม่มี fullTextAnnotation ให้ใช้ textAnnotations
        if 'textAnnotations' in response:
            text_annotations = response['textAnnotations']
            if text_annotations:
                # textAnnotations[0] มักจะมีข้อความทั้งหมด
                full_text = text_annotations[0].get('description', '').strip()
                return full_text if full_text else "[No text detected]"
        
        return "[No text found in response]"
        
    except Exception as e:
        print(f"    -> ❌ Error formatting Vision response: {e}")
        return f"[Error formatting response: {str(e)}]"

def _try_format_as_table(text):
    """
    พยายามจัดรูปแบบข้อความให้เป็นตาราง Markdown
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) < 2:
        return text
    
    # ตรวจสอบว่าแต่ละบรรทัดมี pattern ที่คล้ายคลึงกัน (เช่น มีจำนวนคำใกล้เคียงกัน)
    word_counts = [len(line.split()) for line in lines]
    avg_words = sum(word_counts) / len(word_counts)
    
    # ถ้าแต่ละบรรทัดมีจำนวนคำที่ใกล้เคียงกัน อาจเป็นตาราง
    if all(abs(count - avg_words) <= 2 for count in word_counts) and avg_words >= 2:
        try:
            # พยายามสร้างตาราง
            max_cols = max(word_counts)
            table_lines = []
            
            # Header
            header_parts = lines[0].split()
            while len(header_parts) < max_cols:
                header_parts.append("")
            table_lines.append("| " + " | ".join(header_parts) + " |")
            table_lines.append("| " + " | ".join(['---'] * len(header_parts)) + " |")
            
            # Rows
            for line in lines[1:]:
                parts = line.split()
                while len(parts) < max_cols:
                    parts.append("")
                table_lines.append("| " + " | ".join(parts[:max_cols]) + " |")
            
            return '\n'.join(table_lines)
        except:
            pass
    
    return text

def _google_vision_process_crop(pil_image_crop, element_info=None, retry_count=0, max_retries=3):
    """
    ใช้ Google Vision API ในการประมวลผลรูปภาพที่ crop มา
    รองรับทั้ง text detection และ handwriting detection
    รองรับ retry และ delay เพื่อแก้ปัญหา rate limit
    """
    if not GOOGLE_API_KEY:
        return "[Error: GOOGLE_API_KEY not configured]"
    
    # เพิ่ม delay ก่อนส่ง request เพื่อหลีกเลี่ยง rate limit
    base_delay = 0.5  # 0.5 วินาที (Google Vision มี rate limit น้อยกว่า Typhoon)
    jitter = random.uniform(0.1, 0.3)
    delay = base_delay + jitter + (retry_count * 1)
    
    if retry_count > 0:  # แสดง delay เฉพาะตอน retry
        print(f"    -> Waiting {delay:.1f}s before Vision request (attempt {retry_count + 1}/{max_retries + 1})...")
    
    time.sleep(delay)
    
    # แปลงรูปภาพเป็น base64
    buffered = io.BytesIO()
    pil_image_crop.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # กำหนดประเภทของการตรวจจับ
    detection_type = _detect_text_type(element_info)
    
    # สร้าง request payload
    request_payload = {
        "requests": [
            {
                "image": {
                    "content": base64_image
                },
                "features": [
                    {
                        "type": detection_type,
                        "maxResults": 50  # เพิ่ม maxResults เพื่อให้ได้ผลลัพธ์ครบถ้วน
                    }
                ]
            }
        ]
    }
    
    # ถ้าเป็น DOCUMENT_TEXT_DETECTION ให้เพิ่ม imageContext สำหรับ language hints (ซึ่งจะรวมลายมือด้วย)
    if detection_type == 'DOCUMENT_TEXT_DETECTION':
        request_payload["requests"][0]["imageContext"] = {
            "languageHints": ["th", "en"]  # รองรับทั้งภาษาไทยและอังกฤษ
        }
    
    vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}"
    
    try:
        response = requests.post(
            vision_api_url,
            headers={'Content-Type': 'application/json'},
            json=request_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            response_data = response.json()
            result = _format_vision_response(response_data, element_info)
            
            # แปลง HTML tables เป็น Markdown (ถ้ามี)
            result = convert_html_tables_to_markdown(result)
            
            detection_info = f" (using {detection_type})"
            merged_info = f" (merged: {element_info.get('merged_count', 1)} elements)" if element_info and element_info.get('merged_count', 1) > 1 else ""
            print(f"    -> ✅ Google Vision OCR success{detection_info}{merged_info}")
            return result
            
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            
            # ตรวจสอบ rate limit errors
            if response.status_code == 429 or 'quota' in response.text.lower():
                if retry_count < max_retries:
                    print(f"    -> ⚠️ Rate limit/quota hit, retrying in {2 ** retry_count * 2}s...")
                    time.sleep(2 ** retry_count * 2)  # exponential backoff
                    return _google_vision_process_crop(pil_image_crop, element_info, retry_count + 1, max_retries)
                else:
                    return f"[Vision API Error: Rate limit exceeded after {max_retries + 1} attempts]"
            
            # ตรวจสอบ authentication errors
            elif response.status_code == 401:
                return "[Vision API Error: Invalid API key]"
            
            # errors อื่นๆ
            else:
                if retry_count < max_retries:
                    print(f"    -> ⚠️ Vision API error (HTTP {response.status_code}), retrying...")
                    time.sleep(2)
                    return _google_vision_process_crop(pil_image_crop, element_info, retry_count + 1, max_retries)
                else:
                    return f"[Vision API Error: {error_msg}]"
                
    except requests.exceptions.Timeout:
        if retry_count < max_retries:
            print(f"    -> ⚠️ Request timeout, retrying...")
            time.sleep(3)
            return _google_vision_process_crop(pil_image_crop, element_info, retry_count + 1, max_retries)
        else:
            return f"[Vision API Error: Timeout after {max_retries + 1} attempts]"
    
    except Exception as e:
        print(f"    -> ❌ Google Vision API error: {e}")
        return f"[Vision API Error: {str(e)}]"

# --- HybridProcessor Class ---
class HybridProcessor:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        _initialize_surya_models()

    def run(self) -> list:
        all_pages_content = []
        try:
            print(f"Processing Hybrid (Surya+Google Vision) on PDF: {self.pdf_path.name}")
            with fitz.open(self.pdf_path) as doc:
                print(f"📄 พบเอกสารทั้งหมด {doc.page_count} หน้า จะทำการประมวลผลทุกหน้า...")
                for page_num_zero_based in range(doc.page_count):
                    page_num_one_based = page_num_zero_based + 1
                    print(f"\n{'='*20} Processing Page {page_num_one_based}/{doc.page_count} {'='*20}")
                    page = doc.load_page(page_num_zero_based)
                    
                    # แปลง PDF page เป็น image
                    pix = page.get_pixmap(dpi=200)
                    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Stage 1: Layout Analysis ด้วย Surya (พร้อมการจัดลำดับและรวม bbox)
                    print(f" -- Page {page_num_one_based}: Stage 1 - Layout Analysis with Surya...")
                    elements = _surya_layout_analysis(pil_image)
                    print(f" -- Page {page_num_one_based}: Found {len(elements)} layout elements (sorted by reading order).")

                    # Stage 2: OCR แต่ละ element ด้วย Google Vision (ตามลำดับที่จัดเรียบร้อยแล้ว)
                    print(f" -- Page {page_num_one_based}: Stage 2 - OCR each element with Google Vision (following reading order)...")
                    page_content_parts = []
                    total_elements = len(elements)
                    
                    for i, elem in enumerate(elements):
                        bbox = [int(coord) for coord in elem['bbox']]
                        
                        merged_info = f" (merged: {elem.get('merged_count', 1)} elements)" if elem.get('merged_count', 1) > 1 else ""
                        print(f"    -> Processing element {i+1}/{total_elements} (type: {elem['label']}, conf: {elem['confidence']:.3f}){merged_info}...")
                        
                        # Crop รูปภาพตาม bbox ที่ขยายแล้ว
                        try:
                            cropped_image = pil_image.crop(tuple(bbox))
                            
                            # ตรวจสอบว่า crop ได้รูปที่มีขนาดพอสมควร
                            if cropped_image.size[0] < 10 or cropped_image.size[1] < 10:
                                print(f"    -> ⚠️ Skipping too small crop: {cropped_image.size}")
                                continue
                                
                            # ส่งไป OCR ด้วย Google Vision (พร้อม retry logic)
                            markdown_text = _google_vision_process_crop(cropped_image, elem)
                            if markdown_text.strip() and not markdown_text.startswith('[Vision API Error:') and not markdown_text.startswith('[Error:'):
                                page_content_parts.append(markdown_text.strip())
                                print(f"    -> ✅ Successfully extracted text from element {i+1}")
                            elif markdown_text.startswith('[Vision API Error:') or markdown_text.startswith('[Error:'):
                                print(f"    -> ⚠️ OCR failed for element {i+1}: {markdown_text}")
                                # ยังคงเก็บ error message ไว้เพื่อดูปัญหา
                                page_content_parts.append(markdown_text)
                            else:
                                print(f"    -> ℹ️ No text found in element {i+1}")
                                
                        except Exception as e:
                            print(f"    -> ⚠️ Error processing element {i+1}: {e}")
                            continue
                    
                    # แสดงสถิติการประมวลผล
                    success_count = sum(1 for part in page_content_parts if not (part.startswith('[Vision API Error:') or part.startswith('[Error:')))
                    error_count = len(page_content_parts) - success_count
                    print(f" -- Page {page_num_one_based}: OCR Results - Success: {success_count}, Errors: {error_count}")
                    
                    # Stage 3: รวม results ตามลำดับการอ่าน
                    print(f" -- Page {page_num_one_based}: Stage 3 - Merging results in reading order...")
                    final_markdown_for_page = "\n\n".join(filter(None, page_content_parts))
                    
                    if not final_markdown_for_page.strip():
                        final_markdown_for_page = f"[No readable content found on page {page_num_one_based}]"
                    else:
                        # เพิ่ม header ระบุหน้า
                        final_markdown_for_page = f"## Page {page_num_one_based}\n\n{final_markdown_for_page}"
                    
                    all_pages_content.append(final_markdown_for_page)
                    print(f" -- Page {page_num_one_based}: Successfully processed with proper reading order.")

            print(f"\n🎉 Successfully processed all {len(all_pages_content)} pages with improved layout ordering using Google Vision!")
            return all_pages_content
            
        except Exception as e:
            print(f"❌ Error processing {self.pdf_path.name}: {e}")
            traceback.print_exc()
            return [f"An error occurred while processing {self.pdf_path.name}: {e}"]