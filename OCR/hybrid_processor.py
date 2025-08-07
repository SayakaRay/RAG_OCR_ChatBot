# OCR/hybrid_processor.py (เวอร์ชันแก้ไขสมบูรณ์)

import os
import io
import base64
from pathlib import Path
from PIL import Image
import torch
import cv2
import re
import fitz # PyMuPDF
from transformers import AutoProcessor, VisionEncoderDecoderModel
from dotenv import load_dotenv
from openai import OpenAI
import json
from bs4 import BeautifulSoup

# --- ส่วนที่ 1: Dolphin ---
DEVICE = "cpu"
MODEL, PROCESSOR, TOKENIZER = None, None, None

def _initialize_dolphin_model():
    """โหลดโมเดล Dolphin และ Processor (ทำงานครั้งเดียว)"""
    global MODEL, PROCESSOR, TOKENIZER
    if MODEL is None:
        print("[INFO] Initializing Dolphin Model (อาจใช้เวลาสักครู่)...")
        model_id = "ByteDance/Dolphin"
        try:
            PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            MODEL = VisionEncoderDecoderModel.from_pretrained(model_id, trust_remote_code=True)
            MODEL.eval(); MODEL.to(DEVICE)
            TOKENIZER = PROCESSOR.tokenizer
            print(f"[INFO] ✅ Dolphin Model loaded successfully on {DEVICE}")
        except Exception as e:
            print(f"[ERROR] ❌ Failed to load Dolphin model: {e}")
            return None, None, None

def parse_layout_string_for_figures(layout_string: str, image_dims: tuple):
    img_w, img_h = image_dims
    elements_str = re.split(r'\[(?:PAIR_SEP|RELATION_SEP)\]', layout_string)
    figure_bboxes = []
    pattern = re.compile(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\s*(\w+)')
    for elem_str in elements_str:
        match = pattern.match(elem_str.strip())
        if match and match.groups()[4] == "fig":
            rel_coords = [float(c) for c in match.groups()[:4]]
            figure_bboxes.append([int(rel_coords[0] * img_w), int(rel_coords[1] * img_h), int(rel_coords[2] * img_w), int(rel_coords[3] * img_h)])
    return figure_bboxes

def _dolphin_layout_analysis(pil_image):
    prompt = "Parse the reading order of this document."
    batch_inputs = PROCESSOR([pil_image], return_tensors="pt", padding=True)
    batch_pixel_values = batch_inputs.pixel_values.to(DEVICE)
    prompt_str = f"<s>{prompt} <Answer/>"
    batch_prompt_inputs = TOKENIZER([prompt_str], add_special_tokens=False, return_tensors="pt")
    batch_prompt_ids = batch_prompt_inputs.input_ids.to(DEVICE)
    batch_attention_mask = batch_prompt_inputs.attention_mask.to(DEVICE)
    outputs = MODEL.generate(pixel_values=batch_pixel_values, decoder_input_ids=batch_prompt_ids, decoder_attention_mask=batch_attention_mask, max_length=4096, pad_token_id=TOKENIZER.pad_token_id, eos_token_id=TOKENIZER.eos_token_id, use_cache=True, return_dict_in_generate=True)
    sequence = TOKENIZER.batch_decode(outputs.sequences, skip_special_tokens=False)[0]
    layout_string = sequence.replace(prompt_str, "").replace("<pad>", "").replace("</s>", "").strip()
    return parse_layout_string_for_figures(layout_string, pil_image.size)

# --- ส่วนที่ 2: Typhoon ---
load_dotenv()
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")
TYPHOON_BASE_URL = os.getenv("TYPHOON_BASE_URL", "api.opentyphoon.ai/v1")
if not TYPHOON_BASE_URL.startswith(("http://", "https://")):
    TYPHOON_BASE_URL = "https://" + TYPHOON_BASE_URL

PROMPT_TYPHOON_OCR = ("You are an expert document analyst. Reconstruct the document you see in the image into a clean, well-structured Markdown format. All tables MUST be in plain text Markdown format (using pipes |). Transcribe all text, including figure captions, exactly as you see it. Your final output must be in JSON format with a single key `natural_text` containing the response.")

def convert_html_tables_to_markdown(html_content: str) -> str:
    if '<table>' not in html_content:
        return html_content
    print(" -- Found HTML table, converting to Markdown...")
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

def _typhoon_full_page_ocr(base64_image):
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT_TYPHOON_OCR}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}]
    openai = OpenAI(base_url=TYPHOON_BASE_URL, api_key=TYPHOON_API_KEY)
    response = openai.chat.completions.create(model="typhoon-ocr-preview", messages=messages, max_tokens=4096)
    text_output = response.choices[0].message.content
    try:
        natural_text = json.loads(text_output).get('natural_text', '')
        return convert_html_tables_to_markdown(natural_text)
    except:
        return convert_html_tables_to_markdown(text_output)

# --- HybridProcessor Class ---
class HybridProcessor:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        if MODEL is None: _initialize_dolphin_model()

    def run(self) -> list:
        all_pages_content = []
        try:
            print(f"Processing Hybrid on PDF: {self.pdf_path.name}")
            with fitz.open(self.pdf_path) as doc:
                print(f"📄 พบเอกสารทั้งหมด {doc.page_count} หน้า จะทำการประมวลผลทุกหน้า...")
                for page_num_zero_based in range(doc.page_count):
                    page_num_one_based = page_num_zero_based + 1
                    print(f"\n{'='*20} Processing Page {page_num_one_based}/{doc.page_count} {'='*20}")
                    page = doc.load_page(page_num_zero_based)
                    
                    pix = page.get_pixmap(dpi=200)
                    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    print(f" -- Page {page_num_one_based}: Stage 1 - Layout Analysis with Dolphin...")
                    figure_bboxes = _dolphin_layout_analysis(pil_image)
                    print(f" -- Page {page_num_one_based}: Found {len(figure_bboxes)} figure locations.")

                    print(f" -- Page {page_num_one_based}: Stage 2 - Full-Page OCR with Typhoon...")
                    buffered = io.BytesIO(); pil_image.save(buffered, format="PNG")
                    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    markdown_text = _typhoon_full_page_ocr(base64_image)
                    print(f" -- Page {page_num_one_based}: Received and processed markdown text from Typhoon.")

                    print(f" -- Page {page_num_one_based}: Stage 3 - Merging results...")
                    final_markdown_for_page = markdown_text
                    if figure_bboxes:
                        final_markdown_for_page += "\n\n--- Extracted Figures ---\n\n"
                        for i, bbox in enumerate(figure_bboxes):
                            cropped_image = pil_image.crop(tuple(bbox))
                            buffered = io.BytesIO(); cropped_image.save(buffered, format="PNG")
                            base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                            caption = f"Page {page_num_one_based} - Extracted Figure {i + 1}"
                            final_markdown_for_page += f"![{caption}](data:image/png;base64,{base64_data})\n\n"
                        print(f" -- Page {page_num_one_based}: Successfully embedded {len(figure_bboxes)} figures.")
                    
                    all_pages_content.append(final_markdown_for_page)
            return all_pages_content
        except Exception as e:
            import traceback; traceback.print_exc()
            return [f"An error occurred while processing {self.pdf_path.name}: {e}"]