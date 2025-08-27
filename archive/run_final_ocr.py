# run_final_ocr.py
# สคริปต์เดียวที่สมบูรณ์สำหรับวินิจฉัยและรัน Dolphin OCR Pipeline (เวอร์ชันสุดท้าย)

import sys
import os
import io
import base64
from pathlib import Path
from PIL import Image
import torch
import cv2
import re
import fitz  # PyMuPDF
from transformers import AutoProcessor, VisionEncoderDecoderModel
import tempfile

# --- GLOBAL VARIABLES FOR MODEL ---
MODEL = None
PROCESSOR = None
TOKENIZER = None
DEVICE = "cpu" # บังคับใช้ CPU เพื่อความแน่นอน

# --- HELPER FUNCTIONS ---

def _initialize_model():
    """โหลดโมเดล Dolphin และ Processor (ทำงานครั้งเดียว)"""
    global MODEL, PROCESSOR, TOKENIZER
    if MODEL is None:
        print("[INFO] Initializing Dolphin Model (อาจใช้เวลาสักครู่)...")
        model_id = "ByteDance/Dolphin"
        try:
            PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            MODEL = VisionEncoderDecoderModel.from_pretrained(model_id, trust_remote_code=True)
            MODEL.eval()
            MODEL.to(DEVICE)
            TOKENIZER = PROCESSOR.tokenizer
            print(f"[INFO] ✅ Dolphin Model loaded successfully on {DEVICE}")
        except Exception as e:
            print(f"[ERROR] ❌ Failed to load model: {e}")
            raise

def _prepare_image(pil_image, padding_color=(255, 255, 255)):
    """ปรับขนาดและ Pad รูปภาพให้เป็นสี่เหลี่ยมจัตุรัส"""
    img_w, img_h = pil_image.size
    padded_img = Image.new('RGB', (max(img_w, img_h), max(img_w, img_h)), padding_color)
    padded_img.paste(pil_image, (0, 0))
    return padded_img

# --- ฟังก์ชัน PARSER ที่เขียนขึ้นใหม่ทั้งหมด ---
def parse_layout_string_new(layout_string: str, image_dims: tuple):
    """
    แปลง layout string รูปแบบใหม่ที่ได้จาก Dolphin ให้อยู่ในรูปแบบที่ถูกต้อง
    พร้อมแปลงพิกัดสัดส่วน (relative) เป็นพิกัดจริง (absolute pixels)
    """
    img_w, img_h = image_dims
    print(f"[DEBUG] Parsing layout string with image dimensions: {img_w}x{img_h}")
    
    elements_str = re.split(r'\[(?:PAIR_SEP|RELATION_SEP)\]', layout_string)
    
    results = []
    pattern = re.compile(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\s*(\w+)')

    for elem_str in elements_str:
        match = pattern.match(elem_str.strip())
        if match:
            rel_coords = [float(c) for c in match.groups()[:4]]
            label = match.groups()[4]
            
            abs_x1 = int(rel_coords[0] * img_w)
            abs_y1 = int(rel_coords[1] * img_h)
            abs_x2 = int(rel_coords[2] * img_w)
            abs_y2 = int(rel_coords[3] * img_h)
            
            # แปลง label บางส่วนให้ตรงกับที่โค้ดคาดหวัง
            if label in ["header", "cap", "para", "sec", "foot", "list"]:
                label = "text"
            elif label == "fig":
                label = "figure" # แก้ไขจาก fig เป็น figure
            elif label == "tab":
                label = "table"

            results.append((label, [abs_x1, abs_y1, abs_x2, abs_y2]))
            
    return results

def _model_chat(prompt, image):
    # ... (โค้ดส่วนนี้เหมือนเดิม)
    is_batch = isinstance(image, list)
    images = image if is_batch else [image]
    prompts = prompt if is_batch else [prompt]
    batch_inputs = PROCESSOR(images, return_tensors="pt", padding=True)
    batch_pixel_values = batch_inputs.pixel_values.to(DEVICE)
    prompts = [f"<s>{p} <Answer/>" for p in prompts]
    batch_prompt_inputs = TOKENIZER(prompts, add_special_tokens=False, return_tensors="pt")
    batch_prompt_ids = batch_prompt_inputs.input_ids.to(DEVICE)
    batch_attention_mask = batch_prompt_inputs.attention_mask.to(DEVICE)
    outputs = MODEL.generate(pixel_values=batch_pixel_values, decoder_input_ids=batch_prompt_ids, decoder_attention_mask=batch_attention_mask, max_length=4096, pad_token_id=TOKENIZER.pad_token_id, eos_token_id=TOKENIZER.eos_token_id, use_cache=True, bad_words_ids=[[TOKENIZER.unk_token_id]], return_dict_in_generate=True,)
    sequences = TOKENIZER.batch_decode(outputs.sequences, skip_special_tokens=False)
    results = [seq.replace(p, "").replace("<pad>", "").replace("</s>", "").strip() for p, seq in zip(prompts, sequences)]
    return results if is_batch else results[0]

# --- MAIN PIPELINE FUNCTION ---

def run_full_pipeline(file_path_str: str):
    print(f"\n--- 📂 STEP 1: PREPARING FILE ---")
    file_path = Path(file_path_str)
    if not file_path.exists():
        print(f"[ERROR] ❌ File not found at: {file_path}"); return
    temp_file_to_clean = None
    try:
        if file_path.suffix.lower() == ".pdf":
            print(f"[INFO] PDF detected. Converting first page to PNG...")
            with fitz.open(file_path) as doc, tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_f:
                page = doc.load_page(0); pix = page.get_pixmap(dpi=200); pix.save(temp_f.name)
                image_path = Path(temp_f.name); temp_file_to_clean = image_path
        elif file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            print(f"[INFO] Image file detected. Ensuring it's a usable PNG...")
            with Image.open(file_path) as img, tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_f:
                if img.mode != "RGB": img = img.convert("RGB")
                img.save(temp_f.name, "PNG"); image_path = Path(temp_f.name); temp_file_to_clean = image_path
        else:
            print(f"[ERROR] ❌ Unsupported file type: {file_path.suffix}"); return
        
        print(f"[INFO] ✅ File prepared for processing at: {image_path}")
        
        # --- STAGE 1: LAYOUT ANALYSIS ---
        print(f"\n--- 🔬 STAGE 1: LAYOUT ANALYSIS ---")
        pil_image = Image.open(image_path).convert("RGB")
        layout_prompt = "Parse the reading order of this document."
        print(f"[INFO] Sending full page image to Dolphin for layout analysis...")
        layout_string = _model_chat(layout_prompt, pil_image)
        print(f"[DEBUG] Raw Layout String from Model:\n{layout_string}")
        
        layout_elements = parse_layout_string_new(layout_string, pil_image.size)
        
        if not layout_elements:
            print("[ERROR] ❌ Failed to parse layout from model output. Cannot proceed."); return
        print(f"[INFO] ✅ Layout analysis successful. Found {len(layout_elements)} elements.")
        
        # --- STAGE 2: CONTENT RECOGNITION ---
        print(f"\n--- 🔬 STAGE 2: CONTENT RECOGNITION ---")

        # --- จุดที่แก้ไข ---
        padded_image = _prepare_image(pil_image) # <--- แก้ไขที่บรรทัดนี้ (ลบ [0] ออก)
        
        markdown_parts = []
        final_results = []
        reading_order = 0

        for label, bbox in layout_elements:
            pil_crop = padded_image.crop(tuple(bbox))
            if label == "figure":
                buffered = io.BytesIO(); pil_crop.save(buffered, format="PNG")
                base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                final_results.append({"label": "figure", "text": base64_data, "reading_order": reading_order})
            else:
                prompt = "Parse the table in the image." if label == "table" else "Read text in the image."
                content = _model_chat(prompt, pil_crop)
                final_results.append({"label": label, "text": content, "reading_order": reading_order})
            reading_order += 1
            print(f"  -> Processed element {reading_order}/{len(layout_elements)} ({label})")

        print(f"[INFO] ✅ Content recognition successful for all elements.")
        
        # --- FINAL STEP: ASSEMBLE & SAVE ---
        print(f"\n--- 💾 FINAL STEP: ASSEMBLING AND SAVING ---")
        final_results.sort(key=lambda x: x["reading_order"])
        for result in final_results:
            if result["label"] == "figure":
                # แก้ไข caption ให้มีความหมายมากขึ้น
                markdown_parts.append(f"![Figure extracted at element {result['reading_order']}](data:image/png;base64,{result['text']})")
            else:
                markdown_parts.append(result["text"])
        final_output_text = "\n\n".join(markdown_parts)
        output_dir = Path("./text_document"); output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{file_path.stem}_output.txt"
        output_path.write_text(final_output_text, encoding="utf-8")
        print(f"[SUCCESS] 🎉 Successfully saved OCR result to: {output_path}")

    except Exception as e:
        print(f"[FATAL ERROR] ❌ An unexpected error occurred in the pipeline."); import traceback; traceback.print_exc()
    finally:
        if temp_file_to_clean and temp_file_to_clean.exists():
            temp_file_to_clean.unlink(); print(f"[INFO] 🧹 Cleaned up temporary file: {temp_file_to_clean.name}")

# --- SCRIPT EXECUTION ---
if __name__ == "__main__":
    _initialize_model()
    if len(sys.argv) > 1:
        run_full_pipeline(sys.argv[1])
    else:
        print("\n[Usage] Please provide a file path as an argument."); sys.exit(1)