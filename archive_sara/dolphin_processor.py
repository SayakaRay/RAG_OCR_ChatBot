# OCR/dolphin_processor.py

import os
import io
import base64
from pathlib import Path
from PIL import Image
import torch
import cv2
import re
from transformers import AutoProcessor, VisionEncoderDecoderModel

# --- Helper Functions (ดัดแปลงจาก ByteDance_Dolphin_app.py) ---

def prepare_image(pil_image, padding_color=(255, 255, 255)):
    """ปรับขนาดและ Pad รูปภาพให้เป็นสี่เหลี่ยมจัตุรัส"""
    img_w, img_h = pil_image.size
    padded_img = Image.new('RGB', (max(img_w, img_h), max(img_w, img_h)), padding_color)
    padded_img.paste(pil_image, (0, 0))
    dims = (img_w, img_h, padded_img.width, padded_img.height)
    return padded_img, dims

def parse_layout_string(layout_string: str):
    """แปลง layout string เป็น list ของ (bbox, label)"""
    pattern = re.compile(r'(\w+)\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
    matches = pattern.findall(layout_string)
    return [(match[0], [int(c) for c in match[1:]]) for match in matches]

# --- DolphinProcessor Class ---

class DolphinProcessor:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_model()

    def _initialize_model(self):
        """โหลดโมเดล Dolphin และ Processor"""
        if self.model is None:
            print("[INFO] Initializing Dolphin Model (อาจใช้เวลาสักครู่ในครั้งแรก)...")
            model_id = "ByteDance/Dolphin"
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_id, trust_remote_code=True)
            self.model.eval()
            self.model.to(self.device)
            self.tokenizer = self.processor.tokenizer
            print(f"✅ Dolphin Model loaded on {self.device}")

    def _model_chat(self, prompt, image):
        """รัน inference ด้วยโมเดル Dolphin"""
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        prompts = prompt if is_batch else [prompt]

        batch_inputs = self.processor(images, return_tensors="pt", padding=True)
        batch_pixel_values = batch_inputs.pixel_values.to(self.device)
        
        prompts = [f"<s>{p} <Answer/>" for p in prompts]
        batch_prompt_inputs = self.tokenizer(prompts, add_special_tokens=False, return_tensors="pt")
        batch_prompt_ids = batch_prompt_inputs.input_ids.to(self.device)
        batch_attention_mask = batch_prompt_inputs.attention_mask.to(self.device)
        
        outputs = self.model.generate(
            pixel_values=batch_pixel_values,
            decoder_input_ids=batch_prompt_ids,
            decoder_attention_mask=batch_attention_mask,
            max_length=4096,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        sequences = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
        results = [seq.replace(p, "").replace("<pad>", "").replace("</s>", "").strip() for p, seq in zip(prompts, sequences)]
        return results if is_batch else results[0]

    def _process_elements(self, layout_results, padded_image, dims):
        """ประมวลผลแต่ละ element ที่ได้จาก Layout Analysis"""
        elements_to_process = {"text": [], "table": []}
        final_results = []
        reading_order = 0

        for label, bbox in layout_results:
            x1, y1, x2, y2 = bbox
            pil_crop = padded_image.crop((x1, y1, x2, y2))

            if label == "fig":
                buffered = io.BytesIO()
                pil_crop.save(buffered, format="PNG")
                base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                final_results.append({
                    "label": "figure",
                    "text": base64_data,
                    "reading_order": reading_order,
                })
            else:
                element_type = "table" if label == "tab" else "text"
                elements_to_process[element_type].append({
                    "crop": pil_crop,
                    "reading_order": reading_order
                })
            reading_order += 1
        
        # Batch processing for text and tables
        if elements_to_process["text"]:
            crops = [e["crop"] for e in elements_to_process["text"]]
            orders = [e["reading_order"] for e in elements_to_process["text"]]
            text_contents = self._model_chat(["Read text in the image."] * len(crops), crops)
            for i, content in enumerate(text_contents):
                final_results.append({"label": "text", "text": content, "reading_order": orders[i]})

        if elements_to_process["table"]:
            crops = [e["crop"] for e in elements_to_process["table"]]
            orders = [e["reading_order"] for e in elements_to_process["table"]]
            table_contents = self._model_chat(["Parse the table in the image."] * len(crops), crops)
            for i, content in enumerate(table_contents):
                final_results.append({"label": "table", "text": content, "reading_order": orders[i]})

        final_results.sort(key=lambda x: x["reading_order"])
        return final_results

    def run(self) -> list[dict]:
        """
        รัน Pipeline ของ Dolphin: Layout Analysis -> Content Recognition
        """
        try:
            print(f"Processing with Dolphin: {self.file_path.name}")
            pil_image = Image.open(self.file_path).convert("RGB")

            # --- STAGE 1: Layout Analysis ---
            print(" -- Stage 1: Layout Analysis...")
            layout_prompt = "Parse the reading order of this document."
            layout_string = self._model_chat(layout_prompt, pil_image)
            layout_elements = parse_layout_string(layout_string)
            print(f" -- Found {len(layout_elements)} elements.")

            # --- STAGE 2: Content Recognition ---
            print(" -- Stage 2: Content Recognition...")
            padded_image, dims = prepare_image(pil_image)
            recognition_results = self._process_elements(layout_elements, padded_image, dims)

            # --- Assemble Markdown ---
            markdown_parts = []
            for result in recognition_results:
                if result["label"] == "figure":
                    caption = f"Figure (element {result['reading_order']})"
                    markdown_parts.append(f"![{caption}](data:image/png;base64,{result['text']})")
                else:
                    markdown_parts.append(result["text"])
            
            final_markdown = "\n\n".join(markdown_parts)
            return [{"pages": [{"markdown": final_markdown}]}]

        except Exception as e:
            import traceback
            traceback.print_exc()
            return [{"error": str(e)}]