# OCR/main.py

import asyncio
from pathlib import Path
import traceback
import tempfile
from PIL import Image
import fitz
import os

from .storage import S3Storage
from .image_manager import ImageManager
from .hybrid_processor import HybridProcessor
from .powerpoint_processor import PowerPointProcessor

def _image_to_pdf_path(image_path: Path) -> Path | None:
    print(f"🔄 Converting image file {image_path.name} to PDF...")
    try:
        with Image.open(image_path) as img, tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf_temp_file:
            pdf_path = Path(pdf_temp_file.name)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(pdf_path)
            print(f"✅ Image converted to temporary PDF: {pdf_path}")
            return pdf_path
    except Exception as e:
        print(f"❌ Error converting image to PDF: {e}")
        return None

async def pipeline(FILE_PATH: str, filetype: str, username: str, project_id: str, original_filename: str) -> Path | None:
    path_to_process = None
    temp_files_to_clean = []
    pptx_processor = None
    try:
        original_path = Path(FILE_PATH)
        print(f"--- 📂 Step 1: Preparing file ({filetype}) ---")
        
        if filetype == "pptx":
            pptx_processor = PowerPointProcessor(original_path)
            path_to_process = pptx_processor.convert_to_pdf()
        elif filetype in ["jpg", "jpeg", "png"]:
            path_to_process = _image_to_pdf_path(original_path)
            if path_to_process: temp_files_to_clean.append(path_to_process)
        elif filetype == "pdf":
            path_to_process = original_path
        else:
            raise ValueError(f"Unsupported file type: {filetype}")
        
        if not path_to_process or not path_to_process.exists():
            print("❌ Could not prepare file for OCR, aborting."); return None
        
        print(f"\n--- 🔬 Step 2: Processing with Hybrid OCR (Text Extraction) ---")
        processor = HybridProcessor(path_to_process)
        def sync_run(): return processor.run()
        ocr_page_contents = await asyncio.to_thread(sync_run)
        
        print("\n--- 🖼️ Step 2.5: Processing Images and Uploading to S3 ---")
        s3_storage = S3Storage(os.getenv("AWS_ACCESS_KEY"), os.getenv("AWS_SECRET_KEY"), os.getenv("AWS_BUCKET_NAME"))
        image_manager = ImageManager(storage=s3_storage)
        image_ids_by_page = {}

        with fitz.open(path_to_process) as doc:
            for page_num in range(len(doc)):
                if filetype in ["jpg", "jpeg", "png"]:
                    image_ids = image_manager.upload_single_image(original_path, username, project_id)
                    image_ids_by_page[page_num] = image_ids
                    break 
                
                image_list = doc.get_page_images(page_num)
                if not image_list:
                    continue

                page_image_ids = []
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_temp_file:
                        img_temp_file.write(image_bytes)
                        temp_image_path = Path(img_temp_file.name)
                    
                    ids = image_manager.upload_single_image(temp_image_path, username, project_id)
                    page_image_ids.extend(ids)
                    temp_image_path.unlink()

                if page_image_ids:
                    image_ids_by_page[page_num] = page_image_ids
        
        final_full_text = []
        for i, page_text in enumerate(ocr_page_contents):
            final_full_text.append(page_text)
            if i in image_ids_by_page:
                for image_id in image_ids_by_page[i]:
                    markdown_tag = f"\n\n![Image from document]({image_id}.jpg)\n\n"
                    final_full_text.append(markdown_tag)
                    print(f"✅ Inserted image reference for page {i+1}: {markdown_tag.strip()}")

        print("\n--- 💾 Step 3: Saving result ---")
        output_dir = Path("./text_document")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(original_filename).stem}_output.txt"
        
        output_path.write_text("\n".join(final_full_text), encoding="utf-8")
        print(f"✅ OCR result (with image references) saved to: {output_path}")
        
        return output_path

    except Exception as e:
        print(f"❌ An unexpected error occurred in pipeline: {e}"); traceback.print_exc()
        return None
    finally:
        print("\n--- 🧹 Step 4: Cleaning up temporary files ---")
        if pptx_processor: pptx_processor.cleanup()
        for f in temp_files_to_clean:
            if f and f.exists(): f.unlink()
        print("✅ Cleanup complete.")