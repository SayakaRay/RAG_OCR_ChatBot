# OCR/main.py

import asyncio
from pathlib import Path
import traceback
import tempfile
from PIL import Image

from OCR.hybrid_processor import HybridProcessor
from OCR.powerpoint_processor import PowerPointProcessor

def _image_to_pdf_path(image_path: Path) -> Path | None:
    print(f"🔄 Converting image file {image_path.name} to PDF...")
    try:
        with Image.open(image_path) as img, tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf_temp_file:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(pdf_temp_file.name)
            print(f"✅ Image converted to temporary PDF: {pdf_temp_file.name}")
            return Path(pdf_temp_file.name)
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
        
        print(f"\n--- 🔬 Step 2: Processing with Hybrid OCR ---")
        processor = HybridProcessor(path_to_process)
        def sync_run(): return processor.run()
        ocr_page_contents = await asyncio.to_thread(sync_run)
        
        print("\n--- 💾 Step 3: Saving result ---")
        output_dir = Path("./text_document")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # --- จุดที่แก้ไข: ใช้ original_filename ในการตั้งชื่อ ---
        output_path = output_dir / f"{Path(original_filename).stem}_output.txt"
        
        output_path.write_text("\n\n--- Page Separator ---\n\n".join(ocr_page_contents), encoding="utf-8")
        print(f"✅ OCR result saved to: {output_path}")
        
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