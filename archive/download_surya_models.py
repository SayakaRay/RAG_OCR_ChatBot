# download_surya_models.py
import torch

DEVICE = "cpu"

def preload_all_surya_models():
    """
    ฟังก์ชันนี้จะทำการดาวน์โหลดโมเดลและ processor ทั้งหมดของ Surya
    ที่จำเป็นสำหรับการวิเคราะห์ Layout และการทำ OCR
    """
    print("Starting Surya models pre-loading...")
    print(f"Using device: {DEVICE}")

    try:
        # 1. โหลดโมเดลสำหรับ Layout Detection
        print("\n[1/4] Downloading Layout Detection Model...")
        from surya.model.detection.segformer import load_model as load_layout_model
        layout_model = load_layout_model(device=DEVICE)
        print("✅ Layout Detection Model downloaded.")

        # 2. โหลด Processor สำหรับ Layout Detection
        print("\n[2/4] Downloading Layout Detection Processor...")
        from surya.model.detection.segformer import load_processor as load_layout_processor
        layout_processor = load_layout_processor()
        print("✅ Layout Detection Processor downloaded.")

        # 3. โหลดโมเดลสำหรับ Text Recognition (OCR)
        print("\n[3/4] Downloading OCR Model...")
        from surya.model.recognition.model import load_model as load_ocr_model
        ocr_model = load_ocr_model(device=DEVICE)
        print("✅ OCR Model downloaded.")

        # 4. โหลด Processor สำหรับ Text Recognition (OCR)
        print("\n[4/4] Downloading OCR Processor...")
        from surya.model.recognition.processor import load_processor as load_ocr_processor
        ocr_processor = load_ocr_processor()
        print("✅ OCR Processor downloaded.")
        
        print("\n🎉 All Surya models have been successfully downloaded and cached locally.")

    except Exception as e:
        print(f"\n❌ An error occurred during download: {e}")

if __name__ == "__main__":
    preload_all_surya_models()