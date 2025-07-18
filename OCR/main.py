import asyncio, re
from pathlib import Path
from io import BytesIO
from PIL import Image as PILImage

# Fix import paths for OCR modules
from OCR.config import Settings
from OCR.pdf_processor import PDFProcessor
from OCR.powerpoint_processor import PowerPointProcessor
from OCR.image_processor import ImageProcessor
from OCR.image_manager import ImageManager
from OCR.storage import S3Storage
from OCR.markdown_utils import replace_image_links, render_with_s3
from OCR.chat_assistant import ChatAssistant
import json
# PDF_PATH = Path("horce_test_table.pdf")

def preview_markdown_in_terminal(md_text, get_binary_fn):
    print("\n=========== 📄 MARKDOWN RESPONSE ===========\n")
    # พิมพ์ markdown แบบ raw (แทนการใช้ display())
    print(md_text)

    print("\n=========== 🖼️ IMAGE PREVIEW ===========\n")
    pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    for match in pattern.finditer(md_text):
        image_url = match.group(1)
        fname = image_url.split("/")[-1]
        image_id = fname.rsplit(".", 1)[0]

        try:
            print(f"🔍 Opening image: {fname} (ID: {image_id})")
            binary = get_binary_fn(image_id)
            img = PILImage.open(BytesIO(binary))
            img.show(title=fname)  # เปิดด้วย image viewer จริง
        except Exception as e:
            print(f"Failed to preview image: {fname} — {e}")

async def pipeline(FILE_PATH, filetype):

    if filetype == "pdf":
        pdf_proc = PDFProcessor(FILE_PATH)
        ocr_results = await pdf_proc.run()


        storage = S3Storage(Settings.AWS_ACCESS, Settings.AWS_SECRET,
                            Settings.AWS_BUCKET, Settings.AWS_REGION)
        img_mgr  = ImageManager(storage)

        full_md = []
        for resp in ocr_results:
            for page in resp["pages"]:
                md = page["markdown"]
                img_tags = re.findall(r'!\[img-\d+\.jpeg\]\(img-\d+\.jpeg\)', md)
                for idx, (img, old_tag) in enumerate(zip(page["images"], img_tags)):
                    local_path = img_mgr.save_local(img["image_base64"])
                    annotation = json.loads(img["image_annotation"])
                    description = annotation["description"]

                    new_tag = f"![Image: {local_path.name}]({local_path.name}){description}"
                    md = md.replace(old_tag, new_tag)
                full_md.append(md)

        output_path = Path(__file__).parent.parent / "text_document" / "all_pdf.txt"
        output_path.write_text("\n\n".join(full_md), encoding="utf-8")


        uploaded_ids = img_mgr.upload_folder()
        print(f"Uploaded {len(uploaded_ids)} images to S3")
    
    elif filetype == "pptx":
        # print("PPTX processing is not implemented yet.")
        pptx_proc = PowerPointProcessor(FILE_PATH)
        ocr_results = await pptx_proc.run()


        storage = S3Storage(Settings.AWS_ACCESS, Settings.AWS_SECRET,
                            Settings.AWS_BUCKET, Settings.AWS_REGION)
        img_mgr  = ImageManager(storage)

        full_md = []
        for resp in ocr_results:
            for page in resp["pages"]:
                md = page["markdown"]
                img_tags = re.findall(r'!\[img-\d+\.jpeg\]\(img-\d+\.jpeg\)', md)
                for idx, (img, old_tag) in enumerate(zip(page["images"], img_tags)):
                    local_path = img_mgr.save_local(img["image_base64"])
                    annotation = json.loads(img["image_annotation"])
                    description = annotation["description"]

                    new_tag = f"![Image: {local_path.name}]({local_path.name}){description}"
                    md = md.replace(old_tag, new_tag)
                full_md.append(md)

        output_path = Path(__file__).parent.parent / "text_document" / "all_pdf.txt"
        output_path.write_text("\n\n".join(full_md), encoding="utf-8")


        uploaded_ids = img_mgr.upload_folder()
        print(f"Uploaded {len(uploaded_ids)} images to S3")

    elif filetype in ['jpg', 'jpeg', 'png']:
        img_proc = ImageProcessor(FILE_PATH)
        ocr_results = await img_proc.run()

        storage = S3Storage(Settings.AWS_ACCESS, Settings.AWS_SECRET,
                            Settings.AWS_BUCKET, Settings.AWS_REGION)
        img_mgr = ImageManager(storage)

        full_md = []
        for resp in ocr_results:
            for page in resp["pages"]:
                md = page["markdown"]
                img_tags = re.findall(r'!\[img-\d+\.jpeg\]\(img-\d+\.jpeg\)', md)
                for idx, (img, old_tag) in enumerate(zip(page["images"], img_tags)):
                    local_path = img_mgr.save_local(img["image_base64"])
                    annotation = json.loads(img["image_annotation"])
                    description = annotation["description"]

                    new_tag = f"![Image: {local_path.name}]({local_path.name}){description}"
                    md = md.replace(old_tag, new_tag)
                full_md.append(md)

        output_path = Path(__file__).parent.parent / "text_document" / "all_pdf.txt"
        output_path.write_text("\n\n".join(full_md), encoding="utf-8")

        uploaded_ids = img_mgr.upload_folder()
        print(f"Uploaded {len(uploaded_ids)} images to S3")

    # assistant = ChatAssistant()
    # answer_md = assistant.ask("\n\n".join(full_md),
    #                           "อธิบาย flow การผลิตโซเดียมไฮดรอกไซด์มีรูปประกอบด้วย")

    # #สำหรับ display จริงบน ipynb หรือเว็บแอป
    # render_with_s3(answer_md, storage)


    # #สำหรับ ทดสอบดูผลลัพธ์ชั่วคราว
    # # ยืนยันว่า markdown + รูปมัน "มาจริง" ผ่าน terminal
    # preview_markdown_in_terminal(
    #     answer_md,
    #     lambda image_id: storage.get_image_binary(image_id)
    # )

# if __name__ == "__main__":
#     asyncio.run(pipeline(PDF_PATH))
