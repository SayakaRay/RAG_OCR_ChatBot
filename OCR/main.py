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
import traceback
import json

def preview_markdown_in_terminal(md_text, get_binary_fn):
    print("\n=========== 📄 MARKDOWN RESPONSE ===========\n")
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

async def pipeline(FILE_PATH, filetype, username, project_id):
    try:
        # Select processor based on file type
        if filetype == "pdf":
            processor = PDFProcessor(FILE_PATH)
        elif filetype == "pptx":
            processor = PowerPointProcessor(FILE_PATH)
        elif filetype in ["jpg", "jpeg", "png"]:
            processor = ImageProcessor(FILE_PATH)
        else:
            raise ValueError(f"Unsupported file type: {filetype}")

        # OCR TIMEEEEEEEEEEE
        try:
            ocr_results = await processor.run()
        except Exception as e:
            print(f"❌ OCR processing failed: {e}")
            traceback.print_exc()
            return

        # Init storage + image manager
        try:
            storage = S3Storage(Settings.AWS_ACCESS, Settings.AWS_SECRET, Settings.AWS_BUCKET, Settings.AWS_REGION)
            img_mgr = ImageManager(storage)
        except Exception as e:
            print(f"❌ Failed to initialize S3/ImageManager: {e}")
            return

        # Process OCR results and replace image tags
        full_md = []
        for resp in ocr_results:
            for page in resp.get("pages", []):
                try:
                    md = page.get("markdown", "")
                    images = page.get("images", [])
                    img_tags = re.findall(r'!\[img-\d+\.jpeg\]\(img-\d+\.jpeg\)', md)

                    for img, old_tag in zip(images, img_tags):
                        local_path = img_mgr.save_local(img["image_base64"])
                        annotation = json.loads(img["image_annotation"])
                        description = annotation.get("description", "")
                        new_tag = f"![Image: {local_path.name}]({local_path.name}){description}"
                        md = md.replace(old_tag, new_tag)

                    full_md.append(md)
                except Exception as e:
                    print(f"⚠️ Error processing OCR page: {e}")
                    continue

        # Save markdown output
        try:
            output_path = Path(__file__).parent.parent / "text_document" / "ocr_output.txt"
            output_path.write_text("\n\n".join(full_md), encoding="utf-8")
        except Exception as e:
            print(f"❌ Failed to save markdown to file: {e}")
            return

        # Upload images to S3
        try:
            uploaded_ids = img_mgr.upload_folder(username, project_id)
            print(f"✅ Uploaded {len(uploaded_ids)} images to S3")
        except Exception as e:
            print(f"❌ Failed to upload images to S3: {e}")
            return

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        traceback.print_exc()


# if __name__ == "__main__":
#     asyncio.run(pipeline(PDF_PATH))
