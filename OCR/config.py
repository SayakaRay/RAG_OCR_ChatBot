from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
IMAGE_DIR   = PROJECT_ROOT / "images"
# print(f"Image directory set to: {IMAGE_DIR}")

# print(Path(__file__).parent.parent / "text_document")


class Settings:
    # OCR & LLM
    MISTRAL_KEY  = os.getenv("MISTRAL_API_KEY")
    OPENAI_KEY   = os.getenv("GPT_4O_MINI")

    # AWS
    AWS_ACCESS   = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET   = os.getenv("AWS_SECRET_KEY")
    AWS_BUCKET   = os.getenv("AWS_BUCKET_NAME")
    AWS_REGION   = "ap-southeast-2"
