# test_api_v2.py (เวอร์ชันแก้ไข Typo ของ getenv)

import os
import base64
import json
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI, APIError
from PIL import Image, ImageDraw, ImageFont

print("--- Starting API Connection Test v2 (Corrected) ---")

# 1. โหลด API Key
load_dotenv()
# --- จุดที่แก้ไข ---
api_key = os.getenv("TYPHOON_API_KEY") # <--- แก้ไขชื่อ Key ให้ถูกต้อง (มี O 2 ตัว)

if not api_key:
    print("❌ FATAL: ไม่พบ TYPHOON_API_KEY ในไฟล์ .env")
else:
    print("✅ API Key loaded successfully.")

    # 2. สร้างรูปภาพทดสอบขึ้นมาใหม่ (ขนาด 200x50)
    try:
        print("Creating a test image...")
        img = Image.new('RGB', (200, 50), color = (255, 255, 255))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        d.text((10,10), "Test", fill=(0,0,0), font=font)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        realistic_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        print("✅ Test image created successfully.")

    except Exception as e:
        print(f"❌ Could not create a test image. Make sure Pillow is installed (`pip install Pillow`). Error: {e}")
        realistic_image_base64 = None

    if realistic_image_base64:
        model_to_test = "typhoon-ocr-preview"
        
        print(f"Attempting to call model: '{model_to_test}' with a realistic image...")

        try:
            # 3. สร้าง Client และเรียก API
            client = OpenAI(
                base_url="https://api.opentyphoon.ai/v1",
                api_key=api_key
            )
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read the text in the image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{realistic_image_base64}"}}
                ]
            }]

            print("Sending request to API...")
            response = client.chat.completions.create(
                model=model_to_test,
                messages=messages,
                max_tokens=50
            )
            
            # 4. แสดงผลลัพธ์หากสำเร็จ
            print("\n--- ✅ API Call Successful! ---")
            print("Response object:")
            print(response)
            
            content = response.choices[0].message.content
            print("\nExtracted content:")
            print(content)

        except APIError as e:
            # 5. แสดง Error หากล้มเหลว
            print("\n--- ❌ API Call Failed! ---")
            print(f"Status Code: {e.status_code}")
            print(f"Error Type: {e.type}")
            print(f"Error Body: {e.body}")
            
        except Exception as e:
            print(f"\n--- ❌ An unexpected error occurred ---")
            print(e)