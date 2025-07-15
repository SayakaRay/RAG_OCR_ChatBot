from openai import OpenAI
from OCR.config import Settings

class ChatAssistant:
    def __init__(self):
        self.client = OpenAI(api_key=Settings.OPENAI_KEY)

    def ask(self, context:str, prompt:str) -> str:
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system",
                 "content":("ตอบคำถามจาก context ต่อไปนี้โดยดึงรูปภาพที่จำเป็นมาด้วย "
                            "โดยรหัสรูปภาพจะต้องอยู่ใน context เท่านั้น\n"
                            f"context:\n{context}")},
                {"role":"user","content":prompt}
            ],
            temperature=0.4
        )
        return resp.choices[0].message.content.strip()
