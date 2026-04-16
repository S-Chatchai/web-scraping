import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

TARGET_FILE = 'infoquest_news.json'
BATCH_SIZE = 10

def process_and_update_source_file():
    try:
        with open(TARGET_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            articles = data.get("articles", [])
    except Exception as e:
        print(f"อ่านไฟล์ไม่สำเร็จ: {e}")
        return

    # ค้นหาข่าวที่ยังไม่มี sentiment 
    pending_articles = [a for a in articles if a.get("content") and "sentiment" not in a]
    
    if not pending_articles:
        print("🎉 วิเคราะห์ครบทุกข่าวแล้ว! ไม่มีอะไรต้องทำเพิ่ม")
        return

    print(f"ข่าวที่รอการวิเคราะห์: {len(pending_articles)} รายการ\n")

    def chunk_data(data_list, size):
        for i in range(0, len(data_list), size):
            yield data_list[i:i + size]

    batches = list(chunk_data(pending_articles, BATCH_SIZE))
    
    for batch_idx, current_batch in enumerate(batches, 1):
        print(f"[{batch_idx}/{len(batches)}] กำลังประมวลผล (จำนวน {len(current_batch)} ข่าว) ... ", end="", flush=True)

        # 🌟 จุดที่แก้ไข: สร้าง Tracking ID เพื่อให้จับคู่ง่ายขึ้น 100%
        batch_input_data = []
        article_map = {} 
        
        for i, a in enumerate(current_batch):
            temp_id = f"news_{i}"
            article_map[temp_id] = a # เก็บ Reference ของข่าวต้นฉบับไว้
            
            batch_input_data.append({
                "id": temp_id, # ส่งแค่ ID สั้นๆ ให้ AI ไม่ส่ง Link แล้ว
                "title": a.get("title"),
                "content": a.get("content")[:1000] 
            })

        prompt = f"""
        คุณคือนักวิเคราะห์ข่าวการลงทุน หน้าที่ของคุณคือวิเคราะห์ข่าวและตอบกลับมาเป็น JSON Array เท่านั้น
        
        ข้อมูลข่าว:
        {json.dumps(batch_input_data, ensure_ascii=False)}

        รูปแบบ JSON ที่ต้องการ (ห้ามเปลี่ยน id ที่ส่งไปเด็ดขาด):
        [
            {{
                "id": "ใส่ id ให้ตรงกับที่ส่งไป (เช่น news_0)",
                "sentiment": "Positive / Negative / Neutral",
                "confidence_level": "ตัวเลข 0-100",
                "reason": "เหตุผลสั้นๆ 1 บรรทัด",
                "summary": "สรุป 1-2 ประโยค"
            }}
        ]
        """

        max_retries = 3
        success = False
        batch_results = []
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash-lite', 
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1 # ปรับให้ต่ำลงอีกเพื่อให้ AI นิ่งขึ้น
                    )
                )
                
                batch_results = json.loads(response.text)
                success = True
                break # สำเร็จแล้วหลุดออกจากลูป retry
                
            except json.JSONDecodeError:
                print(f"\n   ⚠️ AI ตอบกลับไม่ใช่ JSON (ลองใหม่ครั้งที่ {attempt+1})")
                time.sleep(3)
            except Exception as api_error:
                print(f"\n   ⚠️ API Error (ครั้งที่ {attempt+1}): {api_error}")
                time.sleep(10)

        # 🌟 จุดที่แก้ไข: จับคู่ข้อมูลด้วย ID แทน Link
        if success:
            matched_count = 0
            for result in batch_results:
                result_id = result.get("id")
                
                if result_id in article_map: # ถ้าหา ID เจอ
                    article = article_map[result_id] # ดึงข่าวตัวจริงมาอัปเดต
                    article["sentiment"] = result.get("sentiment")
                    article["confidence_level"] = result.get("confidence_level")
                    article["reason"] = result.get("reason")
                    article["summary"] = result.get("summary")
                    matched_count += 1
                    
            if matched_count == 0:
                print("⚠️ AI ตอบมาแต่จับคู่ข้อมูลไม่ได้เลย (จะถูกนำไปรันใหม่รอบหน้า)")
            else:
                print(f"✅ สำเร็จ (อัปเดตและเซฟลงไฟล์แล้ว {matched_count}/{len(current_batch)} ข่าว)")
                
            # เซฟลงไฟล์ทับของเดิมทันที
            data["articles"] = articles 
            with open(TARGET_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
        else:
            print("❌ ข้าม Batch นี้เนื่องจาก Error ซ้ำซ้อน (อาจติด Filter ของ Google)")
            
        time.sleep(2) # พักหายใจก่อนส่ง Batch ถัดไป

if __name__ == "__main__":
    process_and_update_source_file()