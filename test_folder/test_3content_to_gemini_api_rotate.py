import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# โหลด API Keys ทั้งหมดจาก .env และแยกเป็น List
API_KEYS_STR = os.getenv("GEMINI_API_KEYS", "")
API_KEYS = [k.strip() for k in API_KEYS_STR.split(",") if k.strip()]

if not API_KEYS:
    print("❌ ไม่พบ API Keys ในไฟล์ .env กรุณากำหนด GEMINI_API_KEYS เช่น key1,key2")
    exit(1)

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

    print(f"ข่าวที่รอการวิเคราะห์: {len(pending_articles)} รายการ")
    print(f"🔑 จำนวน API Keys ที่พร้อมใช้งาน: {len(API_KEYS)} Keys\n")

    def chunk_data(data_list, size):
        for i in range(0, len(data_list), size):
            yield data_list[i:i + size]

    batches = list(chunk_data(pending_articles, BATCH_SIZE))
    
    for batch_idx, current_batch in enumerate(batches, 1):
        # 🌟 จุดที่แก้ไข: สลับ Key ใหม่ทุกๆ Batch 
        # ใช้ (batch_idx - 1) % len(API_KEYS) เพื่อให้วนลูป 0, 1, 2 -> 0, 1, 2
        current_key_idx = (batch_idx - 1) % len(API_KEYS)
        client = genai.Client(api_key=API_KEYS[current_key_idx])
        
        print(f"[{batch_idx}/{len(batches)}] กำลังประมวลผลด้วย Key ที่ {current_key_idx + 1} (จำนวน {len(current_batch)} ข่าว) ... ", end="", flush=True)

        batch_input_data = []
        article_map = {} 
        
        for i, a in enumerate(current_batch):
            temp_id = f"news_{i}"
            article_map[temp_id] = a 
            
            batch_input_data.append({
                "id": temp_id, 
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
                        temperature=0.1
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
                
                # 🌟 ระบบสำรอง: ถ้าสมมติว่า Key ที่ใช้อยู่พังระหว่างทางจริงๆ ก็หมุน Key หนีไปอีกที
                current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                print(f"   🔄 ฉุกเฉิน: สลับไปใช้ Key ที่ {current_key_idx + 1}")
                client = genai.Client(api_key=API_KEYS[current_key_idx])
                time.sleep(5) 

        if success:
            matched_count = 0
            for result in batch_results:
                result_id = result.get("id")
                
                if result_id in article_map:
                    article = article_map[result_id]
                    article["sentiment"] = result.get("sentiment")
                    article["confidence_level"] = result.get("confidence_level")
                    article["reason"] = result.get("reason")
                    article["summary"] = result.get("summary")
                    matched_count += 1
                    
            if matched_count == 0:
                print("⚠️ AI ตอบมาแต่จับคู่ข้อมูลไม่ได้เลย (จะถูกนำไปรันใหม่รอบหน้า)")
            else:
                print(f"✅ สำเร็จ (อัปเดตและเซฟลงไฟล์แล้ว {matched_count}/{len(current_batch)} ข่าว)")
                
            data["articles"] = articles 
            with open(TARGET_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
        else:
            print("❌ ข้าม Batch นี้เนื่องจาก Error ซ้ำซ้อนแม้จะเปลี่ยน Key แล้วก็ตาม")
            
        time.sleep(2) # พักหายใจก่อนส่ง Batch ถัดไป

if __name__ == "__main__":
    process_and_update_source_file()