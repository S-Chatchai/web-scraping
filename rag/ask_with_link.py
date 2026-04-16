import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# โหลด Environment Variables จากไฟล์ .env
load_dotenv()

# ==========================================
# 0. การจัดการตั้งค่าและ API Rotation พร้อม Retry
# ==========================================
MONGO_URI = os.getenv("MONGODB_URI")
GEMINI_KEYS_ENV = os.getenv("GEMINI_API_KEYS", "")

# แปลง String จาก .env เป็น List
API_KEYS = [key.strip() for key in GEMINI_KEYS_ENV.split(",") if key.strip()]

if not API_KEYS or not MONGO_URI:
    raise ValueError("❌ ไม่พบการตั้งค่า GEMINI_API_KEYS หรือ MONGODB_URI ในไฟล์ .env")

# ตัวแปรจำตำแหน่งคีย์ปัจจุบัน
current_key_index = 0

def generate_content_with_retry(prompt):
    """
    ฟังก์ชันเรียก Gemini API โดยมีระบบ Auto-Retry สลับคีย์เมื่อเจอ Error 429
    """
    global current_key_index
    max_retries = len(API_KEYS) 
    
    for attempt in range(max_retries):
        key_num = current_key_index + 1
        current_key = API_KEYS[current_key_index]
        
        print(f"🔄 กำลังใช้ Key ที่ {key_num} (ลงท้ายด้วย ***{current_key[-4:]})")
        
        try:
            client = genai.Client(api_key=current_key)
            
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite', 
                contents=prompt
            )
            
            current_key_index = (current_key_index + 1) % len(API_KEYS)
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Key ที่ {key_num} เกิดข้อผิดพลาด: {error_msg.splitlines()[0]}")
            print(f"⏩ กำลังสลับไปใช้คีย์ถัดไป...")
            
            current_key_index = (current_key_index + 1) % len(API_KEYS)
            
    raise Exception("❌ API Keys ทั้งหมดโควต้าเต็มหรือเกิดข้อผิดพลาด ไม่สามารถสร้างคำตอบได้")

# ==========================================
# 1. ฟังก์ชันให้ AI สกัดชื่อหุ้น (Entity Extraction)
# ==========================================
def extract_ticker(query):
    """ใช้ Gemini API วิเคราะห์ชื่อย่อหุ้น"""
    prompt = f"""
    คุณคือระบบสกัดข้อมูล (Entity Extractor) ที่เชี่ยวชาญตลาดหุ้นไทย
    จงอ่านคำถามของผู้ใช้ แล้วดึง "ชื่อย่อหุ้น (Ticker)" ออกมา
    
    กฎ:
    1. ตอบเป็นชื่อย่อหุ้นภาษาอังกฤษตัวพิมพ์ใหญ่เท่านั้น (เช่น PTT, KTB)
    2. หากเป็นชื่อภาษาไทย ให้แปลงเป็นชื่อย่อ (เช่น กรุงไทย -> KTB)
    3. หากไม่มีชื่อหุ้น ให้ตอบ "NONE"
    
    คำถาม: "{query}"
    ชื่อหุ้น:
    """
    
    print("\n[ขั้นตอนที่ 1: สกัดชื่อหุ้น]")
    ticker_text = generate_content_with_retry(prompt)
    return ticker_text.strip().upper()

# ==========================================
# 2. ฟังก์ชัน RAG ค้นหาและตอบคำถาม (ดึง Link)
# ==========================================
def query_rag_system_auto(collection, embed_model, query):
    print("-" * 60)
    print(f"🙋‍♂️ คำถามผู้ใช้: '{query}'")
    
    # 2.1 สกัดชื่อหุ้น
    target_ticker = extract_ticker(query)
    
    filter_query = {}
    if target_ticker != "NONE" and len(target_ticker) < 10:
        print(f"🎯 Filter: [{target_ticker}]")
        filter_query = { "tickers": { "$in": [target_ticker] } }
    else:
        print("🌐 Mode: ค้นหาจากข่าวทั้งหมด")

    # 2.2 สร้าง Vector สำหรับค้นหา
    print("⏳ สร้าง Vector จากคำถาม...")
    query_vector = embed_model.encode(query, normalize_embeddings=True).tolist()

    # 2.3 Vector Search Pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index", 
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 3
            }
        }
    ]
    
    if filter_query:
        pipeline[0]["$vectorSearch"]["filter"] = filter_query
        
    pipeline.append({
        "$project": {
            "_id": 0, 
            "title": 1, 
            "content": 1,
            "link": 1  # 🔴 เพิ่มการดึง field link ออกมาจากฐานข้อมูล
        }
    })

    print("🔍 ค้นหาข้อมูลในฐานข้อมูล...")
    results = list(collection.aggregate(pipeline))

    if not results:
        print("❌ ไม่พบข้อมูลข่าวที่เกี่ยวข้อง")
        return

    # 2.4 สังเคราะห์คำตอบพร้อมแทรกลิงก์ลงใน Context
    # 🔴 เพิ่มการแนบ Link ลงไปให้ AI อ่าน
    context_text = "\n\n".join(
        [f"หัวข้อ: {doc['title']}\nเนื้อหา: {doc.get('content', '')}\nลิงก์อ้างอิง: {doc.get('link', 'ไม่มีลิงก์')}" 
         for doc in results]
    )
    
    # 🔴 ปรับ Prompt สั่งให้ AI แนบลิงก์เสมอ
    prompt = f"""
    คุณคือผู้ช่วยนักวิเคราะห์การลงทุน ตอบคำถามจากข้อมูลที่กำหนดให้เท่านั้น
    หากไม่มีคำตอบ ให้บอกว่า "ไม่พบข้อมูลที่แน่ชัด"

    ข้อบังคับสำคัญ: 
    - ต้องสรุปคำตอบให้อ่านเข้าใจง่าย
    - **ต้องอ้างอิงลิงก์ที่มา (Link) แนบท้ายข้อมูลหรือข่าวที่คุณใช้ตอบเสมอ** (เช่น [อ้างอิง: https://...])

    [ข้อมูลอ้างอิง]:
    {context_text}

    [คำถามผู้ใช้]: 
    {query}
    """
    
    print("\n[ขั้นตอนที่ 2: สังเคราะห์คำตอบด้วย RAG]")
    final_answer = generate_content_with_retry(prompt)
    
    print("\n" + "=" * 60)
    print("🤖 บทวิเคราะห์ RAG:")
    print(final_answer)
    print("=" * 60 + "\n")

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    USER_QUESTION = "ฉันอยากซื้อหุ้นน้ำมัน แนะนำตัวไหนบ้าง"

    try:
        # เชื่อมต่อ MongoDB
        db_client = MongoClient(MONGO_URI)
        collection = db_client["finance_db"]["news_articles"]
        
        # โหลดโมเดล Embedding
        print("⏳ Loading Embedding Model...")
        embed_model = SentenceTransformer('BAAI/bge-m3')

        # รันระบบ
        query_rag_system_auto(collection, embed_model, USER_QUESTION)
        
    except Exception as e:
        print(f"\n❌ Error หลักของระบบ: {e}")

if __name__ == "__main__":
    main()