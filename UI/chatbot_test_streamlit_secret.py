import os
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from google import genai

# ==========================================
# 0. การตั้งค่า Configuration และ Environment
# ==========================================
st.set_page_config(page_title="AI Stock Assistant", page_icon="📈", layout="centered")

load_dotenv()

def get_config(key, default=None):
    """ดึงค่าจาก Streamlit Secrets ก่อน ถ้าไม่มีให้ไปดูใน Environment Variables"""
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

# ดึงค่า MongoDB URI
MONGO_URI = get_config("MONGODB_URI")

# ดึงค่า API Keys และจัดการให้อยู่ในรูป List เสมอ
raw_keys = get_config("GEMINI_API_KEYS", "")
if isinstance(raw_keys, list):
    API_KEYS = raw_keys
else:
    # กรณีเป็น String จาก .env (เช่น key1,key2)
    API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

if not API_KEYS or not MONGO_URI:
    st.error("❌ ไม่พบการตั้งค่า GEMINI_API_KEYS หรือ MONGODB_URI กรุณาตรวจสอบ Secrets หรือไฟล์ .env")
    st.stop()

# ตัวแปรจำตำแหน่งคีย์ปัจจุบันใน Session State
if "current_key_index" not in st.session_state:
    st.session_state.current_key_index = 0

# ==========================================
# 1. การจัดการ Cache ของ Database และ Model
# ==========================================
@st.cache_resource
def get_database_collection():
    """เชื่อมต่อ MongoDB แค่ครั้งเดียว"""
    client = MongoClient(MONGO_URI)
    return client["finance_db"]["news_articles"]

@st.cache_resource
def get_embedding_model():
    """โหลดโมเดล BAAI/bge-m3 แค่ครั้งเดียว"""
    return SentenceTransformer('BAAI/bge-m3')

# ==========================================
# 2. ฟังก์ชันหลักสำหรับ AI และ RAG
# ==========================================
def generate_content_with_retry(prompt):
    """ฟังก์ชันเรียก Gemini API แบบมีระบบหมุนเวียน API Keys (Key Rotation)"""
    max_retries = len(API_KEYS) 
    
    for attempt in range(max_retries):
        current_key = API_KEYS[st.session_state.current_key_index]
        
        try:
            client = genai.Client(api_key=current_key)
            # ใช้โมเดล gemini-2.0-flash (ปรับตามเวอร์ชันที่เปิดให้ใช้ล่าสุด)
            response = client.models.generate_content(
                model='gemini-2.0-flash', 
                contents=prompt
            )
            # หมุน Key ไปอันถัดไปเสมอเพื่อกระจาย Traffic
            st.session_state.current_key_index = (st.session_state.current_key_index + 1) % len(API_KEYS)
            return response.text
            
        except Exception as e:
            # หาก Key มีปัญหา ให้เลื่อนไปอันถัดไปทันทีและลองใหม่
            st.session_state.current_key_index = (st.session_state.current_key_index + 1) % len(API_KEYS)
            if attempt == max_retries - 1:
                raise Exception(f"❌ API Keys ทั้งหมดโควต้าเต็มหรือเกิดข้อผิดพลาด: {str(e)}")

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
    ticker_text = generate_content_with_retry(prompt)
    return ticker_text.strip().upper()

def query_rag_system(collection, embed_model, query):
    """รัน RAG Pipeline: Vector Search + Generative AI"""
    with st.status("กำลังประมวลผลข้อมูล...", expanded=True) as status:
        st.write("🔍 วิเคราะห์หุ้นจากคำถาม...")
        target_ticker = extract_ticker(query)
        
        filter_query = {}
        if target_ticker != "NONE" and len(target_ticker) < 10:
            st.write(f"🎯 กำหนดเป้าหมายการค้นหาหุ้น: **[{target_ticker}]**")
            filter_query = { "tickers": { "$in": [target_ticker] } }
        else:
            st.write("🌐 ค้นหาข้อมูลครอบคลุมภาพรวมตลาด")

        st.write("⏳ กำลังแปลงข้อความเป็น Vector...")
        query_vector = embed_model.encode(query, normalize_embeddings=True).tolist()

        # สร้าง Pipeline สำหรับ Atlas Vector Search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index", 
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 100,
                    "limit": 5
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
                "link": 1,
            }
        })

        st.write("🔍 ค้นหาข่าวที่เกี่ยวข้องจาก Database...")
        results = list(collection.aggregate(pipeline))

        if not results:
            status.update(label="ไม่พบข้อมูลที่เกี่ยวข้อง", state="error")
            return "❌ ขออภัย ไม่พบข่าวหรือข้อมูลที่เกี่ยวข้องกับคำถามของคุณในระบบฐานข้อมูล"

        st.write("🧠 AI กำลังสรุปคำตอบให้คุณ...")
        context_text = "\n\n".join(
            [f"หัวข้อ: {doc['title']}\nเนื้อหา: {doc.get('content', '')}\nลิงก์อ้างอิง: {doc.get('link', 'ไม่มีลิงก์')}" 
             for doc in results]
        )
        
        prompt = f"""
        คุณคือผู้ช่วยนักวิเคราะห์การลงทุน ตอบคำถามจากข้อมูลที่กำหนดให้เท่านั้น
        หากไม่มีคำตอบที่ชัดเจน ให้บอกว่า "ไม่พบข้อมูลที่แน่ชัด"

        กฎการตอบ:
        - สรุปให้กระชับ เข้าใจง่าย
        - **ต้องแนบลิงก์ที่มา (Link) ทุกครั้งที่อ้างอิงข้อมูล** (เช่น [อ่านต่อ]({results[0].get('link')}))
        - ใช้ภาษาทางการที่เป็นกันเอง

        [ข้อมูลอ้างอิง]:
        {context_text}

        [คำถามผู้ใช้]: 
        {query}
        """
        
        final_answer = generate_content_with_retry(prompt)
        status.update(label="วิเคราะห์เสร็จสมบูรณ์!", state="complete", expanded=False)
        
        return final_answer

# ==========================================
# 3. ส่วน UI Layout
# ==========================================
st.title("📈 AI Stock RAG Assistant")
st.caption("ระบบวิเคราะห์หุ้นไทยอัจฉริยะ (Powered by Gemini & MongoDB Vector Search)")

# Initialize ทรัพยากร
collection = get_database_collection()
embed_model = get_embedding_model()

# ระบบ Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "สวัสดีครับ! ผมเป็น AI วิเคราะห์ข่าวหุ้น คุณอยากทราบข้อมูลของหุ้นตัวไหน หรือสถานการณ์ตลาดเรื่องอะไร ถามมาได้เลยครับ"}
    ]

# วนลูปแสดงข้อความเก่า
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ส่วนรับ Input จากผู้ใช้
if user_question := st.chat_input("ตัวอย่าง: วิเคราะห์แนวโน้มหุ้น PTT หรือ ข่าวล่าสุดของ CPALL"):
    # บันทึกคำถามผู้ใช้
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # ให้ AI ประมวลผลคำตอบ
    with st.chat_message("assistant"):
        try:
            response = query_rag_system(collection, embed_model, user_question)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")