import streamlit as st
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from google import genai

# ==========================================
# 0. ตั้งค่าหน้าเพจ Streamlit
# ==========================================
st.set_page_config(page_title="AI Stock Assistant", page_icon="📈", layout="centered")

# ดึงค่าจาก Streamlit Secrets
# หมายเหตุ: ใน Streamlit Secrets เราสามารถเก็บค่าเป็น List ได้โดยตรงในไฟล์ .toml
try:
    MONGO_URI = st.secrets["MONGODB_URI"]
    # ดึงค่า API_KEYS (ซึ่งแนะนำให้เขียนเป็นรูปแบบ List ใน Secrets)
    API_KEYS = st.secrets["GEMINI_API_KEYS"]
    
    # กรณีที่ใน Secrets เก็บเป็น String คั่นด้วย comma (เผื่อไว้)
    if isinstance(API_KEYS, str):
        API_KEYS = [key.strip() for key in API_KEYS.split(",") if key.strip()]

except KeyError:
    st.error("❌ ไม่พบการตั้งค่า MONGODB_URI หรือ GEMINI_API_KEYS ใน Streamlit Secrets")
    st.stop()

if not API_KEYS or not MONGO_URI:
    st.error("❌ ข้อมูลการตั้งค่าใน Secrets ไม่ครบถ้วน")
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
    """ฟังก์ชันเรียก Gemini API แบบมีระบบ Auto-Retry"""
    max_retries = len(API_KEYS) 
    
    for attempt in range(max_retries):
        current_key = API_KEYS[st.session_state.current_key_index]
        
        try:
            client = genai.Client(api_key=current_key)
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite', 
                contents=prompt
            )
            # เลื่อนไปใช้คีย์ถัดไปเพื่อกระจายโควต้า
            st.session_state.current_key_index = (st.session_state.current_key_index + 1) % len(API_KEYS)
            return response.text
            
        except Exception as e:
            st.session_state.current_key_index = (st.session_state.current_key_index + 1) % len(API_KEYS)
            if attempt == max_retries - 1:
                raise Exception(f"❌ API Keys ทั้งหมดโควต้าเต็มหรือเกิดข้อผิดพลาด: {str(e)}")

def extract_ticker(query):
    """ใช้ Gemini API วิเคราะห์ชื่อย่อหุ้น หรือคาดเดากลุ่มหุ้น"""
    prompt = f"""
    คุณคือระบบสกัดข้อมูล (Entity Extractor) ที่เชี่ยวชาญตลาดหุ้นไทย
    จงอ่านคำถามของผู้ใช้ แล้วดึง "ชื่อย่อหุ้น (Ticker)" ออกมา
    
    กฎ:
    1. ตอบเป็นชื่อย่อหุ้นภาษาอังกฤษตัวพิมพ์ใหญ่เท่านั้น
    2. หากมีหุ้นหลายตัว ให้คั่นด้วยเครื่องหมายจุลภาค (comma) เช่น PTT, SCC, KBANK
    3. **สำคัญมาก:** หากผู้ใช้ถามถึง "กลุ่มอุตสาหกรรม" หรือ "หมวดหมู่" ให้คุณดึงชื่อย่อหุ้นตัวท็อปๆ ในกลุ่มนั้นมา เช่น PTT, PTTEP, TOP, BCP
    4. หากเป็นชื่อภาษาไทย ให้แปลงเป็นชื่อย่อ (เช่น กรุงไทย -> KTB)
    5. หากไม่มีชื่อหุ้น และไม่สามารถตีความกลุ่มอุตสาหกรรมได้เลย ให้ตอบ "NONE"
    
    คำถาม: "{query}"
    ชื่อหุ้น:
    """
    ticker_text = generate_content_with_retry(prompt)
    return ticker_text.strip().upper()

def query_rag_system(collection, embed_model, query):
    """รัน RAG Pipeline และคืนค่าคำตอบกลับไปที่ UI"""
    with st.status("กำลังประมวลผลข้อมูล...", expanded=True) as status:
        st.write("🔍 สกัดชื่อหุ้นจากคำถาม...")
        extracted_text = extract_ticker(query)
        
        filter_query = {}
        search_query = query 
        
        if extracted_text != "NONE" and extracted_text != "":
            ticker_list = [t.strip() for t in extracted_text.split(",") if t.strip()]
            
            if ticker_list:
                st.write(f"🎯 กำหนดเป้าหมายการค้นหา: **[{', '.join(ticker_list)}]**")
                filter_query = { "tickers": { "$in": ticker_list } }
                search_query = f"{query} ข้อมูลของหุ้น {', '.join(ticker_list)}"
        else:
            st.write("🌐 ค้นหาข้อมูลจากข่าวการลงทุนทั้งหมด")

        st.write("⏳ กำลังสร้าง Vector ข้อมูล...")
        query_vector = embed_model.encode(search_query, normalize_embeddings=True).tolist()

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index", 
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 150, 
                    "limit": 10           
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

        st.write("🔍 กำลังค้นหาข้อมูลที่เกี่ยวข้องจาก Database...")
        results = list(collection.aggregate(pipeline))

        if not results:
            status.update(label="การค้นหาเสร็จสิ้น (ไม่พบข้อมูล)", state="error")
            return "❌ ไม่พบข้อมูลข่าวที่เกี่ยวข้องในระบบฐานข้อมูล"

        st.write("🧠 กำลังให้ AI สังเคราะห์คำตอบ...")
        context_text = "\n\n".join(
            [f"หัวข้อ: {doc['title']}\nเนื้อหา: {doc.get('content', '')}\nลิงก์อ้างอิง: {doc.get('link', 'ไม่มีลิงก์')}" 
             for doc in results]
        )
        
        prompt = f"""
        คุณคือผู้ช่วยนักวิเคราะห์การลงทุน ตอบคำถามจากข้อมูลที่กำหนดให้เท่านั้น
        หากต้องการเปรียบเทียบ ให้วิเคราะห์จุดเด่นและจุดด้อยของแต่ละฝั่งจากข้อมูลที่มีให้เห็นภาพชัดเจน
        หากไม่มีคำตอบ ให้บอกว่า "ไม่พบข้อมูลที่แน่ชัด"

        ข้อบังคับสำคัญ: 
        - ต้องสรุปคำตอบให้อ่านเข้าใจง่าย
        - **ต้องอ้างอิงลิงก์ที่มา (Link) แนบท้ายข้อมูลหรือข่าวที่คุณใช้ตอบเสมอ** (เช่น [อ่านเพิ่มเติม](https://...))

        [ข้อมูลอ้างอิง]:
        {context_text}

        [คำถามผู้ใช้]: 
        {query}
        """
        
        final_answer = generate_content_with_retry(prompt)
        status.update(label="ประมวลผลสำเร็จ!", state="complete", expanded=False)
        
        return final_answer

# ==========================================
# 3. ส่วนการออกแบบ UI (Streamlit Layout)
# ==========================================
st.title("📈 AI Stock RAG Assistant")
st.markdown("ระบบวิเคราะห์ข่าวสารและหุ้นไทยด้วยเทคโนโลยี AI Vector Search")

# โหลดทรัพยากร
collection = get_database_collection()
embed_model = get_embedding_model()

# เก็บประวัติการแชท
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "สวัสดีครับ ต้องการให้ช่วยหาข่าวหรือวิเคราะห์หุ้นตัวไหน พิมพ์ถามมาได้เลยครับ"}
    ]

# แสดงประวัติการสนทนาทั้งหมด
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# กล่องรับข้อความผู้ใช้งาน
if user_question := st.chat_input("พิมพ์คำถามของคุณที่นี่ เช่น 'ฉันอยากซื้อหุ้นน้ำมัน แนะนำตัวไหนบ้าง?'"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        try:
            response = query_rag_system(collection, embed_model, user_question)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")