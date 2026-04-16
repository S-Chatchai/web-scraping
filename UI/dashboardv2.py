import os
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv

# --- 1. ตั้งค่าหน้าเพจ Streamlit ---
st.set_page_config(page_title="Thai Stock News Dashboard", page_icon="📈", layout="wide")
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

# --- 2. ฟังก์ชันดึงข้อมูลจาก MongoDB (พร้อมทำ Caching) ---
@st.cache_data(ttl=300) # รีเฟรชข้อมูลใหม่ทุกๆ 5 นาที
def load_data():
    if not MONGODB_URI:
        st.error("❌ ไม่พบ MONGODB_URI ในไฟล์ .env")
        st.stop()
        
    client = MongoClient(MONGODB_URI)
    db = client["finance_db"]
    
    # ดึงเฉพาะข่าวที่ผ่านการวิเคราะห์ AI แล้ว
    cursor = db["news_articles"].find({"sentiment": {"$exists": True}})
    df = pd.DataFrame(list(cursor))
    
    if not df.empty:
        df['_id'] = df['_id'].astype(str) 
        # แปลงข้อมูล list ในคอลัมน์ tickers ให้เป็น string คั่นด้วยลูกน้ำ
        df['tickers_str'] = df.get('tickers', []).apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
    return df

# โหลดข้อมูล
df = load_data()

if df.empty:
    st.warning("⚠️ ยังไม่มีข้อมูลข่าวที่ถูกวิเคราะห์ใน Database กรุณารันโค้ดขั้นตอนที่ 3 ก่อน")
    st.stop()

# --- 3. ส่วน UI: แถบด้านข้าง (Sidebar Filters) ---
st.sidebar.header("🔍 กรองข้อมูล (Filters)")

# 🌟 ฟีเจอร์ใหม่: ช่องค้นหาชื่อหุ้น (Text Input)
search_ticker = st.sidebar.text_input("🔎 ค้นหาชื่อหุ้น (เช่น PTT, KBANK)")

# Filter 1: Sentiment
sentiment_options = df['sentiment'].dropna().unique().tolist()
selected_sentiments = st.sidebar.multiselect("อารมณ์ของข่าว (Sentiment)", sentiment_options, default=sentiment_options)

# Filter 2: Impact Level
impact_options = df['impact_level'].dropna().unique().tolist()
selected_impacts = st.sidebar.multiselect("ระดับผลกระทบ (Impact Level)", impact_options, default=impact_options)

# Filter 3: Sector
sector_options = df['sector'].dropna().unique().tolist()
selected_sectors = st.sidebar.multiselect("กลุ่มอุตสาหกรรม (Sector)", sector_options, default=sector_options)

# นำ Filter หมวดหมู่มากรอง DataFrame พื้นฐานก่อน
filtered_df = df[
    (df['sentiment'].isin(selected_sentiments)) &
    (df['impact_level'].isin(selected_impacts)) &
    (df['sector'].isin(selected_sectors))
]

# 🌟 ฟีเจอร์ใหม่: นำคำค้นหาชื่อหุ้นมากรอง DataFrame เพิ่มเติม (ถ้ามีการพิมพ์ค้นหา)
if search_ticker:
    # ใช้ str.contains ค้นหาคำที่พิมพ์ (case=False คือไม่สนตัวพิมพ์เล็กพิมพ์ใหญ่)
    filtered_df = filtered_df[filtered_df['tickers_str'].str.contains(search_ticker.strip(), case=False, na=False)]

# --- 4. ส่วน UI: หน้าจอหลัก (Main Content) ---
st.title("📊 Thai Stock News Dashboard")
st.markdown("ระบบวิเคราะห์ข่าวการลงทุนอัตโนมัติด้วย AI (Gemini)")

# สร้างกล่อง KPI ด้านบน
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📰 จำนวนข่าวทั้งหมด", f"{len(filtered_df)} ข่าว")
with col2:
    pos_count = len(filtered_df[filtered_df['sentiment'] == 'Positive'])
    st.metric("🟢 ข่าวเชิงบวก", pos_count)
with col3:
    neg_count = len(filtered_df[filtered_df['sentiment'] == 'Negative'])
    st.metric("🔴 ข่าวเชิงลบ", neg_count)
with col4:
    high_impact_count = len(filtered_df[filtered_df['impact_level'] == 'High'])
    st.metric("⚡ ผลกระทบสูง", high_impact_count)

st.divider()

# สร้างกราฟ 2 ฝั่งซ้ายขวา
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("สัดส่วนอารมณ์ตลาด (Market Sentiment)")
    if not filtered_df.empty:
        sentiment_counts = filtered_df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts, color="#3b82f6") 
        
with chart_col2:
    st.subheader("กลุ่มอุตสาหกรรมที่ถูกพูดถึง (Sectors)")
    if not filtered_df.empty:
        sector_counts = filtered_df['sector'].value_counts()
        st.bar_chart(sector_counts, color="#8b5cf6")

st.divider()

# --- 5. ส่วน UI: ฟีดข่าว (News Feed) ---
st.subheader("📋 รายละเอียดข่าวล่าสุด")

if filtered_df.empty:
    st.info("ไม่มีข่าวที่ตรงกับเงื่อนไข หรือไม่พบหุ้นที่คุณค้นหา")
else:
    for index, row in filtered_df.iterrows(): 
        
        emoji = "🟢" if row.get('sentiment') == "Positive" else "🔴" if row.get('sentiment') == "Negative" else "⚪"
        impact_alert = " ⚡ [HIGH IMPACT]" if row.get('impact_level') == "High" else ""
        
        title_display = f"{emoji} {row.get('time', 'N/A')} | {row.get('title', 'ไม่มีหัวข้อ')} {impact_alert}"
        
        with st.expander(title_display):
            st.markdown(f"**หุ้นที่เกี่ยวข้อง:** `{row.get('tickers_str', 'N/A')}` | **หมวดหมู่:** `{row.get('sector', 'N/A')}`")
            st.markdown(f"**สรุป:** {row.get('summary', 'ไม่มีข้อมูลสรุป')}")
            st.markdown(f"**เหตุผล:** {row.get('reason', 'N/A')}")
            st.caption(f"Trend: {row.get('trend')} | Confidence: {row.get('confidence_level')}% | แหล่งที่มา: {row.get('source', 'N/A')}")
            
            if row.get('link'):
                st.markdown(f"[อ่านข่าวต้นฉบับ]({row.get('link')})")