from pymongo import MongoClient, UpdateOne
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # ใช้สำหรับดู progress bar

def main():
    # ==========================================
    # 1. ตั้งค่าการเชื่อมต่อ MongoDB Atlas
    # ==========================================
    MONGO_URI = ""
    
    try:
        client = MongoClient(MONGO_URI)
        db = client["finance_db"]
        collection = db["news_articles"]
        print(f"✅ เชื่อมต่อ MongoDB สำเร็จ! (Collection: {collection.name})")
    except Exception as e:
        print(f"❌ เชื่อมต่อ MongoDB ไม่สำเร็จ: {e}")
        return

    # ==========================================
    # 2. โหลดโมเดล BAAI/bge-m3
    # ==========================================
    print("กำลังโหลดโมเดล BAAI/bge-m3 (อาจใช้เวลาสักครู่)...")
    model = SentenceTransformer('BAAI/bge-m3')

    # ==========================================
    # 3. ดึงข้อมูลที่ยังไม่มี 'embedding' มาประมวลผล
    # ==========================================
    # หมายเหตุ: กรองเอาเฉพาะเอกสารที่ยังไม่มีฟิลด์ embedding เพื่อประหยัดเวลาและทรัพยากร
    query = {"embedding": {"$exists": False}}
    cursor = collection.find(query)
    total_docs = collection.count_documents(query)

    if total_docs == 0:
        print("✨ ข้อมูลทั้งหมดถูกทำ Embedding เรียบร้อยแล้ว ไม่พบข้อมูลใหม่")
        return

    print(f"พบข้อมูลทั้งหมด {total_docs} รายการที่ต้องทำ Embedding")

    # ==========================================
    # 4. เริ่มกระบวนการทำ Embedding และเตรียม Update แบบ Bulk
    # ==========================================
    batch_size = 50  # ทำทีละ 50 รายการเพื่อไม่ให้กิน RAM จนเกินไป
    updates = []
    
    # ใช้ tqdm เพื่อแสดงความคืบหน้า
    for doc in tqdm(cursor, total=total_docs, desc="Processing Embeddings"):
        # รวม text ที่ต้องการทำ vector (ปรับแต่งได้ตามความต้องการ)
        # แนะนำให้ใช้ title + content (หรือ summary)
        title = doc.get("title", "")
        summary = doc.get("summary", "")
        content = doc.get("content", "")
        
        text_to_embed = f"หัวข้อ: {title}\nสรุป: {summary}\nเนื้อหา: {content}"
        
        # สร้าง Vector
        vector = model.encode(text_to_embed, normalize_embeddings=True).tolist()
        
        # เตรียมคำสั่งอัปเดตโดยใช้ _id เดิม
        updates.append(
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {"embedding": vector}}
            )
        )

        # เมื่อครบขนาด batch ให้ทำการเขียนลงฐานข้อมูลทีเดียว
        if len(updates) >= batch_size:
            collection.bulk_write(updates)
            updates = []

    # อัปเดตรายการที่เหลือ (ถ้ามี)
    if updates:
        collection.bulk_write(updates)

    print(f"\n✅ เสร็จสมบูรณ์! อัปเดตข้อมูลทั้งหมด {total_docs} รายการเรียบร้อยแล้ว")

if __name__ == "__main__":
    main()
