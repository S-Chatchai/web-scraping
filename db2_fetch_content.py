import asyncio
import re
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

# โหลดตัวแปรจาก .env
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("❌ MONGODB_URI not found. Please check your .env file.")

def clean_markdown_content(text):
    if not text: return ""
    # ลบ [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', '', text)
    # ลบ ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
    # ลบ https://... หรือ www...
    text = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', '', text)
    # ลบสัญลักษณ์ Markdown
    text = text.replace('#', '').replace('*', '')
    return re.sub(r'\n\s*\n', '\n\n', text).strip()

async def main():
    print("☁️ Connecting to MongoDB Atlas...")
    client = MongoClient(MONGODB_URI)
    db = client["finance_db"] # ต้องตรงกับชื่อ Database ในขั้นตอนที่ 1
    collection = db["news_articles"]

    # ค้นหาเฉพาะบทความที่ยังไม่มีฟิลด์ "content" 
    # (เพื่อจะได้ไม่ดึงซ้ำบทความที่เคยดึงเนื้อหาไปแล้ว)
    articles_to_scrape = list(collection.find({"content": {"$exists": False}}))
    
    if not articles_to_scrape:
        print("✅ ไม่มีข่าวใหม่ที่ต้องดึงเนื้อหา (ทุกข่าวมีเนื้อหาครบแล้ว)")
        return

    print(f"📊 พบข่าวที่ต้องดึงเนื้อหาเพิ่มเติมจำนวน {len(articles_to_scrape)} ข่าว")

    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        excluded_tags=['nav', 'footer', 'header', 'aside', 'script', 'style', 'form'],
        word_count_threshold=10
    )

    async with AsyncWebCrawler() as crawler:
        for index, article in enumerate(articles_to_scrape):
            url = article.get("link")
            article_id = article.get("_id") # ดึง ID ของ Document ใน Database มาใช้
            title = article.get('title') or "No Title"
            
            print(f"[{index+1}/{len(articles_to_scrape)}] Scrapping: {title[:30]}...")
            
            if not url:
                continue

            try:
                result = await crawler.arun(url=url, config=config)
                if result.success:
                    try:
                        raw = result.markdown.fit_markdown if result.markdown.fit_markdown else result.markdown.raw
                    except AttributeError:
                        raw = str(result.markdown)
                    
                    content = clean_markdown_content(raw)
                else:
                    content = f"Error: Scrape failed - {result.error_message}"
            except Exception as e:
                content = f"Error: Exception occurred - {str(e)}"
            
            # อัปเดตเนื้อหาที่ดึงได้ (และ clean แล้ว) กลับเข้าไปใน Database ทันทีทีละรายการ
            collection.update_one(
                {"_id": article_id},
                {"$set": {"content": content}}
            )

    print("\n✅ อัปเดตเนื้อหาข่าวทั้งหมดลง MongoDB Atlas เรียบร้อยแล้ว!")

if __name__ == "__main__":
    asyncio.run(main())