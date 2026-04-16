import asyncio
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
from pymongo import MongoClient, UpdateOne # เพิ่มไลบรารีสำหรับ MongoDB

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai import LLMExtractionStrategy, LLMConfig

# --- Configuration ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI") # ดึง Connection String ของ Atlas จาก .env
OUTPUT_FILE = "infoquest_news_test.json"

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Please check your .env file.")
if not MONGODB_URI:
    raise ValueError("❌ MONGODB_URI not found. Please check your .env file.")

class Article(BaseModel):
    title: Optional[str] = Field(description="The article headline (in original language)")
    link: Optional[str] = Field(description="The full URL or relative path to the article")
    # time: Optional[str] = Field(description="The published time or date (as shown on the page)")

async def main():
    target_url = "https://www.infoquest.co.th/stock"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting crawler on: {target_url}")

    llm_config = LLMConfig(
        provider="gemini/gemini-2.5-flash-lite", 
        api_token=GEMINI_API_KEY
    )
    
    extraction_strategy = LLMExtractionStrategy(
        llm_config=llm_config,
        schema=Article.model_json_schema(), 
        extraction_type="schema",
        instruction=(
            "Extract ALL news articles you can find from the scraped Thai financial news website content. "
            "Return an array of objects matching the provided schema. If a field is not found, leave it null."
        )
    )

    crawl_config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        cache_mode=CacheMode.BYPASS 
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=target_url,
            config=crawl_config
        )

    if not result.success:
        print("❌ Crawl failed:", result.error_message)
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Extraction successful. Parsing data...")
    
    try:
        extracted_data = json.loads(result.extracted_content)
    except (json.JSONDecodeError, TypeError):
        print("❌ Failed to parse extracted content.")
        return

    if isinstance(extracted_data, dict):
        articles = next((v for v in extracted_data.values() if isinstance(v, list)), [extracted_data])
    else:
        articles = extracted_data

    # # --- การบันทึกลงไฟล์ JSON (เก็บไว้เป็น Backup) ---
    # output = {
    #     "source": target_url,
    #     "scraped_at": datetime.now().isoformat(),
    #     "total_articles": len(articles),
    #     "articles": articles,
    # }

    # with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    #     json.dump(output, f, ensure_ascii=False, indent=2)
    # print(f"📁 Extracted {len(articles)} articles → saved to '{OUTPUT_FILE}'")

    # --- การบันทึกลง MongoDB Atlas ---
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ☁️ Connecting to MongoDB Atlas...")
    client = MongoClient(MONGODB_URI)
    db = client["finance_db"] # ตั้งชื่อ Database
    collection = db["news_articles"] # ตั้งชื่อ Collection

    operations = []
    for article in articles:
        if article.get("link"):
            # เพิ่ม Metadata เพื่อประโยชน์ในการทำ Data Analysis ภายหลัง
            article["scraped_at"] = datetime.now().isoformat()
            article["source"] = target_url
            
            # สร้างคำสั่ง UpdateOne สำหรับแต่ละ Article
            operations.append(
                UpdateOne(
                    {"link": article["link"]}, # ค้นหาจาก URL ว่ามีข่าวนี้หรือยัง
                    {"$set": article},         # ถ้ามีให้อัปเดต, ถ้าไม่มีให้สร้างใหม่
                    upsert=True
                )
            )

    if operations:
        db_result = collection.bulk_write(operations)
        print(f"✅ MongoDB Update Complete: {db_result.upserted_count} inserted, {db_result.modified_count} updated.")

    if articles:
        print("\nPreview (Top 3):")
        print(json.dumps(articles[:3], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())