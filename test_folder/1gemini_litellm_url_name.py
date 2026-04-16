import asyncio
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai import LLMExtractionStrategy, LLMConfig

# --- Configuration ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OUTPUT_FILE = "infoquest_news_test.json"

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

class Article(BaseModel):
    title: Optional[str] = Field(description="The article headline (in original language)")
    link: Optional[str] = Field(description="The full URL or relative path to the article")
    time: Optional[str] = Field(description="The published time or date (as shown on the page)")

async def main():
    # --- กำหนด URL ตรงนี้ เพื่อให้เรียกใช้ได้ทั่วทั้งฟังก์ชัน ---
    # target_url = "https://www.kaohoon.com/"
    target_url = "https://www.infoquest.co.th/stock"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting crawler on: {target_url}")

    llm_config = LLMConfig(
        provider="gemini/gemini-2.5-flash-lite", # ปรับเป็นชื่อโมเดลที่รองรับในปัจจุบัน
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
        # ใช้ตัวแปร target_url ที่เราประกาศไว้ข้างบน
        result = await crawler.arun(
            url=target_url,
            config=crawl_config
        )

    if not result.success:
        print("❌ Crawl failed:", result.error_message)
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Extraction successful. Parsing data...")
    
    try:
        extracted_data = json.loads(result.extracted_content)
    except (json.JSONDecodeError, TypeError):
        print("❌ Failed to parse extracted content.")
        return

    if isinstance(extracted_data, dict):
        articles = next((v for v in extracted_data.values() if isinstance(v, list)), [extracted_data])
    else:
        articles = extracted_data

    # นำ target_url มาใส่ใน metadata
    output = {
        "source": target_url,
        "scraped_at": datetime.now().isoformat(),
        "total_articles": len(articles),
        "articles": articles,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Extracted {len(articles)} articles → saved to '{OUTPUT_FILE}'")
    
    if articles:
        print("Preview (Top 3):")
        print(json.dumps(articles[:3], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())