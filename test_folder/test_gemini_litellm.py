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
# Load variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OUTPUT_FILE = "infoquest_news.json"

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

# --- Define the extraction schema using Pydantic ---
class Article(BaseModel):
    title: Optional[str] = Field(description="The article headline (in original language)")
    link: Optional[str] = Field(description="The full URL or relative path to the article")
    time: Optional[str] = Field(description="The published time or date (as shown on the page, e.g. '10:30', '2 hours ago', '09/04/2025')")

async def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting crawler with LiteLLM extraction...")

    # 1. Configure the LLM using the required LLMConfig object
    llm_config = LLMConfig(
        provider="gemini/gemini-2.5-flash-lite", 
        api_token=GEMINI_API_KEY
    )
    
    # 2. Pass the config to the extraction strategy
    extraction_strategy = LLMExtractionStrategy(
        llm_config=llm_config,
        schema=Article.model_json_schema(), 
        extraction_type="schema",
        instruction=(
            "Extract ALL news articles you can find from the scraped Thai financial news website content. "
            "Return an array of objects matching the provided schema. If a field is not found, leave it null."
        )
    )

    # 3. Configure the crawl run (newer crawl4ai requires parameters to be passed via CrawlerRunConfig)
    crawl_config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        cache_mode=CacheMode.BYPASS # Good practice when tracking live news
    )

    async with AsyncWebCrawler() as crawler:
        # 4. Execute the crawl using the config object
        result = await crawler.arun(
            # url="https://www.infoquest.co.th/stock",
            # url="https://www.posttoday.com/business/stockholder",
            url="https://www.kaohoon.com/",
            config=crawl_config
        )

    if not result.success:
        print("❌ Crawl failed:", result.error_message)
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Crawl and extraction successful. Parsing data...")
    
    try:
        extracted_data = json.loads(result.extracted_content)
    except json.JSONDecodeError as e:
        print("❌ Failed to parse extracted content as JSON. Raw response:")
        print(result.extracted_content)
        return
    except TypeError:
        print("❌ Extracted content is empty or invalid.")
        return

    # Handle dictionary vs list wrapping based on LiteLLM outputs
    if isinstance(extracted_data, dict):
        articles = next((v for v in extracted_data.values() if isinstance(v, list)), [extracted_data])
    else:
        articles = extracted_data

    # Add metadata
    output = {
        "source": url,
        "scraped_at": datetime.now().isoformat(),
        "total_articles": len(articles),
        "articles": articles,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Extracted {len(articles)} articles → saved to '{OUTPUT_FILE}'")
    
    if articles:
        print("Preview:")
        print(json.dumps(articles[:3], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())