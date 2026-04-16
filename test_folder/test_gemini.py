import asyncio
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler
import google.generativeai as genai

# --- Configuration ---
# Load variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OUTPUT_FILE = "infoquest_news.json"

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)


def extract_with_gemini(markdown_content: str) -> list[dict]:
    """Send markdown content to Gemini and extract structured news data."""
    model = genai.GenerativeModel("models/gemini-2.5-flash-lite")

    prompt = f"""
You are a data extraction assistant. Below is markdown content scraped from a Thai financial news website (infoquest.co.th/stock).

Extract ALL news articles you can find. For each article, return:
- title: the article headline (in original language)
- link: the full URL or relative path to the article
- time: the published time or date (as shown on the page, e.g. "10:30", "2 hours ago", "09/04/2025")

Return ONLY a valid JSON array like this (no explanation, no markdown fences):
[
  {{
    "title": "Article headline here",
    "link": "https://...",
    "time": "10:30"
  }}
]

If a field is not found, use null for its value.
Do not include any text outside the JSON array.

--- MARKDOWN CONTENT START ---
{markdown_content[:12000]}
--- MARKDOWN CONTENT END ---
"""

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Improved JSON cleaning logic
    if "```" in raw:
        # Extract content between triple backticks
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


async def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting crawler...")

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url="https://www.infoquest.co.th/stock")

    if not result.success:
        print("❌ Crawl failed:", result.error_message)
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Crawl successful. Sending to Gemini...")
    markdown = result.markdown

    try:
        articles = extract_with_gemini(markdown)
    except json.JSONDecodeError as e:
        print("❌ Failed to parse Gemini response as JSON. Raw response:")
        # Optional: print(response.text) for debugging
        print(e)
        return
    except Exception as e:
        print("❌ Gemini API error:", e)
        return

    # Add metadata
    output = {
        "source": "https://www.infoquest.co.th/stock",
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