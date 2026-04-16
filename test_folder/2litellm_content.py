import asyncio
import json
import re
import os
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

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
    file_path = "infoquest_news_test.json"
    
    if not os.path.exists(file_path):
        print(f"❌ ไม่พบไฟล์ {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    articles = data.get("articles", [])
    
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        excluded_tags=['nav', 'footer', 'header', 'aside', 'script', 'style', 'form'],
        word_count_threshold=10
    )

    async with AsyncWebCrawler() as crawler:
        for index, article in enumerate(articles):
            url = article.get("link")
            print(f"[{index+1}/{len(articles)}] Scrapping: {article.get('title')[:30]}...")
            
            try:
                result = await crawler.arun(url=url, config=config)
                if result.success:
                    # ใช้โครงสร้าง result.markdown.fit_markdown ตามเวอร์ชันที่คุณใช้
                    try:
                        raw = result.markdown.fit_markdown if result.markdown.fit_markdown else result.markdown.raw
                    except:
                        raw = str(result.markdown)
                    
                    article["content"] = clean_markdown_content(raw)
                else:
                    article["content"] = "Error"
            except Exception as e:
                article["content"] = f"Error: {str(e)}"

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("\n✅ อัปเดตไฟล์ JSON เรียบร้อยแล้ว (ลบลิงก์ออกทั้งหมด)")

if __name__ == "__main__":
    asyncio.run(main())