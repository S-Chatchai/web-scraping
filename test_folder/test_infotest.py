import asyncio
import re
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

def clean_markdown_links(text):
    """ฟังก์ชันลบ [text](url) -> text และลบรูปภาพออกทั้งหมด"""
    if not text: return ""
    # 1. ลบลิงก์ [ข้อความ](https://...) -> เหลือแค่ "ข้อความ"
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # 2. ลบรูปภาพ ![alt](https://...) -> ลบทิ้งไปเลย
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
    # 3. ลบช่องว่างที่ซ้ำซ้อนกันเกินไป
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    return text

async def main():
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        # กรองส่วนที่ไม่ใช่เนื้อหาออกตั้งแต่ระดับ HTML
        excluded_tags=['nav', 'footer', 'header', 'aside', 'script', 'style', 'form', 'button'],
        word_count_threshold=10
    )

    async with AsyncWebCrawler() as crawler:
        print("--- กำลังดึงข้อมูลจาก InfoQuest ---")
        result = await crawler.arun(
            url="https://www.infoquest.co.th/2026/584493",
            config=config
        )
        
        if result.success:
            # ดึงเนื้อหาตามโครงสร้างใหม่ที่ระบบแนะนำ
            # โครงสร้าง: result.markdown (Object) -> fit_markdown (String)
            try:
                # ลองใช้ fit_markdown ที่ผ่านการคำนวณความหนาแน่นของข้อความมาแล้ว
                raw_content = result.markdown.fit_markdown if result.markdown.fit_markdown else result.markdown.raw
            except AttributeError:
                # กันเหนียวถ้าเข้าถึง .fit_markdown ไม่ได้ ให้ใช้ markdown ปกติ
                raw_content = str(result.markdown)

            # ทำความสะอาดลิงก์และรูปภาพ
            final_content = clean_markdown_links(raw_content)
            
            # แสดงผลบน Terminal
            print("\n" + "="*50)
            print("MAIN CONTENT RESULT:")
            print("="*50 + "\n")
            
            if final_content:
                print(final_content)
            else:
                print("ไม่พบเนื้อหาหลักในหน้านี้")
                
            print("\n" + "="*50)
        else:
            print(f"❌ เกิดข้อผิดพลาด: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())