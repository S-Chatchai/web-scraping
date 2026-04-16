import asyncio
from crawl4ai import *

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://www.infoquest.co.th/2026/584498",
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())