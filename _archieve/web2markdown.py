import asyncio
from crawl4ai import AsyncWebCrawler

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url="https://www.chitika.com/hybrid-retrieval-rag/")
        return result.markdown

if __name__ == "__main__":
    res = asyncio.run(main())
    with open("./text_document/result.txt", "w", encoding="utf-8") as f:
        f.write(res)