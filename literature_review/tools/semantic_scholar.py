"""
Semantic Scholar API 工具

提供异步学术文献检索功能，支持批量抓取论文摘要和元数据。
"""

import aiohttp
import asyncio
import os
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


async def search_papers(query: str, limit: int = 10, proxy: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    异步搜索 Semantic Scholar 学术论文

    Args:
        query: 搜索关键词
        limit: 返回结果数量限制
        proxy: 代理服务器地址（可选），如 "http://127.0.0.1:7890"

    Returns:
        论文列表，每篇论文包含 paperId, title, abstract, authors, year, venue 等字段
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,externalIds,title,abstract,authors,year,venue,citationCount"
    }

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"x-api-key": api_key} if api_key else {}

    timeout = aiohttp.ClientTimeout(total=15)

    proxy_url = proxy or os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")

    try:
        connector = aiohttp.TCPConnector(ssl=False) if proxy_url else None
        async with aiohttp.ClientSession(
            timeout=timeout, 
            headers=headers,
            connector=connector
        ) as session:
            async with session.get(url, params=params, proxy=proxy_url) as response:
                if response.status != 200:
                    logger.error(f"API request failed with status {response.status}: {response.reason}")
                    return []

                result = await response.json()
                data = result.get("data", [])

                filtered_papers = [paper for paper in data if paper.get("abstract")]

                logger.info(f"Retrieved {len(filtered_papers)} papers with abstracts (out of {len(data)} total)")
                return filtered_papers

    except asyncio.TimeoutError:
        logger.error(f"Request timeout after {timeout.total} seconds")
        return []
    except aiohttp.ClientError as e:
        logger.error(f"HTTP client error: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during API call: {e}")
        return []


if __name__ == "__main__":
    async def main():
        query = "LLM RAG Medical"
        limit = 5

        proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        if proxy:
            logger.info(f"Using proxy: {proxy}")

        logger.info(f"Searching for '{query}' (limit: {limit})")
        papers = await search_papers(query, limit, proxy=proxy)

        if not papers:
            logger.warning("No papers found")
            return

        print(f"\nFound {len(papers)} papers:\n")
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "N/A")
            year = paper.get("year", "N/A")
            abstract = paper.get("abstract", "")
            abstract_preview = abstract[:100] + "..." if len(abstract) > 100 else abstract

            print(f"[{i}] {title}")
            print(f"    Year: {year}")
            print(f"    Abstract: {abstract_preview}\n")

    asyncio.run(main())
