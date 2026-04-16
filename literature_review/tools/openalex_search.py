"""
OpenAlex API 工具

提供学术文献检索功能，支持批量抓取论文摘要和元数据。

OpenAlex 是一个免费、开放的学术图谱 API，响应速度快，支持高速通道（mailto 参数）。
"""

import requests
import os
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def _reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> str:
    """
    将 OpenAlex 的倒排索引摘要还原为正常字符串
    
    Args:
        inverted_index: 倒排索引字典，格式为 {"word": [position1, position2, ...]}
    
    Returns:
        还原后的摘要字符串
    """
    if not inverted_index:
        return ""
    
    words = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))
    
    words = sorted(words)
    return " ".join([w[1] for w in words])


def search_papers(query: str, limit: int = 10, proxy: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    搜索 OpenAlex 学术论文

    Args:
        query: 搜索关键词
        limit: 返回结果数量限制
        proxy: 代理服务器地址（可选）

    Returns:
        论文列表，每篇论文包含 paper_id, title, abstract, authors, year, venue 等字段
    """
    url = "https://api.openalex.org/works"
    
    # 从环境变量获取 mailto 参数（OpenAlex 要求提供联系方式以使用高速通道）
    mailto = os.getenv("OPENALEX_MAILTO", "test@example.com")
    
    params = {
        "search": query,
        "per-page": limit,
        "mailto": mailto,
        "sort": "cited_by_count:desc"  # 按引用数排序，优先获取高质量论文
    }

    proxy_url = proxy or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

    try:
        response = requests.get(url, params=params, proxies=proxies, timeout=15)
        
        if response.status_code != 200:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            return []

        result = response.json()
        works = result.get("results", [])

        papers = []
        for work in works:
            inverted_index = work.get("abstract_inverted_index")
            abstract = _reconstruct_abstract(inverted_index) if inverted_index else ""
            
            if not abstract:
                continue
            
            authorships = work.get("authorships", [])
            authors = [
                authorship.get("author", {}).get("display_name", "")
                for authorship in authorships
                if authorship.get("author", {}).get("display_name")
            ]
            
            primary_location = work.get("primary_location") or {}
            source = primary_location.get("source") or {}
            venue = source.get("display_name", "")
            
            paper = {
                "paper_id": work.get("id", ""),
                "title": work.get("title", ""),
                "abstract": abstract,
                "authors": authors,
                "year": work.get("publication_year"),
                "venue": venue,
                "citations": work.get("cited_by_count", 0)
            }
            
            papers.append(paper)

        logger.info(f"Retrieved {len(papers)} papers with abstracts (out of {len(works)} total)")
        return papers

    except requests.exceptions.Timeout:
        logger.error(f"Request timeout after 15 seconds")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP client error: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during API call: {e}")
        return []


if __name__ == "__main__":
    async def main():
        query = "LLM RAG Medical"
        limit = 5

        proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
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
            venue = paper.get("venue", "N/A")
            abstract = paper.get("abstract", "")
            abstract_preview = abstract[:150] + "..." if len(abstract) > 150 else abstract

            print(f"[{i}] {title}")
            print(f"    Year: {year} | Venue: {venue}")
            print(f"    Abstract: {abstract_preview}\n")

    asyncio.run(main())
