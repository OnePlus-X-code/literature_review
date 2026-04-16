"""
Writer Agent - 学术综述撰写专家

负责基于提取的学术观点，使用对比驱动写作（Contrastive Prompting）生成综述章节。
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import os
    import sys
    import requests
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from literature_review.agents.base import BaseAgent
    from literature_review.state import ResearchState, Paper, AcademicExtraction
else:
    from .base import BaseAgent
    from ..state import ResearchState, Paper, AcademicExtraction


class WriterAgent(BaseAgent):
    """
    撰写 Agent
    
    职责：
    1. 从全局状态中获取提取的学术观点
    2. 使用对比驱动写作（Contrastive Prompting）生成综述
    3. 将生成的章节存入 draft_sections
    """

    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: str,
        model: str = "qwen-max"
    ):
        super().__init__(
            name="Writer",
            role="学术综述撰写专家，负责基于文献数据撰写深度批判性综述",
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            model=model
        )

    async def process(self, state: ResearchState) -> Dict[str, Any]:
        """
        处理状态并返回更新后的字段
        
        Args:
            state: 当前研究状态
            
        Returns:
            需要更新的字段字典
        """
        extractions = state.get("extractions", [])
        
        if not extractions:
            logger.error("No extractions found in state. Cannot write without source material.")
            return {"draft_sections": state.get("draft_sections", {})}

        logger.info(f"Found {len(extractions)} extractions to synthesize")

        outline = state.get("outline", [])
        if not outline:
            section_title = "综合文献综述 (Literature Synthesis)"
            logger.info(f"No outline found, using default section title: {section_title}")
        else:
            section_title = outline[0].get("section", "综合文献综述 (Literature Synthesis)")

        papers = state.get("papers", {})
        context_text = self._prepare_context(extractions, papers)

        system_prompt = "你是一位顶尖的学术领域分析师。你的任务是基于提供的文献数据，撰写一段具有深度批判性的学术综述。请直接输出 Markdown 格式的正文，不要任何寒暄。"

        user_prompt = f"""
请撰写章节：【{section_title}】

以下是知识库中提取的文献素材：
{context_text}

【强制写作规范 - Contrastive Prompting】：
1. 绝对禁止流水账式的罗列（例如"A 说了什么，B 说了什么"）。
2. 必须提取文献间的【共识】与【分歧】。请对比不同文献在解决相同问题时，所用方法或结论的优劣差异。
3. 引用规范（极其重要）：
   - 每一句陈述必须有文献支撑
   - 如果素材中作者字段是 `[xxxx]` 或年份是 `N/A`，请使用 `[论文 ID] 或 [标题片段] 进行引用，例如 ([Neural Holography], 2023) 或 ([ID:abc123], N/A)
   - **绝对禁止**编造具体的作者姓名（如 Wang, Zhang, Smith 等）和具体年份（如 2022, 2023, 2024 等）
   - 违规后果：编造引用会导致整篇综述被拒稿
4. 语言要求严谨、客观，体现出强烈的学术思辨性。
"""

        logger.info(f"Calling LLM to write section: {section_title}")
        draft_text = await self.call_llm(system_prompt, user_prompt, json_mode=False)

        if "draft_sections" not in state:
            state["draft_sections"] = {}
        
        state["draft_sections"][section_title] = draft_text

        logger.info(f"Successfully wrote section: {section_title}")
        return {"draft_sections": state["draft_sections"]}

    def _prepare_context(self, extractions: List[AcademicExtraction], papers: Dict[str, Paper]) -> str:
        """
        准备格式化的文献素材上下文
        
        Args:
            extractions: 学术观点提取列表
            papers: 论文字典
            
        Returns:
            格式化的长文本
        """
        context_parts = []
        
        for extraction in extractions:
            paper_id = extraction.paper_id
            paper = papers.get(paper_id)
            
            if not paper:
                logger.warning(f"Paper {paper_id} not found in state, skipping")
                continue
            
            # 处理作者：如果缺失，直接使用 paper_id 或标题前缀
            if paper.authors_list and len(paper.authors_list) > 0:
                first_author = paper.authors_list[0]
                author_display = f"{first_author} et al." if len(paper.authors_list) > 1 else first_author
            else:
                # 没有作者信息时，使用 paper_id 的前 8 位或标题前 20 个字
                author_display = f"[{paper_id[:8]}]" if len(paper_id) > 8 else f"[{paper.title[:20]}...]"
            
            # 处理年份：如果缺失，明确标注为 N/A
            year_display = paper.year if paper.year else "N/A"
            
            context_parts.append(f"""
---
作者：{author_display}
年份：{year_display}
标题：{paper.title}
目的：{extraction.purpose}
方法：{extraction.methodology}
结论：{extraction.conclusion}
局限性：{extraction.limitations or 'N/A'}
---
""")
        
        return "\n".join(context_parts)


if __name__ == "__main__":
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    
    # ==========================================
    # 1. 代理与网络安全设置 (极其重要)
    # ==========================================
    # 清除代理环境变量，让 requests 直接连接
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    # 注意：不要设置 NO_PROXY，让 requests 自己处理
    
    # 加载 literature_review 目录下的 .env 文件
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)
    
    initial_state: ResearchState = {
        "query": "RAG in healthcare",
        "phase": "drafting",
        "outline": [],
        "papers": {
            "paper1": Paper(
                paper_id="paper1",
                title="Retrieval-Augmented Generation for Medical Question Answering",
                abstract="This paper presents a RAG system for medical QA.",
                authors_list=["Smith", "Johnson", "Williams"],
                year=2023,
                venue="Medical AI Journal"
            ),
            "paper2": Paper(
                paper_id="paper2",
                title="Knowledge Graph Enhanced RAG for Clinical Decision Support",
                abstract="This paper proposes KG-enhanced RAG for clinical decisions.",
                authors_list=["Chen", "Wang", "Li"],
                year=2024,
                venue="Nature Medicine"
            )
        },
        "extractions": [
            AcademicExtraction(
                paper_id="paper1",
                purpose="开发一个基于 RAG 的医疗问答系统，提高医疗信息检索的准确性",
                methodology="使用 Dense Passage Retrieval (DPR) 结合 BART 生成器，在 PubMed 数据集上训练",
                conclusion="RAG 系统在医疗问答任务上达到了 85% 的准确率，显著优于传统检索方法",
                limitations="仅适用于英文医疗文献，对多语言支持不足"
            ),
            AcademicExtraction(
                paper_id="paper2",
                purpose="利用知识图谱增强 RAG 系统，提升临床决策支持的可靠性",
                methodology="构建医学知识图谱，将图谱嵌入与向量检索相结合，使用 GPT-4 作为生成器",
                conclusion="KG 增强方法在复杂临床推理任务上比纯向量检索提高了 12% 的 F1 分数",
                limitations="知识图谱构建成本高，需要领域专家手动验证"
            )
        ],
        "glossary": {},
        "messages": [],
        "draft_sections": {},
        "final_report": "",
        "iteration": 0,
        "max_iterations": 3,
        "review_feedback": []
    }

    async def main():
        # 从环境变量加载配置
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        base_url = os.getenv("DASHSCOPE_BASE_URL", "")
        model = os.getenv("OPENAI_MODEL", "qwen-max")
        
        if not api_key or not base_url:
            print("Error: Please set DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL in .env")
            return
        
        agent = WriterAgent(
            llm_api_key=api_key,
            llm_base_url=base_url,
            model=model
        )
        
        result = await agent.process(initial_state)
        
        print("\n" + "="*80)
        print("生成的综述内容:")
        print("="*80)
        for section_title, content in result.get("draft_sections", {}).items():
            print(f"\n## {section_title}\n")
            print(content)
        print("="*80)

    asyncio.run(main())
