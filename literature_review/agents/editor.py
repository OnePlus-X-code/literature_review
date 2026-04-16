"""
Editor Agent - 学术综述主编

负责对通过审核的文献综述进行最终润色、排版和参考文献格式化。
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from literature_review.agents.base import BaseAgent
    from literature_review.state import ResearchState, ResearchPhase, Paper
else:
    from .base import BaseAgent
    from ..state import ResearchState, ResearchPhase, Paper


class EditorAgent(BaseAgent):
    """
    主编 Agent
    
    职责：
    1. 对通过审核的文献综述进行最终润色和排版
    2. 格式化参考文献列表
    3. 生成完整的终稿报告
    """

    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: str,
        model: str = "qwen-max"
    ):
        super().__init__(
            name="Editor",
            role="资深学术主编，负责文献综述的最终润色和排版",
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
        draft_sections = state.get("draft_sections", {})
        papers = state.get("papers", {})
        
        if not draft_sections:
            logger.error("No draft sections found. Cannot create final report.")
            return {"phase": state.get("phase", "drafting")}

        logger.info(f"Editing {len(draft_sections)} section(s) with {len(papers)} paper(s)")

        # 组装终稿内容
        assembled_draft = self._assemble_draft(draft_sections)
        references = self._generate_references(papers)
        
        # 检测缺失元数据的论文
        missing_meta_papers = []
        for pid, p in papers.items():
            if not p.year or not p.authors_list or "Unknown" in p.authors_list:
                missing_meta_papers.append(f"- {p.title} (ID: {pid})")

        warning_text = ""
        if missing_meta_papers:
            warning_text = "\n\n【⚠️ 引用补充提示 (Action Required)】：\n受限于检索接口，以下文献缺失完整的作者或年份信息。为保证学术严谨性，请在定稿时手动核查并补充：\n" + "\n".join(missing_meta_papers)

        logger.info("Calling LLM for final polishing...")
        
        system_prompt = "你是一位资深学术主编。你的任务是对已经通过审核的文献综述草稿进行最后的润色和排版。请直接输出最终的 Markdown 全文。"

        user_prompt = f"""
请对以下综述草稿进行润色，使其语言更加流畅，各章节过渡更加自然。

【综述草稿】：
{assembled_draft}

【参考文献】：
{references}
{warning_text}

排版要求：
1. 保持原文的引用标记不变。
2. 修正明显的语病。
3. 在文末添加"**参考文献 (References)**"章节，列出上述参考文献。
4. 如果提供了【⚠️ 引用补充提示】，请务必将其原样保留在参考文献列表的最下方。
5. 请务必在正文的最后（"**参考文献 (References)**"章节之前），单独创建一个全新的章节标题（例如："## 1.5 本章小结"），用一段话对全文的脉络和核心结论进行高度凝练的总结。千万不要把它混在上一节的末尾。
"""

        final_report = await self.call_llm(system_prompt, user_prompt, json_mode=False)
        
        logger.info(f"Final report generated, length: {len(final_report)} characters")

        # 状态路由更新
        return {
            "phase": ResearchPhase.COMPLETED.value,
            "final_report": final_report
        }

    def _assemble_draft(self, draft_sections: Dict[str, str]) -> str:
        """
        将所有草稿章节拼接成完整的草稿
        
        Args:
            draft_sections: 草稿段落字典
            
        Returns:
            完整的草稿文本
        """
        if not draft_sections:
            return "无草稿内容"

        parts = []
        for section_title, content in draft_sections.items():
            parts.append(f"## {section_title}\n\n{content}")
        
        return "\n\n".join(parts)

    def _generate_references(self, papers: Dict[str, Paper]) -> str:
        """
        生成格式化的参考文献列表（APA 格式）
        
        Args:
            papers: 论文字典
            
        Returns:
            格式化的参考文献文本
        """
        if not papers:
            return "无参考文献"

        references = []
        for paper_id, paper in papers.items():
            ref = self._format_paper_reference(paper)
            references.append(ref)
        
        return "\n".join(references)

    def _format_paper_reference(self, paper: Paper) -> str:
        """
        将单篇论文格式化为 APA 引用格式
        
        Args:
            paper: 论文对象
            
        Returns:
            APA 格式的引用字符串
        """
        # 处理作者列表
        authors = self._format_authors(paper.authors_list)
        
        # 处理年份（缺失则用 N/A）
        year = paper.year if paper.year else "N/A"
        
        # 处理标题
        title = paper.title if paper.title else "Untitled"
        
        # 处理期刊/会议名称（缺失则用 N/A）
        venue = paper.venue if paper.venue else "N/A"
        
        # 处理 paper_id（作为 DOI 或 ID）
        paper_id = paper.paper_id if paper.paper_id else "Unknown ID"
        
        # APA 格式：Authors (Year). Title. *Venue*. DOI/ID: paper_id
        return f"{authors} ({year}). {title}. *{venue}*. DOI/ID: {paper_id}"

    def _format_authors(self, authors_list: List[str]) -> str:
        """
        格式化作者列表
        
        Args:
            authors_list: 作者列表
            
        Returns:
            格式化的作者字符串
        """
        if not authors_list:
            return "Unknown Authors"
        
        # 如果作者数量 <= 3，列出所有作者
        if len(authors_list) <= 3:
            return ", ".join(authors_list)
        
        # 如果作者数量 > 3，只列出第一作者 + et al.
        return f"{authors_list[0]} et al."


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
    
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "")
    model = os.getenv("OPENAI_MODEL", "qwen-max")
    
    if not api_key or not base_url:
        print("Error: Please set DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL in .env")
        sys.exit(1)
    
    # ==========================================
    # 测试：伪造一篇论文和一段合格草稿
    # ==========================================
    print("\n" + "="*80)
    print("测试：Editor Agent 终稿生成")
    print("="*80)
    
    from literature_review.state import Paper
    
    # 伪造两篇论文
    papers = {
        "paper1": Paper(
            paper_id="10.1038/s41591-023-02345-6",
            title="Retrieval-Augmented Generation for Medical Question Answering",
            abstract="This paper presents a RAG system for medical question answering using DPR and BART.",
            authors_list=["Smith, John", "Johnson, Mary", "Williams, Robert"],
            year=2023,
            venue="Nature Medicine"
        ),
        "paper2": Paper(
            paper_id="10.1016/j.cell.2024.01.001",
            title="Knowledge Graph Enhanced RAG for Clinical Decision Support",
            abstract="This paper proposes a knowledge graph enhanced RAG framework for clinical decision support.",
            authors_list=["Chen, Wei", "Li, Xiaoming", "Wang, Fang", "Zhang, Qiang"],
            year=2024,
            venue="Cell"
        )
    }
    
    # 伪造一段合格的草稿
    draft_sections = {
        "综合文献综述 (Literature Synthesis)": """
近年来，检索增强生成（Retrieval-Augmented Generation, RAG）范式在医疗人工智能领域展现出显著潜力，其核心目标在于弥合大规模语言模型的生成能力与权威医学知识之间的鸿沟。现有研究虽路径各异，但在一个关键共识上高度一致：单纯依赖参数化知识的语言模型难以满足临床场景对准确性与可解释性的严苛要求，必须通过外部知识源进行动态增强 (Smith et al., 2023; Chen et al., 2024)。

然而，如何有效整合外部知识，学界呈现出方法论上的显著分歧，主要体现为"纯向量检索"与"结构化知识融合"两条技术路线的张力。Smith 等人（2023）代表了前者，其工作聚焦于优化非结构化文本的端到端检索与生成流程，采用 Dense Passage Retrieval (DPR) 从 PubMed 文献库中召回相关段落，并由 BART 模型进行答案生成。该方法的优势在于实现相对简洁、可扩展性强，且在标准问答任务上实现了 85% 的准确率，显著超越传统关键词检索基线。然而，其局限性亦源于对文本表层语义的依赖——系统缺乏对医学概念间深层逻辑关系的建模能力，难以应对需要多跳推理或因果推断的复杂临床问题 (Smith et al., 2023)。

与此形成鲜明对比的是，Chen 等人（2024）主张引入结构化先验知识，通过构建医学知识图谱（Knowledge Graph, KG）显式编码疾病、症状、药物及治疗方案间的语义关联，并将图谱嵌入与向量检索结果融合后输入 GPT-4 生成器。实证表明，该方法在复杂临床决策任务上的 F1 分数较纯向量检索提升 12%，验证了结构化知识对高阶推理的增益效应 (Chen et al., 2024)。但这一性能提升以高昂的知识工程成本为代价，包括依赖领域专家进行图谱构建与验证，严重制约了系统的可迁移性与部署效率。

综上，当前 RAG 在医疗领域的演进呈现出"效率—深度"的权衡困境：Smith 等人的方法优先保障了系统的通用性与可扩展性，却牺牲了复杂推理能力；而 Chen 等人的方案虽在推理可靠性上取得突破，却因知识图谱的构建瓶颈难以规模化应用。这一根本性分歧揭示了未来研究的关键方向——亟需探索轻量化、可自更新的知识表示机制，既能保留结构化推理的优势，又能规避人工构建的高昂成本，从而在临床实用性与认知深度之间达成新的平衡。
"""
    }
    
    initial_state: ResearchState = {
        "query": "RAG in healthcare",
        "phase": "combining",
        "outline": [],
        "papers": papers,
        "extractions": [],
        "glossary": {},
        "messages": [],
        "draft_sections": draft_sections,
        "final_report": "",
        "iteration": 1,
        "max_iterations": 3,
        "review_feedback": []
    }
    
    agent = EditorAgent(
        llm_api_key=api_key,
        llm_base_url=base_url,
        model=model
    )
    
    async def run_test():
        result = await agent.process(initial_state)
        
        print("\n" + "="*80)
        print("终稿生成结果:")
        print("="*80)
        print(f"最终阶段：{result.get('phase')}")
        print(f"终稿长度：{len(result.get('final_report', ''))} 字符")
        print("\n" + "="*80)
        print("生成的终稿内容:")
        print("="*80)
        print(result.get("final_report", "No content"))
        print("="*80)
        
        # 验证是否成功完成
        if result.get("phase") == "completed":
            print("\n[OK] 测试成功：终稿生成完成！")
        else:
            print("\n[FAIL] 测试失败：阶段未更新为 completed")
    
    asyncio.run(run_test())
