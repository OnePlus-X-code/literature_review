"""
Scout Agent - 学术文献侦察兵

负责调用 OpenAlex API 抓取论文，并使用 LLM 并发提取学术观点。
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from literature_review.agents.base import BaseAgent
    from literature_review.state import ResearchState, Paper, AcademicExtraction
    from literature_review.tools.openalex_search import search_papers
else:
    from .base import BaseAgent
    from ..state import ResearchState, Paper, AcademicExtraction
    from ..tools.openalex_search import search_papers


class ScoutAgent(BaseAgent):
    """
    侦察兵 Agent
    
    职责：
    1. 调用 OpenAlex API 搜索学术论文
    2. 使用 LLM 并发提取每篇论文的研究目的、方法、结论和局限性
    3. 将提取结果存入全局状态
    """

    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: str,
        model: str = "qwen-max"
    ):
        super().__init__(
            name="Scout",
            role="学术文献侦察兵，负责检索和提取论文核心观点",
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            model=model
        )

    async def _filter_papers_by_relevance(
        self,
        papers: Dict[str, Paper],
        section_title: str,
        description: str
    ) -> Dict[str, Paper]:
        """
        基于 LLM CoT（思维链）的相关性过滤机制
        
        Args:
            papers: 论文字典
            section_title: 章节标题
            description: 写作意图描述
            
        Returns:
            过滤后的论文字典
        """
        if not papers:
            return papers
        
        logger.info(f"Starting relevance filtering for section: {section_title}")
        
        # 组装候选论文列表
        candidate_papers = []
        for paper_id, paper in papers.items():
            # 截断摘要防止超长（前 800 字符）
            abstract_truncated = paper.abstract[:800] if len(paper.abstract) > 800 else paper.abstract
            
            candidate_papers.append({
                "paper_id": paper_id,
                "title": paper.title,
                "abstract": abstract_truncated
            })
        
        # 如果没有候选论文，直接返回
        if not candidate_papers:
            return {}
        
        # 构造 LLM Prompt
        system_prompt = "你是一位极其严苛的学术期刊审稿人。你的任务是根据章节标题和写作意图，审查一组候选论文的【相关性】。请执行严格的一票否决：如果论文的学科领域根本不属于本章节的专业范畴（例如本章讲光学/计算机，论文却是护理学、宏观政策等），必须打 0 分。请严格输出 JSON 格式。"
        
        # 构造候选论文列表文本
        papers_text = "\n\n".join([
            f"Paper ID: {p['paper_id']}\n标题：{p['title']}\n摘要：{p['abstract']}"
            for p in candidate_papers
        ])
        
        user_prompt = f"""【章节标题】：{section_title}

【写作意图】：{description}

【候选论文列表】：
{papers_text}

请对每篇论文进行相关性评估，并严格以 JSON 格式输出，包含一个 evaluations 列表。列表中每个对象必须包含以下字段：
- paper_id: 论文 ID（字符串）
- step1_domain_analysis: 一句话概括该论文的学科领域（例如："光学成像与全息技术"、"护理学定性研究"、"宏观经济学政策分析"）
- step2_intent_alignment: 分析该论文是否支撑本章的写作意图（是/否，并说明理由）
- score: 相关性评分（0-5 分，整数）
  - 5 分：完全匹配本章主题
  - 4 分：高度相关
  - 3 分：基本相关
  - 2 分：勉强相关
  - 1 分：几乎无关
  - 0 分：完全无关或跨学科垃圾（如护理学、宏观政策等）
- is_kept: 布尔值（score >= 3 才为 true，否则为 false）

请严格输出纯 JSON，不要任何额外解释。"""

        try:
            # 调用 LLM
            response = await self.call_llm(system_prompt, user_prompt, json_mode=True)
            parsed = self.parse_json_response(response)
            
            if not parsed or "evaluations" not in parsed:
                logger.warning("Failed to parse LLM evaluation response, keeping all papers as fallback")
                return papers
            
            evaluations = parsed.get("evaluations", [])
            
            # 构建过滤后的论文字典
            filtered_papers = {}
            kept_count = 0
            removed_count = 0
            
            for eval_item in evaluations:
                paper_id = eval_item.get("paper_id")
                score = eval_item.get("score", 0)
                is_kept = eval_item.get("is_kept", False)
                step1_domain = eval_item.get("step1_domain_analysis", "")
                step2_alignment = eval_item.get("step2_intent_alignment", "")
                
                if is_kept and score >= 3 and paper_id in papers:
                    filtered_papers[paper_id] = papers[paper_id]
                    kept_count += 1
                    logger.info(f"✅ 保留论文 [{paper_id[:8]}]: {score}分 - {step1_domain} | {step2_alignment}")
                elif paper_id in papers:
                    removed_count += 1
                    logger.warning(f"❌ 剔除论文 [{paper_id[:8]}]: {score}分 - {step1_domain} | 原因：{step2_alignment}")
            
            logger.info(f"Relevance filtering complete: 保留 {kept_count} 篇，剔除 {removed_count} 篇")
            return filtered_papers
            
        except Exception as e:
            logger.error(f"Error during relevance filtering: {e}. Keeping all papers as fallback.")
            return papers

    async def process(self, state: ResearchState) -> Dict[str, Any]:
        """
        处理状态并返回更新后的字段
        
        Args:
            state: 当前研究状态
            
        Returns:
            需要更新的字段字典
        """
        query = state.get("query", "")
        if not query:
            logger.error("No query provided in state")
            return {"papers": {}, "extractions": []}

        logger.info(f"Searching papers for query: {query}")
        
        # 使用 asyncio.to_thread 将同步调用包装为异步（Python 3.9+）
        papers_data = await asyncio.to_thread(search_papers, query, 15)
        
        if not papers_data:
            logger.warning("No papers found from OpenAlex")
            return {"papers": {}, "extractions": []}

        logger.info(f"Found {len(papers_data)} papers from OpenAlex")

        # 将 OpenAlex 数据转换为 Paper 对象字典
        initial_papers: Dict[str, Paper] = {}
        for paper_data in papers_data:
            paper_id = paper_data.get("paper_id", "")
            title = paper_data.get("title", "")
            abstract = paper_data.get("abstract", "")
            
            if not abstract:
                logger.warning(f"Paper {paper_id} has no abstract, skipping")
                continue
            
            paper_obj = Paper(
                paper_id=paper_id,
                title=title,
                abstract=abstract,
                authors_list=paper_data.get("authors", []),
                year=paper_data.get("year"),
                venue=paper_data.get("venue")
            )
            initial_papers[paper_id] = paper_obj
        
        # 【关键步骤】基于 LLM CoT 的相关性过滤
        # 从 outline 中提取当前章节的标题和写作意图
        outline = state.get("outline", [])
        section_title = ""
        description = ""
        
        if outline and len(outline) > 0:
            section_title = outline[0].get("section", "")
            description = outline[0].get("description", "")
        
        # 如果有章节信息，执行相关性过滤
        if section_title and description:
            logger.info(f"Applying LLM-based relevance filtering for section: {section_title}")
            filtered_papers = await self._filter_papers_by_relevance(
                initial_papers,
                section_title,
                description
            )
            
            # 如果过滤后为空，提前结束
            if not filtered_papers:
                logger.error(f"All papers filtered out as irrelevant for section: {section_title}. Stopping this worker.")
                return {"papers": {}, "extractions": []}
            
            logger.info(f"Relevance filtering complete: {len(filtered_papers)} papers kept")
        else:
            logger.warning("No section title or description found in outline, skipping relevance filtering")
            filtered_papers = initial_papers

        logger.info(f"Starting LLM extraction for {len(filtered_papers)} filtered papers...")

        async def _extract_single_paper_from_filtered(paper_id: str, paper: Paper) -> Tuple[Optional[Paper], Optional[AcademicExtraction]]:
            """
            为单篇过滤后的论文提取学术观点
            
            Args:
                paper_id: 论文 ID
                paper: Paper 对象
                
            Returns:
                (Paper 实例，AcademicExtraction 实例)
            """
            try:
                if not paper.abstract:
                    logger.warning(f"Paper {paper_id} has no abstract, skipping")
                    return None, None

                system_prompt = "你是一位严谨的学术文献分析专家。请仔细阅读摘要，并严格以 JSON 格式输出分析结果。如果摘要中没有明确提及某项内容，请如实填写'未提及'，绝不要凭空捏造。"
                
                user_prompt = f"论文标题：{paper.title}\n\n摘要内容：{paper.abstract}\n\n请分析上述摘要，输出 JSON，必须包含以下四个键（全部小写字符串）：\n- purpose: 该研究的核心目标或试图解决的问题是什么？\n- methodology: 该研究使用了什么具体的方法、模型、算法或数据集？\n- conclusion: 该研究得出的最核心结论或实验结果是什么？\n- limitations: 作者明确指出的局限性，或者你基于专业知识发现的潜在缺陷是什么？"

                response = await self.call_llm(system_prompt, user_prompt, json_mode=True)
                parsed = self.parse_json_response(response)

                if not parsed:
                    logger.warning(f"Failed to parse LLM response for paper {paper_id}")
                    return paper, None

                extraction_obj = AcademicExtraction(
                    paper_id=paper_id,
                    purpose=parsed.get("purpose", "未提及"),
                    methodology=parsed.get("methodology", "未提及"),
                    conclusion=parsed.get("conclusion", "未提及"),
                    limitations=parsed.get("limitations", "未提及")
                )

                logger.info(f"Successfully extracted: {paper.title[:50]}...")
                return paper, extraction_obj

            except Exception as e:
                logger.error(f"Error extracting paper: {e}")
                return None, None

        # 为过滤后的论文创建提取任务
        tasks = [_extract_single_paper_from_filtered(paper_id, paper) for paper_id, paper in filtered_papers.items()]
        results = await asyncio.gather(*tasks)

        new_papers: Dict[str, Paper] = {}
        new_extractions: List[AcademicExtraction] = []

        for paper_obj, extraction_obj in results:
            if paper_obj and extraction_obj:
                new_papers[paper_obj.paper_id] = paper_obj
                new_extractions.append(extraction_obj)

        state["papers"].update(new_papers)
        state["extractions"].extend(new_extractions)

        logger.info(f"Extraction complete: {len(new_papers)} papers, {len(new_extractions)} extractions")

        return {
            "papers": state["papers"],
            "extractions": state["extractions"]
        }


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # ==========================================
    # 代理与网络安全设置
    # ==========================================
    # 清除代理环境变量，让 aiohttp 直接连接
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)

    async def test_relevance_filter():
        """测试基于 LLM CoT 的相关性过滤机制"""
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        base_url = os.getenv("DASHSCOPE_BASE_URL", "")
        model = os.getenv("OPENAI_MODEL", "qwen-max")

        if not api_key or not base_url:
            print("Error: Please set DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL in .env")
            return

        scout = ScoutAgent(
            llm_api_key=api_key,
            llm_base_url=base_url,
            model=model
        )

        print("=" * 80)
        print("测试：基于 LLM CoT 的相关性过滤机制")
        print("=" * 80)
        print()

        # 模拟章节信息
        section_title = "物理感知神经网络在计算全息中的应用"
        description = "探讨如何将物理衍射模型（如 ASM）作为神经网络的固定层，摆脱数据集依赖。"

        # 硬编码伪造 3 篇 Paper
        from literature_review.state import Paper
        
        test_papers = {
            # 论文 1：真实相关的光学论文
            "paper_1_good": Paper(
                paper_id="paper_1_good",
                title="Physics-informed deep learning for computational holography",
                abstract="We present a physics-informed neural network approach for holographic phase retrieval. Our method embeds the angular spectrum method (ASM) as a fixed differentiable layer within the network architecture. By leveraging unsupervised learning with wave propagation constraints, we eliminate the need for paired training datasets. Experimental results demonstrate superior reconstruction quality compared to purely data-driven methods.",
                authors_list=["Zhang, Wei", "Li, Jing"],
                year=2023,
                venue="Optics Express"
            ),
            # 论文 2：跨学科垃圾数据 - 宏观经济学
            "paper_2_bad_econ": Paper(
                paper_id="paper_2_bad_econ",
                title="Impact of AI on UN Sustainable Development Goals and Energy Consumption",
                abstract="This study examines the macroeconomic implications of artificial intelligence adoption on sustainable development. Using panel data from 150 countries, we analyze how AI-driven automation affects energy consumption patterns and economic growth. Our findings suggest that AI implementation requires careful policy coordination to balance technological progress with environmental sustainability goals.",
                authors_list=["Smith, John", "Johnson, Mary"],
                year=2022,
                venue="Journal of Environmental Economics"
            ),
            # 论文 3：跨学科垃圾数据 - 护理学
            "paper_3_bad_nursing": Paper(
                paper_id="paper_3_bad_nursing",
                title="Development of Semi-Structured Interviews in Nursing Research",
                abstract="This qualitative study explores the methodology of semi-structured interviews in nursing practice. Through thematic analysis of healthcare worker experiences, we identify key themes in patient care communication. Our findings highlight the importance of empathetic listening and cultural sensitivity in clinical settings. This research contributes to nursing education and patient-centered care frameworks.",
                authors_list=["Brown, Sarah", "Davis, Emily"],
                year=2021,
                venue="Journal of Nursing Research"
            )
        }

        print(f"【章节标题】：{section_title}")
        print(f"【写作意图】：{description}")
        print()
        print(f"【候选论文】：{len(test_papers)} 篇")
        for pid, paper in test_papers.items():
            print(f"  - [{pid[:8]}] {paper.title[:60]}...")
        print()
        print("=" * 80)
        print("开始相关性过滤...")
        print("=" * 80)
        print()

        # 运行过滤器
        filtered_papers = await scout._filter_papers_by_relevance(
            test_papers,
            section_title,
            description
        )

        print()
        print("=" * 80)
        print("过滤结果")
        print("=" * 80)
        print(f"原始论文数：{len(test_papers)}")
        print(f"过滤后论文数：{len(filtered_papers)}")
        print()
        
        print("保留的论文：")
        for pid in filtered_papers:
            print(f"  ✅ [{pid[:8]}] {test_papers[pid].title[:60]}...")
        
        print()
        print("剔除的论文：")
        for pid in test_papers:
            if pid not in filtered_papers:
                print(f"  ❌ [{pid[:8]}] {test_papers[pid].title[:60]}...")
        
        print()
        print("=" * 80)
        print("验证结果：")
        print("=" * 80)
        
        # 验证预期结果
        expected_kept = ["paper_1_good"]
        expected_removed = ["paper_2_bad_econ", "paper_3_bad_nursing"]
        
        success = True
        for pid in expected_kept:
            if pid not in filtered_papers:
                print(f"❌ 错误：预期保留的论文 [{pid}] 被剔除")
                success = False
        
        for pid in expected_removed:
            if pid in filtered_papers:
                print(f"❌ 错误：预期剔除的论文 [{pid}] 被保留")
                success = False
        
        if success:
            print("✅ 测试通过！LLM 成功识别并剔除了跨学科垃圾论文。")
        else:
            print("❌ 测试失败！请检查 LLM 的判断逻辑。")
        
        print("=" * 80)

    # 运行测试
    asyncio.run(test_relevance_filter())
