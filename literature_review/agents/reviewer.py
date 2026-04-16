"""
Reviewer Agent - 学术综述审稿专家

负责对 Writer 生成的文献综述草稿进行质量审查，严查幻觉、引用格式和综述深度。
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
    from literature_review.state import ResearchState, ResearchPhase, AcademicExtraction
else:
    from .base import BaseAgent
    from ..state import ResearchState, ResearchPhase, AcademicExtraction


class ReviewerAgent(BaseAgent):
    """
    审稿专家 Agent
    
    职责：
    1. 对 Writer 生成的文献综述草稿进行质量审查
    2. 检查幻觉、引用格式、综述深度
    3. 决定是否通过或打回重写
    """

    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: str,
        model: str = "qwen-max"
    ):
        super().__init__(
            name="Reviewer",
            role="顶刊审稿专家 (Reviewer 2)，负责审查文献综述质量",
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
        extractions = state.get("extractions", [])
        
        if not draft_sections:
            logger.error("No draft sections found. Cannot review empty draft.")
            return {"phase": state.get("phase", "drafting")}

        logger.info(f"Reviewing {len(draft_sections)} draft section(s)")

        # 迭代控制
        iteration = state.get("iteration", 0) + 1
        max_iterations = state.get("max_iterations", 3)
        
        logger.info(f"Current iteration: {iteration}/{max_iterations}")

        # 如果草稿为空，直接返回
        if not draft_sections:
            logger.warning("Empty draft sections, skipping review")
            return {"phase": state.get("phase", "drafting")}

        # 组装审核素材
        facts_text = self._prepare_facts(extractions)
        draft_text = self._prepare_draft(draft_sections)

        system_prompt = "你是一位极其严苛的顶刊审稿专家 (Reviewer 2)。你的任务是对作者提交的文献综述草稿进行审查，严查幻觉和引用格式。请严格输出 JSON 格式。"

        user_prompt = f"""
【金标准事实数据】：
{facts_text}

【作者提交的草稿】：
{draft_text}

请根据以上数据进行审查。

审查标准：
1. 幻觉检查：草稿中提到的所有数据、方法、结论，必须在事实数据中有对应支撑。
2. 引用规范：草稿中是否对每个核心结论都进行了引用标记？（允许使用 作者年份、标题片段 或 paper_id 进行引用，只要没有凭空捏造即可）。
3. 综述深度：是否仅仅是简单的罗列（A 说了啥，B 说了啥），而没有进行对比和综合？

请以 JSON 格式输出你的审稿结果：
{{
    "passed": true 或 false,
    "feedback": "如果不通过，请在这里给出极其具体、尖锐的修改意见，指出具体哪一段哪一句话有问题。如果通过，请回复'Accepted'。"
}}
"""

        logger.info("Calling LLM for review...")
        response = await self.call_llm(system_prompt, user_prompt, json_mode=True)
        
        review_result = self.parse_json_response(response)
        passed = review_result.get("passed", False)
        feedback = review_result.get("feedback", "No feedback provided")

        logger.info(f"Review result: passed={passed}")
        logger.info(f"Feedback: {feedback[:200]}...")

        # 状态路由更新
        review_feedback = state.get("review_feedback", [])
        
        # 检查是否达到最大迭代次数
        if iteration >= max_iterations:
            logger.warning(f"Reached max iterations ({max_iterations}). Forcing phase to COMBINING.")
            next_phase = ResearchPhase.COMBINING.value
            feedback_with_warning = f"[⚠️ 已达到最大迭代次数 {max_iterations}，强制进入下一阶段] {feedback}"
        else:
            if passed:
                logger.info("Review passed! Moving to COMBINING phase.")
                next_phase = ResearchPhase.COMBINING.value
                feedback_with_warning = feedback
            else:
                logger.info("Review failed. Sending back to DRAFTING phase.")
                next_phase = ResearchPhase.DRAFTING.value
                feedback_with_warning = feedback

        # 如果未通过，将 feedback 加入历史记录
        if not passed:
            review_feedback.append({
                "iteration": iteration,
                "feedback": feedback,
                "phase": "reviewing"
            })

        return {
            "phase": next_phase,
            "iteration": iteration,
            "review_feedback": review_feedback
        }

    def _prepare_facts(self, extractions: List[AcademicExtraction]) -> str:
        """
        将 extractions 转换成字符串（Ground Truth 金标准）
        
        Args:
            extractions: 学术观点提取列表
            
        Returns:
            格式化的事实文本
        """
        if not extractions:
            return "无可用事实数据"

        facts_parts = []
        for extraction in extractions:
            facts_parts.append(f"""
---
论文 ID: {extraction.paper_id}
研究目的：{extraction.purpose}
研究方法：{extraction.methodology}
主要结论：{extraction.conclusion}
局限性：{extraction.limitations or 'N/A'}
---
""")
        
        return "\n".join(facts_parts)

    def _prepare_draft(self, draft_sections: Dict[str, str]) -> str:
        """
        将 draft_sections 里的草稿整合成字符串
        
        Args:
            draft_sections: 草稿段落字典
            
        Returns:
            整合后的草稿文本
        """
        if not draft_sections:
            return "无草稿内容"

        draft_parts = []
        for section_title, content in draft_sections.items():
            draft_parts.append(f"## {section_title}\n\n{content}")
        
        return "\n\n".join(draft_parts)


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
    # 测试用例 1：坏草稿测试
    # ==========================================
    print("\n" + "="*80)
    print("测试用例 1：坏草稿测试（伪造数据 + 无引用 + 流水账）")
    print("="*80)
    
    # 构造一个包含伪造数据的坏草稿
    bad_draft = """
## 综合文献综述

近年来，RAG 在医疗领域取得了巨大进展。Smith 等人提出了一种基于 RAG 的医疗问答系统，准确率达到 99%，显著优于所有现有方法。

Johnson 等人开发了一个多模态 RAG 系统，可以同时处理文本和医学影像数据，在肺癌检测任务中达到了 98.5% 的敏感性。

Williams 等人则专注于药物发现领域，他们的 RAG 系统成功预测了 1000 多种新药物的活性，准确率高达 97%。

总的来说，这些研究都表明 RAG 技术在医疗领域有着广泛的应用前景。
"""
    
    # 构造真实的 extractions（金标准）
    from literature_review.state import AcademicExtraction
    
    extractions = [
        AcademicExtraction(
            paper_id="paper1",
            purpose="开发基于 RAG 的医疗问答系统",
            methodology="使用 Dense Passage Retrieval (DPR) 从 PubMed 检索相关段落，并使用 BART 作为生成器",
            conclusion="在标准医疗问答任务上实现了 85% 的准确率，显著优于传统关键词检索基线",
            limitations="完全依赖非结构化文本检索，难以捕捉医学概念间的深层语义关系"
        ),
        AcademicExtraction(
            paper_id="paper2",
            purpose="引入结构化知识图谱增强 RAG 的推理能力",
            methodology="构建医学知识图谱（KG），将医学实体及其关系显式建模，并与向量检索结果联合输入至 GPT-4 生成器",
            conclusion="在复杂临床推理任务中 F1 分数较纯向量检索提升 12%",
            limitations="知识图谱构建高度依赖专家人工验证，面临可扩展性瓶颈"
        )
    ]
    
    initial_state: ResearchState = {
        "query": "RAG in healthcare",
        "phase": "reviewing",
        "outline": [],
        "papers": {},
        "extractions": extractions,
        "glossary": {},
        "messages": [],
        "draft_sections": {
            "综合文献综述": bad_draft
        },
        "final_report": "",
        "iteration": 0,
        "max_iterations": 3,
        "review_feedback": []
    }
    
    agent = ReviewerAgent(
        llm_api_key=api_key,
        llm_base_url=base_url,
        model=model
    )
    
    async def run_test():
        result = await agent.process(initial_state)
        
        print("\n" + "="*80)
        print("审查结果:")
        print("="*80)
        print(f"下一阶段：{result.get('phase')}")
        print(f"迭代次数：{result.get('iteration')}")
        print(f"审查反馈：{result.get('review_feedback', [])[-1] if result.get('review_feedback') else 'N/A'}")
        print("="*80)
        
        # 验证是否成功驳回
        if result.get("phase") == "drafting":
            print("\n[OK] 测试成功：坏草稿被成功驳回！")
        else:
            print("\n[FAIL] 测试失败：坏草稿竟然通过了审查！")
    
    asyncio.run(run_test())
