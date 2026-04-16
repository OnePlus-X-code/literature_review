"""
Supervisor Agent - 学术总指导

负责将用户宽泛的研究课题进行扩展和拆解，确定核心论点，
并划分为 2-3 个正文子章节，为每个子章节设计精准的布尔检索表达式。
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
    from literature_review.state import ResearchState, ResearchPhase
else:
    from .base import BaseAgent
    from ..state import ResearchState, ResearchPhase


class SupervisorAgent(BaseAgent):
    """
    学术总指导 Agent
    
    职责：
    1. 将用户宽泛的研究课题进行扩展和拆解
    2. 确定整篇综述的核心论点
    3. 划分为 2-3 个正文子章节
    4. 为每个子章节设计精准的布尔检索表达式（使用双引号锁定专有名词）
    """

    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: str,
        model: str = "qwen-max"
    ):
        super().__init__(
            name="Supervisor",
            role="学术总指导，负责课题扩展、核心论点确定与大纲拆解",
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
        query = state.get("query", "")
        if not query:
            logger.error("No query provided in state. Cannot plan without a research topic.")
            return {"phase": state.get("phase", "planning")}

        logger.info(f"Planning research for query: {query}")

        # 提取反馈历史
        messages = state.get("messages", [])
        feedback_context = ""
        if messages:
            feedback_strs = [f"- {msg['content']}" for msg in messages if msg.get("role") == "user"]
            if feedback_strs:
                feedback_context = "\n【导师（用户）的修改意见】：\n" + "\n".join(feedback_strs) + "\n\n请务必严格根据上述意见，调整并重新生成大纲！"

        system_prompt = "你是一位极具战略眼光的学术导师（Supervisor）。你的任务是为用户的研究生学位论文【绪论（研究背景与现状）】规划正文大纲。请严格输出 JSON 格式。"

        user_prompt = f"""
用户的原始研究课题是：{query}
{feedback_context}

【大纲规划严格规范】：
1. 不要规划"引言"和"总结"章节（主编会负责统一撰写）。
2. 优先级判定：如果【导师（用户）的修改意见】中提供了明确的章节结构，请**绝对优先遵循用户提供的大纲**进行细化和检索词翻译；如果用户没有提供具体大纲，则请自动规划：第 1 章必须是【基础与理论模型】（必须带 Review/Tutorial 检索限制），后续 2-3 章为细分前沿方向。
3. 为每个子章节设计一个极其精准的学术检索布尔表达式（必须使用双引号锁定专有名词，并合理使用 AND/OR）。

请严格以 JSON 格式输出，包含以下键：
{{
    "core_thesis": "整篇绪论的核心论点或研究主线（一句话精准总结）",
    "outline": [
        {{
            "section": "章节标题（第 1 章必须是基础理论）",
            "search_query": "为该章节量身定制的精准布尔检索词",
            "description": "要求 Writer 在该章节重点探讨的核心内容"
        }},
        ...
    ]
}}
"""

        logger.info("Calling LLM for research planning...")
        response = await self.call_llm(system_prompt, user_prompt, json_mode=True)
        
        parsed = self.parse_json_response(response)
        
        if not parsed:
            logger.error("Failed to parse LLM response")
            return {"phase": state.get("phase", "planning")}

        core_thesis = parsed.get("core_thesis", "")
        outline = parsed.get("outline", [])

        logger.info(f"Core thesis: {core_thesis[:100]}...")
        logger.info(f"Outline: {len(outline)} sections")
        
        for i, section in enumerate(outline, 1):
            section_title = section.get("section", "")
            search_query = section.get("search_query", "")
            logger.info(f"  Section {i}: {section_title}")
            logger.info(f"    Search query: {search_query}")

        # 状态更新
        return {
            "core_thesis": core_thesis,
            "outline": outline,
            "phase": ResearchPhase.PLANNING.value
        }


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
    # 测试：Supervisor Agent - 带反馈机制的极限测试
    # ==========================================
    print("\n" + "="*80)
    print("测试：Supervisor Agent - 学位论文绪论大纲规划（带反馈机制）")
    print("="*80)
    
    # 构造 initial_state
    initial_state: ResearchState = {
        "query": "基于神经网络的计算全息",
        "phase": "planning",
        "outline": [],
        "papers": {},
        "extractions": [],
        "glossary": {},
        "messages": [],
        "draft_sections": {},
        "final_report": "",
        "iteration": 0,
        "max_iterations": 3,
        "review_feedback": [],
        "core_thesis": ""
    }
    
    agent = SupervisorAgent(
        llm_api_key=api_key,
        llm_base_url=base_url,
        model=model
    )
    
    async def run_test():
        # ========== 第一次调用：生成初始大纲 ==========
        print("\n" + "="*80)
        print("【第一轮】生成初始大纲（无反馈）")
        print("="*80)
        print(f"\n研究课题：{initial_state['query']}")
        print("-"*80)
        
        result1 = await agent.process(initial_state)
        
        print("\n【核心论点】")
        print(f"{result1.get('core_thesis', '')}")
        
        print(f"\n【大纲拆解】（共 {len(result1.get('outline', []))} 个子章节）")
        print("-" * 80)
        
        outline1 = result1.get("outline", [])
        for i, section in enumerate(outline1, 1):
            section_title = section.get("section", "")
            search_query = section.get("search_query", "")
            description = section.get("description", "")
            
            print(f"\n第{i}章：{section_title}")
            print(f"  检索表达式：{search_query}")
            print(f"  写作要求：{description}")
            
            # 验证第 1 章是否为理论基础且包含 Review/Tutorial
            if i == 1:
                if "Review" in search_query or "Tutorial" in search_query:
                    print(f"  [✅ OK] 第 1 章为理论基础，且检索词包含 Review/Tutorial")
                else:
                    print(f"  [❌ WARN] 第 1 章检索词未包含 Review/Tutorial，建议优化")
            
            print("-" * 80)
        
        # ========== 第二次调用：带用户反馈 ==========
        print("\n" + "="*80)
        print("【第二轮】带用户反馈的修改")
        print("="*80)
        
        # 追加用户反馈
        initial_state["messages"].append({
            "role": "user",
            "content": "第一章理论部分保留，但后面的前沿方向请全部聚焦于 '纯相位全息图的无监督生成'，不要写其他方向。"
        })
        
        print(f"\n研究课题：{initial_state['query']}")
        print("\n【导师（用户）的修改意见】：")
        print(f"- {initial_state['messages'][0]['content']}")
        print("-"*80)
        
        result2 = await agent.process(initial_state)
        
        print("\n【核心论点】")
        print(f"{result2.get('core_thesis', '')}")
        
        print(f"\n【大纲拆解】（共 {len(result2.get('outline', []))} 个子章节）")
        print("-" * 80)
        
        outline2 = result2.get("outline", [])
        for i, section in enumerate(outline2, 1):
            section_title = section.get("section", "")
            search_query = section.get("search_query", "")
            description = section.get("description", "")
            
            print(f"\n第{i}章：{section_title}")
            print(f"  检索表达式：{search_query}")
            print(f"  写作要求：{description}")
            
            # 验证是否听从了用户反馈
            if i >= 2:
                if "无监督" in search_query or "unsupervised" in search_query.lower() or "pure phase" in search_query.lower() or "纯相位" in search_query:
                    print(f"  [✅ OK] 已根据用户反馈聚焦于'纯相位全息图的无监督生成'")
                else:
                    print(f"  [⚠️ WARN] 可能未完全遵循用户反馈，请检查检索词是否聚焦于'纯相位/无监督'")
            
            print("-" * 80)
        
        # ========== 总结验证 ==========
        print("\n" + "="*80)
        print("【验证总结】")
        print("="*80)
        
        # 验证第 1 章
        first_section_query = outline2[0].get("search_query", "") if outline2 else ""
        if "Review" in first_section_query or "Tutorial" in first_section_query:
            print("✅ 第 1 章：理论基础 + Review/Tutorial 检索限制 [通过]")
        else:
            print("❌ 第 1 章：未包含 Review/Tutorial 检索限制 [失败]")
        
        # 验证后续章节是否聚焦
        feedback_followed = True
        for i, section in enumerate(outline2[1:], 2):
            search_query = section.get("search_query", "").lower()
            if not ("无监督" in search_query or "unsupervised" in search_query or 
                    "pure phase" in search_query or "纯相位" in search_query):
                feedback_followed = False
                break
        
        if feedback_followed:
            print("✅ 后续章节：已聚焦于'纯相位全息图的无监督生成' [通过]")
        else:
            print("⚠️ 后续章节：可能未完全聚焦于'纯相位全息图的无监督生成' [需人工检查]")
        
        print("="*80)
        
        return result2
    
    asyncio.run(run_test())
