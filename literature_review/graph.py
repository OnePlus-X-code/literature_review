"""
Iterative Academic Review Framework - Graph Orchestration

使用 LangGraph 编排多智能体协作流程，支持并发执行与人类确认。
"""

import os
import asyncio
import logging
from typing import Dict, Any, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import Send

from .state import ResearchState, ResearchPhase
from .agents.scout import ScoutAgent
from .agents.writer import WriterAgent
from .agents.reviewer import ReviewerAgent
from .agents.editor import EditorAgent
from .agents.supervisor import SupervisorAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


class WorkerState(TypedDict):
    """并发 Worker 的状态"""
    section_title: str
    search_query: str
    description: str


async def build_graph(llm_api_key: str, llm_base_url: str, model: str = "qwen-max") -> StateGraph:
    """
    构建并编译 LangGraph 状态图
    
    Args:
        llm_api_key: LLM API 密钥
        llm_base_url: LLM API 基础 URL
        model: 模型名称
        
    Returns:
        编译后的 StateGraph
    """
    logger.info("Building multi-agent collaboration graph with parallel execution...")
    
    # 实例化五个 Agent
    supervisor = SupervisorAgent(
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        model=model
    )
    
    scout = ScoutAgent(
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        model=model
    )
    
    writer = WriterAgent(
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        model=model
    )
    
    reviewer = ReviewerAgent(
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        model=model
    )
    
    editor = EditorAgent(
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        model=model
    )
    
    # ==========================================
    # 核心节点函数
    # ==========================================
    
    async def run_supervisor(state: ResearchState) -> Dict[str, Any]:
        """运行 Supervisor Agent"""
        logger.info(f"[SUPERVISOR] Planning research for query: {state.get('query')}")
        return await supervisor.process(state)
    
    def human_approval(state: ResearchState) -> Dict[str, Any]:
        """人类确认节点 - 交互式大纲编辑器 (Terminal-based HITL)"""
        current_outline = list(state.get("outline", []))
        current_thesis = state.get("core_thesis", "")
        messages = list(state.get("messages", []))
        
        while True:
            # 展示当前大纲
            print("\n" + "="*80)
            print(f"【大纲编辑器】核心论点：{current_thesis}")
            print("="*80)
            
            if not current_outline:
                print("[WARN] 当前大纲为空，请添加章节")
            else:
                for i, sec in enumerate(current_outline):
                    print(f"[{i}] 章节：{sec['section']}")
                    print(f"    检索：{sec['search_query']}")
                    print(f"    要求：{sec.get('description', 'N/A')}")
                    print()
            
            print("="*80)
            print("指令说明:")
            print("  [go]      - 确认并开始执行 (Confirm)")
            print("  [chat]    - 与 AI 对话修改大纲 (Chat with AI)")
            print("  [数字]    - 修改对应索引的章节 (Edit section)")
            print("  [a]       - 新增章节 (Add)")
            print("  [d]       - 删除章节 (Delete)")
            print("="*80)
            
            cmd = input("请选择指令：").strip().lower()
            
            if cmd == 'go':
                if not current_outline:
                    print("[ERROR] 大纲不能为空！请至少添加一个章节。")
                    continue
                break
            
            elif cmd == 'chat':
                # 接收用户自然语言反馈
                print("\n--- 与 AI 对话修改大纲 ---")
                feedback = input("请输入您的修改意见（例如：'第一章保留，后面的章节请聚焦于纯相位全息图的无监督生成'）：").strip()
                
                if feedback:
                    # 追加到 messages
                    messages.append({
                        "role": "user",
                        "content": feedback
                    })
                    print(f"[OK] 已记录您的反馈，将返回 Supervisor 重新规划")
                    # 返回 planning_retry 阶段，触发 Supervisor 重新生成
                    return {"messages": messages, "phase": "planning_retry"}
                else:
                    print("[ERROR] 反馈内容不能为空")
                    continue
            
            elif cmd == 'a':
                # 新增章节
                print("\n--- 新增章节 ---")
                title = input("输入新章节标题：").strip()
                if not title:
                    print("[ERROR] 章节标题不能为空")
                    continue
                
                query = input("输入精准检索词 (建议带双引号，如 \"Deep Learning\" AND \"CGH\"): ").strip()
                if not query:
                    print("[ERROR] 检索词不能为空")
                    continue
                
                desc = input("输入写作要求 (例如：重点对比不同方法的优劣): ").strip()
                if not desc:
                    desc = "综合论述该主题的核心进展"
                
                current_outline.append({
                    "section": title,
                    "search_query": query,
                    "description": desc
                })
                print(f"[OK] 已添加章节：{title}")
            
            elif cmd == 'd':
                # 删除章节
                if not current_outline:
                    print("[ERROR] 没有可删除的章节")
                    continue
                
                try:
                    idx = int(input(f"输入要删除的章节索引 (0-{len(current_outline)-1}): ").strip())
                    if 0 <= idx < len(current_outline):
                        deleted = current_outline.pop(idx)
                        print(f"[OK] 已删除章节：{deleted['section']}")
                    else:
                        print(f"[ERROR] 索引必须在 0-{len(current_outline)-1} 之间")
                except ValueError:
                    print("[ERROR] 请输入有效的数字索引")
            
            elif cmd.isdigit():
                # 修改章节
                idx = int(cmd)
                if 0 <= idx < len(current_outline):
                    sec = current_outline[idx]
                    print(f"\n--- 修改章节 [{idx}]: {sec['section']} ---")
                    
                    # 修改标题
                    new_title = input(f"新标题 (原值：{sec['section']}): ").strip()
                    if new_title:
                        sec['section'] = new_title
                    
                    # 修改检索词
                    new_query = input(f"新检索词 (原值：{sec['search_query']}): ").strip()
                    if new_query:
                        sec['search_query'] = new_query
                    
                    # 修改写作要求
                    new_desc = input(f"新写作要求 (原值：{sec.get('description', 'N/A')}): ").strip()
                    if new_desc:
                        sec['description'] = new_desc
                    
                    print(f"[OK] 已更新章节：{sec['section']}")
                else:
                    print(f"[ERROR] 索引必须在 0-{len(current_outline)-1} 之间")
            
            else:
                print("[ERROR] 未知指令，请重新输入。")
        
        # 用户确认
        print("\n" + "="*80)
        print("[OK] 大纲已锁定，正在拉起并发 Worker 小分队...")
        print("="*80)
        
        return {"outline": current_outline, "messages": messages, "phase": "retrieving"}
    
    async def run_worker(worker_state: WorkerState) -> Dict[str, Any]:
        """并发执行 Scout 和 Writer 的微型流水线"""
        logger.info(f"[WORKER] Starting parallel task for: {worker_state['section_title']}")
        
        # 构造一个局部状态，欺骗 Scout 和 Writer，让它们只专注当前章节
        local_state: ResearchState = {
            "query": worker_state["search_query"],
            "phase": "retrieving",
            "outline": [{"section": worker_state["section_title"], "description": worker_state["description"]}],
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
        
        # 1. 跑 Scout
        scout_result = await scout.process(local_state)
        local_state["papers"].update(scout_result.get("papers", {}))
        local_state["extractions"].extend(scout_result.get("extractions", []))
        
        # 2. 跑 Writer
        writer_result = await writer.process(local_state)
        
        # 3. 仅返回需要合并到全局 state 的增量数据！（利用 reducers 自动合并）
        return {
            "papers": scout_result.get("papers", {}),
            "extractions": scout_result.get("extractions", []),
            "draft_sections": writer_result.get("draft_sections", {})
        }
    
    async def run_reviewer(state: ResearchState) -> Dict[str, Any]:
        """运行 Reviewer Agent"""
        logger.info("[REVIEWER] Starting review process...")
        return await reviewer.process(state)
    
    async def run_editor(state: ResearchState) -> Dict[str, Any]:
        """运行 Editor Agent"""
        logger.info("[EDITOR] Starting final polishing...")
        return await editor.process(state)
    
    # ==========================================
    # 并发路由映射
    # ==========================================
    
    def map_sections(state: ResearchState):
        """读取 outline，动态分发 Send 对象给多个 worker"""
        return [
            Send("worker", {
                "section_title": sec["section"],
                "search_query": sec["search_query"],
                "description": sec["description"]
            }) for sec in state.get("outline", [])
        ]
    
    # ==========================================
    # 条件路由函数（修改版）
    # ==========================================
    
    def reviewer_router(state: ResearchState):
        """
        Reviewer 之后的路由决策
        
        Returns:
            "editor" - 如果通过审核，进入编辑阶段
            Send 列表 - 如果未通过，返回 worker 重新撰写（保持大纲不变）
        """
        phase = state.get("phase", "drafting")
        iteration = state.get("iteration", 0)
        draft_sections = state.get("draft_sections", {})
        papers = state.get("papers", {})
        
        logger.info(f"[ROUTER] Current phase: {phase}, iteration: {iteration}")
        logger.info(f"[ROUTER] Papers: {len(papers)}, Draft sections: {len(draft_sections)}")
        
        # 检查是否有论文和草稿
        if not papers:
            logger.error("[ROUTER] No papers found! Cannot proceed without source material.")
            # 没有论文时，无法重写，只能返回 supervisor 重新规划
            return "supervisor"
        
        if not draft_sections:
            logger.error("[ROUTER] No draft found! Cannot review empty draft.")
            # 没有草稿时，无法重写，只能返回 supervisor 重新规划
            return "supervisor"
        
        # 正常路由逻辑
        if phase == ResearchPhase.COMBINING.value:
            logger.info("[ROUTER] Review passed! Moving to Editor.")
            return "editor"
        else:
            logger.info(f"[ROUTER] Review failed! Sending back to Workers for rewrite (iteration {iteration}).")
            # 核心修复：保持大纲不变，直接将现有大纲重新分发给 Worker 重写
            return [
                Send("worker", {
                    "section_title": sec["section"],
                    "search_query": sec["search_query"],
                    "description": sec["description"]
                }) for sec in state.get("outline", [])
            ]
    
    def outline_approval_router(state: ResearchState):
        """
        Human Approval 之后的路由决策
        
        Returns:
            "supervisor" - 如果 phase 是 planning_retry，返回 Supervisor 重新规划
            Send 列表 - 否则，将大纲分发给 Worker 并发执行
        """
        if state.get("phase") == "planning_retry":
            logger.info("[ROUTER] User provided feedback, returning to Supervisor for replanning")
            return "supervisor"
        else:
            logger.info("[ROUTER] Outline approved, sending to Workers for concurrent execution")
            return [
                Send("worker", {
                    "section_title": sec["section"],
                    "search_query": sec["search_query"],
                    "description": sec["description"]
                }) for sec in state.get("outline", [])
            ]
    
    # ==========================================
    # 构建状态图
    # ==========================================
    
    workflow = StateGraph(ResearchState)
    
    # 添加节点
    workflow.add_node("supervisor", run_supervisor)
    workflow.add_node("human_approval", human_approval)
    workflow.add_node("worker", run_worker)
    workflow.add_node("reviewer", run_reviewer)
    workflow.add_node("editor", run_editor)
    
    # 设置入口点
    workflow.set_entry_point("supervisor")
    
    # 添加普通边
    workflow.add_edge("supervisor", "human_approval")
    workflow.add_edge("editor", END)
    
    # 添加条件边
    # human_approval -> outline_approval_router -> supervisor 或 worker (并发执行)
    workflow.add_conditional_edges(
        "human_approval",
        outline_approval_router,
        ["supervisor", "worker"]  # 目标节点列表
    )
    
    # worker -> reviewer (LangGraph 会自动等待所有并发 worker 完成)
    workflow.add_edge("worker", "reviewer")
    
    # reviewer -> reviewer_router -> editor 或 worker (并发重写) 或 supervisor (极端情况)
    workflow.add_conditional_edges(
        "reviewer",
        reviewer_router,
        ["editor", "worker", "supervisor"]  # 支持多种路由目标
    )
    
    logger.info("Graph compilation completed with parallel execution support!")
    
    # 编译并返回
    return workflow.compile()


async def build_headless_graph(llm_api_key: str, llm_base_url: str, model: str = "qwen-max"):
    """
    构建无需人工交互的执行图（用于 UI 界面调用）
    
    直接从 outline 开始并发执行，跳过 Supervisor 和 Human Approval 阶段
    """
    from langgraph.graph import StateGraph, END
    from langgraph.types import Send
    
    logger.info("Building headless graph for UI execution...")
    
    # 实例化 Agent
    scout = ScoutAgent(
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        model=model
    )
    
    writer = WriterAgent(
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        model=model
    )
    
    reviewer = ReviewerAgent(
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        model=model
    )
    
    editor = EditorAgent(
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        model=model
    )
    
    # ==========================================
    # 节点函数
    # ==========================================
    
    async def run_worker_node(worker_state: WorkerState) -> Dict[str, Any]:
        """并发执行 Scout 和 Writer 的微型流水线"""
        logger.info(f"[WORKER] Starting parallel task for: {worker_state['section_title']}")
        
        # 构造一个局部状态
        local_state: ResearchState = {
            "query": worker_state["search_query"],
            "phase": "retrieving",
            "outline": [{"section": worker_state["section_title"], "description": worker_state["description"]}],
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
        
        # 1. 跑 Scout
        scout_result = await scout.process(local_state)
        local_state["papers"].update(scout_result.get("papers", {}))
        local_state["extractions"].extend(scout_result.get("extractions", []))
        
        # 2. 跑 Writer
        writer_result = await writer.process(local_state)
        
        # 3. 返回需要合并到全局 state 的增量数据
        return {
            "papers": scout_result.get("papers", {}),
            "extractions": scout_result.get("extractions", []),
            "draft_sections": writer_result.get("draft_sections", {})
        }
    
    async def run_reviewer_node(state: ResearchState) -> Dict[str, Any]:
        """运行 Reviewer Agent"""
        logger.info("[REVIEWER] Starting review process...")
        return await reviewer.process(state)
    
    async def run_editor_node(state: ResearchState) -> Dict[str, Any]:
        """运行 Editor Agent"""
        logger.info("[EDITOR] Starting final polishing...")
        return await editor.process(state)
    
    # ==========================================
    # 路由函数
    # ==========================================
    
    def map_sections_headless(state: ResearchState):
        """读取 outline，动态分发 Send 对象给多个 worker"""
        return [
            Send("worker", {
                "section_title": sec["section"],
                "search_query": sec["search_query"],
                "description": sec["description"]
            }) for sec in state.get("outline", [])
        ]
    
    def headless_router(state: ResearchState):
        """
        Reviewer 之后的路由决策（无头版本）
        
        Returns:
            "editor" - 如果通过审核，进入编辑阶段
            Send 列表 - 如果未通过，返回 worker 重新撰写
        """
        phase = state.get("phase", "drafting")
        iteration = state.get("iteration", 0)
        draft_sections = state.get("draft_sections", {})
        papers = state.get("papers", {})
        
        logger.info(f"[ROUTER] Current phase: {phase}, iteration: {iteration}")
        logger.info(f"[ROUTER] Papers: {len(papers)}, Draft sections: {len(draft_sections)}")
        
        # 检查是否有论文和草稿
        if not papers:
            logger.error("[ROUTER] No papers found! Cannot proceed without source material.")
            return "editor"  # 无头模式下直接继续，避免卡住
        
        if not draft_sections:
            logger.error("[ROUTER] No draft found! Cannot review empty draft.")
            return "editor"  # 无头模式下直接继续，避免卡住
        
        # 正常路由逻辑
        if phase == ResearchPhase.COMBINING.value:
            logger.info("[ROUTER] Review passed! Moving to Editor.")
            return "editor"
        else:
            logger.info(f"[ROUTER] Review failed! Sending back to Workers for rewrite (iteration {iteration}).")
            return [
                Send("worker", {
                    "section_title": sec["section"],
                    "search_query": sec["search_query"],
                    "description": sec["description"]
                }) for sec in state.get("outline", [])
            ]
    
    # ==========================================
    # 构建状态图
    # ==========================================
    
    workflow = StateGraph(ResearchState)
    
    # 添加节点
    workflow.add_node("worker", run_worker_node)
    workflow.add_node("reviewer", run_reviewer_node)
    workflow.add_node("editor", run_editor_node)
    
    # 设置入口点（傀儡节点，用于触发 outline -> Send 转换）
    workflow.add_node("dispatcher", lambda x: {"phase": "retrieving"})
    workflow.set_entry_point("dispatcher")
    
    # 添加条件边：dispatcher -> worker (并发执行)
    workflow.add_conditional_edges(
        "dispatcher",
        map_sections_headless,
        ["worker"]
    )
    
    # worker -> reviewer
    workflow.add_edge("worker", "reviewer")
    
    # reviewer -> headless_router -> editor 或 worker
    workflow.add_conditional_edges(
        "reviewer",
        headless_router,
        ["editor", "worker"]
    )
    
    # editor -> END
    workflow.add_edge("editor", END)
    
    logger.info("Headless graph compilation completed!")
    
    # 编译并返回
    return workflow.compile()

