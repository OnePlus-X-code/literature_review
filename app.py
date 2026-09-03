"""
Multi-Agent Literature Review — Streamlit Demo

一个将"学术综述生成"拆解为 规划—检索—抽取—写作—审稿—编辑 的多智能体研究工作流。
本文件只包含 UI 与编排逻辑，不修改任何 Agent / LangGraph 核心实现。
"""

import streamlit as st
import asyncio
import html
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# 1. 代理与网络安全设置
# ==========================================
# 清除代理环境变量，防止污染 LLM / OpenAlex 请求
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# 优先加载项目根目录 .env；若不存在，回退到 literature_review/.env（本地兼容）
project_root = Path(__file__).parent
load_dotenv(dotenv_path=project_root / ".env")
load_dotenv(dotenv_path=project_root / "literature_review" / ".env")

# ==========================================
# 2. 导入项目模块
# ==========================================
from literature_review.agents.supervisor import SupervisorAgent
from literature_review.graph import build_headless_graph
from literature_review.state import ResearchState

# ==========================================
# 3. 页面配置与全局样式
# ==========================================
st.set_page_config(
    page_title="Multi-Agent Literature Review",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 白色/浅灰背景 + 黑灰正文 + 单一蓝色强调色
st.markdown("""
<style>
    #stDecoration { display: none; }

    .wf-step {
        background: #f8f9fb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 10px 6px;
        text-align: center;
    }
    .wf-num {
        color: #2563eb;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.08em;
    }
    .wf-name {
        color: #111827;
        font-size: 14px;
        font-weight: 600;
        margin-top: 2px;
    }
    .section-card {
        background: #f8f9fb;
        border: 1px solid #e5e7eb;
        border-left: 3px solid #2563eb;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .section-title {
        color: #111827;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .section-meta {
        color: #4b5563;
        font-size: 13px;
        line-height: 1.6;
    }
    .section-label {
        color: #2563eb;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .hitl-callout {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 4px solid #2563eb;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #1f2937;
        font-size: 14px;
        line-height: 1.7;
    }
    .tag-pill {
        display: inline-block;
        background: #eff6ff;
        color: #2563eb;
        border: 1px solid #bfdbfe;
        border-radius: 999px;
        padding: 3px 14px;
        font-size: 13px;
        font-weight: 500;
        margin-right: 8px;
    }
    button[kind="primary"],
    [data-testid="stBaseButton-primary"] {
        background-color: #2563eb !important;
        border-color: #2563eb !important;
    }
    button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. 初始化 Session State
# ==========================================
if "step" not in st.session_state:
    st.session_state.step = 1
if "query" not in st.session_state:
    st.session_state.query = ""
if "outline_json" not in st.session_state:
    st.session_state.outline_json = ""
if "core_thesis" not in st.session_state:
    st.session_state.core_thesis = ""
if "final_report" not in st.session_state:
    st.session_state.final_report = ""
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# ==========================================
# 5. 核心功能函数（调用既有 Agent / Graph，逻辑不变）
# ==========================================

def _initial_state(query: str, phase: str, outline: list = None) -> ResearchState:
    return {
        "query": query,
        "phase": phase,
        "outline": outline or [],
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


async def generate_outline(query: str, custom_outline: str = "") -> dict:
    """
    调用 Supervisor Agent 生成核心论点、章节大纲与每章检索词。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "")
    model = os.getenv("OPENAI_MODEL", "qwen-max")

    if not api_key or not base_url:
        st.error("请在 .env 文件中设置 DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL")
        return None

    supervisor = SupervisorAgent(
        llm_api_key=api_key,
        llm_base_url=base_url,
        model=model
    )

    initial_state = _initial_state(query, phase="planning")

    if custom_outline and custom_outline.strip():
        initial_state["messages"].append({
            "role": "user",
            "content": f"请严格按照以下我提供的大纲草案进行扩展，为每个章节配置精准的检索词：\n{custom_outline}"
        })

    return await supervisor.process(initial_state)


async def execute_research(outline: list, query: str) -> str:
    """
    执行无头研究流程（Worker 并发检索与撰写 → Reviewer 审稿迭代 → Editor 终稿）。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "")
    model = os.getenv("OPENAI_MODEL", "qwen-max")

    if not api_key or not base_url:
        st.error("请在 .env 文件中设置 DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL")
        return None

    graph = await build_headless_graph(
        llm_api_key=api_key,
        llm_base_url=base_url,
        model=model
    )

    initial_state = _initial_state(query, phase="retrieving", outline=outline)

    final_state = await graph.ainvoke(initial_state)
    return final_state.get("final_report", "")


def render_section_cards(outline: list) -> None:
    """将大纲 JSON 渲染为章节卡片（标题 / Search Query / Writing Goal）。"""
    for i, section in enumerate(outline, 1):
        title = html.escape(str(section.get("section", "Untitled Section")))
        search_query = html.escape(str(section.get("search_query", "N/A")))
        description = html.escape(str(section.get("description", "N/A")))
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title">{i}. {title}</div>
            <div class="section-meta"><span class="section-label">SEARCH QUERY</span>&nbsp;&nbsp;{search_query}</div>
            <div class="section-meta"><span class="section-label">WRITING GOAL</span>&nbsp;&nbsp;{description}</div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# 6. 页面主体
# ==========================================

st.title("Multi-Agent Literature Review")
st.markdown(
    "让多个 AI Agent 按“规划—检索—写作—审稿—编辑”的研究流程协作，"
    "而不是让单一模型一次生成整篇综述。"
)

# 顶部工作流导览
st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
workflow_steps = ["Plan", "Retrieve", "Extract", "Write", "Review", "Edit"]
wf_cols = st.columns(len(workflow_steps))
for col, (idx, name) in zip(wf_cols, enumerate(workflow_steps, 1)):
    with col:
        st.markdown(f"""
        <div class="wf-step">
            <div class="wf-num">{idx:02d}</div>
            <div class="wf-name">{name}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
st.divider()

# 侧边栏：工作流说明 + API 状态（低视觉权重）
with st.sidebar:
    st.markdown("**System Workflow**")
    st.caption(
        "Supervisor 规划核心论点与章节结构 → 人工确认大纲 → "
        "Worker 并行检索、抽取与写作 → Reviewer 统一审稿，不通过则迭代 → "
        "Editor 整合润色并生成参考文献。"
    )
    st.divider()

    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "")
    model = os.getenv("OPENAI_MODEL", "qwen-max")

    st.caption("API Status")
    if api_key and base_url:
        st.caption(":green[● Configured]")
    else:
        st.caption(":red[● Missing] — 请在 .env 中配置 DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL")
    st.caption(f"Model: `{model}`")


# ==========================================
# 步骤 1: Define the Research Question
# ==========================================
if st.session_state.step == 1:
    st.header("Step 1 · Define the Research Question")
    st.markdown("输入研究主题。系统会首先生成核心论点、章节结构与每章检索策略。")

    query = st.text_input(
        "Research Topic",
        placeholder="例如：基于神经网络的计算全息",
        label_visibility="collapsed"
    )

    st.caption(
        "Examples: 基于神经网络的计算全息 · "
        "Large Language Models for Scientific Discovery · "
        "Retrieval-Augmented Generation in Education"
    )

    with st.expander("（选填）提供你自己的大纲草案"):
        custom_outline = st.text_area(
            "自定义大纲草案（支持自然语言描述；留空则由 AI 自动规划）",
            placeholder="""例如：
1.1 研究背景与意义
1.2 传统技术进展
1.3 基于深度学习的前沿技术
1.4 面临的挑战与发展趋势""",
            height=180
        )

    if st.button("Generate Research Plan", type="primary", disabled=st.session_state.is_running):
        if not query:
            st.warning("请输入研究课题")
        else:
            st.session_state.is_running = True
            st.session_state.query = query

            with st.spinner("Supervisor is planning the core thesis, outline and search queries..."):
                try:
                    result = asyncio.run(generate_outline(query, custom_outline))

                    if result:
                        outline = result.get("outline", [])
                        st.session_state.outline_json = json.dumps(outline, ensure_ascii=False, indent=2)
                        st.session_state.core_thesis = result.get("core_thesis", "")
                        st.session_state.step = 2
                        st.rerun()
                    else:
                        st.error("大纲生成失败，请检查 API 配置")

                except Exception as e:
                    st.error(f"生成失败：{str(e)}")
                finally:
                    st.session_state.is_running = False


# ==========================================
# 步骤 2: Review the Research Plan (Human-in-the-loop)
# ==========================================
elif st.session_state.step == 2:
    st.header("Step 2 · Review the Research Plan")

    if st.button("← Back", disabled=st.session_state.is_running):
        st.session_state.step = 1
        st.rerun()

    # Core Thesis
    if st.session_state.core_thesis:
        st.subheader("Core Thesis")
        st.info(st.session_state.core_thesis)

    # 章节卡片（解析 outline_json，而非展示原始 JSON）
    st.subheader("Planned Sections")
    try:
        outline_preview = json.loads(st.session_state.outline_json)
        render_section_cards(outline_preview)
    except (json.JSONDecodeError, TypeError):
        st.warning("大纲解析失败，请在下方高级编辑器中检查 JSON 格式。")

    # Human-in-the-loop 说明
    st.markdown("""
    <div class="hitl-callout">
        <strong>Human-in-the-loop</strong><br>
        AI 已完成研究规划。请在启动文献检索与生成前确认研究结构。<br>
        你仍然可以通过下方高级编辑器修改章节、检索词和写作目标。
    </div>
    """, unsafe_allow_html=True)

    # 高级 JSON 编辑（默认折叠，避免成为页面最显眼的内容）
    with st.expander("Advanced Outline Editing (JSON)"):
        outline_text = st.text_area(
            "直接编辑大纲 JSON（章节标题 / search_query / description）",
            value=st.session_state.outline_json,
            height=320
        )

    # 执行按钮
    if st.button("Start Multi-Agent Research", type="primary", disabled=st.session_state.is_running):
        try:
            outline = json.loads(outline_text)

            if not isinstance(outline, list) or len(outline) == 0:
                st.warning("大纲必须是非空列表")
            else:
                st.session_state.is_running = True

                # 阶段说明为静态流程展示，不是实时 Agent 事件流
                st.caption(
                    "Pipeline stages: Retrieving literature → Extracting evidence → "
                    "Drafting sections → Reviewing → Editing"
                )

                with st.spinner("Research workflow is running（通常需要几分钟，取决于章节数与审稿迭代次数）..."):
                    try:
                        final_report = asyncio.run(execute_research(outline, st.session_state.query))

                        if final_report:
                            st.session_state.final_report = final_report
                            st.session_state.step = 3
                            st.rerun()
                        else:
                            st.error("生成失败：未生成最终报告")

                    except Exception as e:
                        st.error(f"执行失败：{str(e)}")
                        with st.expander("Error details"):
                            import traceback
                            st.code(traceback.format_exc())
                    finally:
                        st.session_state.is_running = False

        except json.JSONDecodeError as e:
            st.error(f"JSON 格式错误：{str(e)}")
            st.caption("请检查 JSON 格式：确保使用英文引号与逗号。")


# ==========================================
# 步骤 3: Research Complete
# ==========================================
elif st.session_state.step == 3:
    st.header("Research Complete")

    st.markdown("""
    <div style="margin-bottom: 16px;">
        <span class="tag-pill">Multi-Agent Workflow</span>
        <span class="tag-pill">Literature Retrieval</span>
        <span class="tag-pill">Review Loop</span>
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        label="Download Markdown Report",
        data=st.session_state.final_report,
        file_name=f"{st.session_state.query[:30]}_综述报告.md",
        mime="text/markdown",
        type="primary"
    )

    st.divider()
    st.markdown(st.session_state.final_report)

    st.divider()
    if st.button("Start New Research"):
        st.session_state.step = 1
        st.session_state.query = ""
        st.session_state.outline_json = ""
        st.session_state.core_thesis = ""
        st.session_state.final_report = ""
        st.rerun()


# ==========================================
# 页脚
# ==========================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #9ca3af; font-size: 13px;'>
    Multi-Agent Literature Review · LangGraph + Streamlit + OpenAlex<br>
    Supervisor → Human-in-the-loop → Workers → Reviewer → Editor
</div>
""", unsafe_allow_html=True)
