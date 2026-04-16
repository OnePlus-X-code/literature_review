"""
Streamlit 可视化前端 - 多智能体学术综述生成引擎

提供交互式界面，支持：
1. 大纲规划（Supervisor Agent）
2. 大纲编辑（JSON 格式）
3. 并发执行（Worker 小分队）
4. 最终报告展示与下载
"""

import streamlit as st
import asyncio
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# 1. 代理与网络安全设置
# ==========================================
# 清除代理环境变量
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# 加载 .env 文件
project_root = Path(__file__).parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# ==========================================
# 2. 导入项目模块
# ==========================================
from literature_review.agents.supervisor import SupervisorAgent
from literature_review.graph import build_headless_graph
from literature_review.state import ResearchState

# ==========================================
# 3. 页面配置
# ==========================================
st.set_page_config(
    page_title="多智能体学术综述生成引擎",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 4. 初始化 Session State
# ==========================================
if "step" not in st.session_state:
    st.session_state.step = 1
if "query" not in st.session_state:
    st.session_state.query = ""
if "outline_json" not in st.session_state:
    st.session_state.outline_json = ""
if "final_report" not in st.session_state:
    st.session_state.final_report = ""
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# ==========================================
# 5. 核心功能函数
# ==========================================

async def generate_outline(query: str, custom_outline: str = "") -> dict:
    """
    调用 Supervisor Agent 生成学术大纲
    
    Args:
        query: 研究课题
        custom_outline: 用户自定义大纲（可选）
    
    Returns:
        Supervisor 返回的大纲字典
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "")
    model = os.getenv("OPENAI_MODEL", "qwen-max")
    
    if not api_key or not base_url:
        st.error("❌ 错误：请在 .env 文件中设置 DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL")
        return None
    
    # 初始化 Supervisor
    supervisor = SupervisorAgent(
        llm_api_key=api_key,
        llm_base_url=base_url,
        model=model
    )
    
    # 构造初始状态
    initial_state: ResearchState = {
        "query": query,
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
    
    # 如果用户提供了自定义大纲，添加到 messages 中
    if custom_outline and custom_outline.strip():
        initial_state["messages"].append({
            "role": "user",
            "content": f"请严格按照以下我提供的大纲草案进行扩展，为每个章节配置精准的检索词：\n{custom_outline}"
        })
    
    # 调用 Supervisor
    result = await supervisor.process(initial_state)
    
    return result


async def execute_research(outline: list, query: str) -> str:
    """
    执行无头研究流程（并发检索 + 撰写 + 审稿 + 编辑）
    
    Args:
        outline: 大纲列表
        query: 研究课题
    
    Returns:
        最终生成的综述报告
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "")
    model = os.getenv("OPENAI_MODEL", "qwen-max")
    
    if not api_key or not base_url:
        st.error("❌ 错误：请在 .env 文件中设置 DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL")
        return None
    
    # 构建无头执行图
    graph = await build_headless_graph(
        llm_api_key=api_key,
        llm_base_url=base_url,
        model=model
    )
    
    # 构造初始状态
    initial_state: ResearchState = {
        "query": query,
        "phase": "retrieving",
        "outline": outline,
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
    
    # 执行研究流程
    final_state = await graph.ainvoke(initial_state)
    
    return final_state.get("final_report", "")


# ==========================================
# 6. 页面主体
# ==========================================

st.title("🎓 多智能体学术综述生成引擎")
st.markdown("---")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置")
    
    # 显示 API 配置状态
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "")
    
    if api_key and base_url:
        st.success("✅ API 配置正常")
    else:
        st.error("❌ API 配置缺失")
        st.info("请在 .env 文件中设置 DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL")
    
    st.markdown("---")
    st.markdown("**工作流程**：")
    st.markdown("1. 📝 输入研究课题")
    st.markdown("2. 🤖 AI 生成学术大纲")
    st.markdown("3. ✏️ 编辑优化大纲")
    st.markdown("4. 🚀 并发执行检索与撰写")
    st.markdown("5. 📄 查看并下载综述报告")


# 步骤指示器
col1, col2 = st.columns(2)
with col1:
    if st.session_state.step >= 1:
        st.success("**步骤 1: 大纲规划** ✅")
    else:
        st.info("**步骤 1: 大纲规划**")

with col2:
    if st.session_state.step >= 2:
        st.success("**步骤 2: 执行生成** ✅")
    else:
        st.info("**步骤 2: 执行生成**")

st.markdown("---")

# ==========================================
# 步骤 1: 大纲规划
# ==========================================
if st.session_state.step == 1:
    st.header("📝 步骤 1: 大纲规划")
    
    # 研究课题输入
    query = st.text_input(
        "研究课题",
        placeholder="例如：基于神经网络的计算全息",
        help="输入您的研究主题"
    )
    
    # 自定义大纲输入
    custom_outline = st.text_area(
        "用户自定义大纲（选填，支持自然语言描述）",
        placeholder="""例如：
1.1 研究背景与意义
1.2 传统技术进展
1.3 基于深度学习的前沿技术
1.4 面临的挑战与发展趋势""",
        height=200,
        help="如果留空，AI 将自动生成标准大纲；如果填写，AI 将严格遵循您的规划"
    )
    
    # 生成大纲按钮
    if st.button("⚡ 生成学术大纲", type="primary", disabled=st.session_state.is_running):
        if not query:
            st.warning("⚠️ 请输入研究课题")
        else:
            st.session_state.is_running = True
            st.session_state.query = query
            
            with st.spinner("🤖 Supervisor 正在规划学术大纲，请稍候..."):
                try:
                    # 调用 Supervisor
                    result = asyncio.run(generate_outline(query, custom_outline))
                    
                    if result:
                        # 提取大纲并转换为 JSON
                        outline = result.get("outline", [])
                        core_thesis = result.get("core_thesis", "")
                        
                        # 转换为格式化的 JSON 字符串
                        outline_json = json.dumps(outline, ensure_ascii=False, indent=2)
                        st.session_state.outline_json = outline_json
                        
                        # 显示核心论点
                        st.success("✅ 大纲生成成功！")
                        st.subheader("💡 核心论点")
                        st.info(core_thesis)
                        
                        # 显示大纲预览
                        st.subheader("📋 大纲预览")
                        for i, section in enumerate(outline, 1):
                            with st.expander(f"第{i}章：{section.get('section', '未知章节')}"):
                                st.markdown(f"**检索词**: `{section.get('search_query', 'N/A')}`")
                                st.markdown(f"**写作要求**: {section.get('description', 'N/A')}")
                        
                        # 进入步骤 2
                        st.session_state.step = 2
                        st.rerun()
                    else:
                        st.error("❌ 大纲生成失败，请检查 API 配置")
                
                except Exception as e:
                    st.error(f"❌ 生成失败：{str(e)}")
                finally:
                    st.session_state.is_running = False


# ==========================================
# 步骤 2: 大纲编辑与执行
# ==========================================
elif st.session_state.step == 2:
    st.header("✏️ 步骤 2: 大纲编辑与执行")
    
    # 返回上一步按钮
    if st.button("⬅️ 返回上一步"):
        st.session_state.step = 1
        st.rerun()
    
    st.markdown("### 📋 大纲编辑器")
    st.info("💡 提示：您可以直接修改下方的 JSON，调整章节标题、检索词或写作要求")
    
    # JSON 编辑器
    outline_json = st.text_area(
        "大纲编辑器 (JSON 格式，可直接修改检索词与章节)",
        value=st.session_state.outline_json,
        height=400,
        help="支持直接编辑 JSON，修改章节标题、检索词、写作要求等"
    )
    
    # 执行按钮
    if st.button("🚀 确认大纲，全面开工", type="primary", disabled=st.session_state.is_running):
        try:
            # 解析 JSON
            outline = json.loads(outline_json)
            
            if not isinstance(outline, list) or len(outline) == 0:
                st.warning("⚠️ 大纲必须是非空列表")
            else:
                st.session_state.is_running = True
                
                # 显示执行进度
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("🕵️‍♂️ 多个学术小分队并发检索中，请耐心等待 (约 2-3 分钟)..."):
                    try:
                        # 更新进度
                        status_text.text("📊 阶段 1/4: 并发检索文献...")
                        progress_bar.progress(25)
                        
                        # 执行研究流程
                        final_report = asyncio.run(execute_research(outline, st.session_state.query))
                        
                        if final_report:
                            # 更新进度
                            status_text.text("✅ 完成！")
                            progress_bar.progress(100)
                            
                            # 保存最终报告
                            st.session_state.final_report = final_report
                            
                            st.success("🎉 综述生成完成！")
                            
                            # 显示报告
                            st.markdown("---")
                            st.subheader("📄 学术综述报告")
                            st.markdown(final_report)
                            
                            # 下载按钮
                            st.download_button(
                                label="📥 下载 Markdown 文件",
                                data=final_report,
                                file_name=f"{st.session_state.query[:30]}_综述报告.md",
                                mime="text/markdown"
                            )
                            
                            # 重置状态
                            st.session_state.step = 2  # 保持在步骤 2
                        
                        else:
                            st.error("❌ 生成失败：未生成最终报告")
                    
                    except Exception as e:
                        st.error(f"❌ 执行失败：{str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                    
                    finally:
                        st.session_state.is_running = False
                
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON 格式错误：{str(e)}")
            st.info("💡 请检查 JSON 格式是否正确，确保使用英文引号和逗号")


# ==========================================
# 页脚
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎓 多智能体学术综述生成引擎 | 基于 LangGraph + Streamlit</p>
    <p>支持：Supervisor 大纲规划 | Worker 并发检索 | Reviewer 质量审查 | Editor 最终润色</p>
</div>
""", unsafe_allow_html=True)
