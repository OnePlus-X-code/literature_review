"""
Iterative Academic Review Framework - Main Entry Point

多智能体学术综述生成系统的主启动脚本。
"""

import os
import sys
import asyncio
import argparse
import logging
import re
from pathlib import Path

from dotenv import load_dotenv

from literature_review.graph import build_graph
from literature_review.state import ResearchState

# ==========================================
# 1. 代理与网络安全设置 (极其重要)
# ==========================================
# 在加载 .env 之前先清除代理环境变量，防止污染
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
# 注意：不要设置 NO_PROXY，让 requests 自己处理

# 加载项目根目录下的 .env 文件
project_root = Path(__file__).parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


async def run_research(query: str, max_iterations: int = 3, draft_outline: str = ""):
    """
    执行完整的学术研究流程
    
    Args:
        query: 研究问题
        max_iterations: 最大迭代次数（Writer-Reviewer 循环）
        draft_outline: 用户提供的大纲草案（可选）
    """
    logger.info("="*80)
    logger.info("启动多智能体学术综述生成系统")
    logger.info("="*80)
    
    # ==========================================
    # 1. 验证环境配置
    # ==========================================
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "")
    model = os.getenv("OPENAI_MODEL", "qwen-max")
    
    if not api_key or not base_url:
        logger.error("错误：请在 .env 文件中设置 DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL")
        sys.exit(1)
    
    logger.info(f"LLM 模型：{model}")
    logger.info(f"API 基础 URL: {base_url}")
    
    # ==========================================
    # 2. 初始化研究状态（完全干净，无任何伪造数据）
    # ==========================================
    initial_state: ResearchState = {
        "query": query,
        "phase": "planning",  # 从 planning 开始，先让 Supervisor 做规划
        "outline": [],
        "papers": {},
        "extractions": [],
        "glossary": {},
        "messages": [],
        "draft_sections": {},
        "final_report": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "review_feedback": [],
        "core_thesis": ""
    }
    
    # 如果用户提供了大纲草案，初始化到 messages 中
    if draft_outline and draft_outline.strip():
        logger.info("检测到用户提供的大纲草案，将严格遵循用户规划")
        initial_state["messages"].append({
            "role": "user",
            "content": f"请严格按照以下我提供的大纲草案进行扩展，为每个章节配置精准的检索词：\n{draft_outline}"
        })
    
    logger.info(f"研究问题：{query}")
    logger.info(f"最大迭代次数：{max_iterations}")
    if draft_outline and draft_outline.strip():
        logger.info(f"用户提供大纲草案：{len(draft_outline)} 字符")
    logger.info("="*80)
    
    # ==========================================
    # 3. 构建并运行工作流
    # ==========================================
    logger.info("正在构建多智能体协作网络...")
    graph = await build_graph(
        llm_api_key=api_key,
        llm_base_url=base_url,
        model=model
    )
    logger.info("工作流构建完成！")
    logger.info("="*80)
    
    logger.info("开始执行研究流程...")
    logger.info("阶段 1: Supervisor - 课题规划与大纲拆解")
    logger.info("阶段 2: Human Approval - 人类确认断点")
    logger.info("阶段 3: Worker (并发) - Scout 检索 + Writer 撰写")
    logger.info("阶段 4: Reviewer - 质量审查")
    logger.info("阶段 5: Editor - 润色终稿")
    logger.info("="*80)
    
    try:
        final_state = await graph.ainvoke(initial_state)
        
        logger.info("="*80)
        logger.info("研究流程完成！")
        logger.info("="*80)
        
        # ==========================================
        # 4. 保存结果
        # ==========================================
        final_report = final_state.get("final_report", "")
        
        if final_report:
            # 生成安全的文件名
            safe_query = re.sub(r'[^\w\s-]', '_', query).strip()
            safe_query = re.sub(r'[-\s]+', '_', safe_query)
            safe_query = safe_query[:50]  # 限制长度
            
            output_file = project_root / f"{safe_query}_综述报告.md"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# 学术综述：{query}\n\n")
                f.write(final_report)
            
            logger.info(f"文件已保存至：{output_file}")
            logger.info("="*80)
            
            # 打印统计信息
            logger.info("统计信息:")
            logger.info(f"  - 最终阶段：{final_state.get('phase')}")
            logger.info(f"  - 迭代次数：{final_state.get('iteration')}")
            logger.info(f"  - 检索论文数：{len(final_state.get('papers', {}))}")
            logger.info(f"  - 提取观点数：{len(final_state.get('extractions', []))}")
            logger.info(f"  - 草稿章节数：{len(final_state.get('draft_sections', {}))}")
            logger.info(f"  - 终稿长度：{len(final_report)} 字符")
            logger.info("="*80)
            
            return final_state
        else:
            logger.error("错误：未生成最终报告")
            return None
            
    except Exception as e:
        logger.error(f"研究流程失败：{e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # ==========================================
    # 配置文件处理 (research_plan.txt)
    # ==========================================
    project_root = Path(__file__).parent
    plan_file = project_root / "research_plan.txt"
    
    query = ""
    draft_outline = ""
    
    if not plan_file.exists():
        # 创建模板文件
        template = """课题：基于神经网络的计算全息

大纲草案：
(说明：如果下方留空，AI 将自动为您生成标准大纲；如果您填写了具体章节，AI 将严格按照您的规划生成检索词)

"""
        with open(plan_file, 'w', encoding='utf-8') as f:
            f.write(template)
        
        print("="*80)
        print("[OK] 已生成 research_plan.txt 模板文件")
        print("="*80)
        print(f"\n文件路径：{plan_file}")
        print("\n请在文件中填写：")
        print("  1. 课题：您的研究主题")
        print("  2. 大纲草案：（可选）具体的章节规划")
        print("\n填写完成后重新运行程序即可。")
        print("="*80)
        sys.exit(0)
    else:
        # 读取配置文件
        print("="*80)
        print("[INFO] 读取 research_plan.txt 配置文件")
        print("="*80)
        
        with open(plan_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析课题
        query_match = re.search(r'课题\s*[:：]\s*(.+?)(?=\n|$)', content)
        if query_match:
            query = query_match.group(1).strip()
            print(f"研究课题：{query}")
        else:
            print("❌ 错误：未找到'课题'字段，请检查 research_plan.txt 格式")
            sys.exit(1)
        
        # 解析大纲草案
        outline_match = re.search(r'大纲草案\s*[:：]?\s*\n(.*)', content, re.DOTALL)
        if outline_match:
            draft_outline = outline_match.group(1).strip()
            # 去掉说明文字
            draft_outline = re.sub(r'\(说明：[^)]+\)\s*', '', draft_outline)
            draft_outline = draft_outline.strip()
            
            if draft_outline:
                print(f"大纲草案：{len(draft_outline)} 字符")
                print("-"*80)
                print(draft_outline[:500] + ("..." if len(draft_outline) > 500 else ""))
                print("-"*80)
            else:
                print("大纲草案：(空) AI 将自动生成标准大纲")
        else:
            print("大纲草案：(未找到) AI 将自动生成标准大纲")
        
        print("="*80)
        print()
    
    # ==========================================
    # CLI 入口（备用，当没有配置文件时使用命令行参数）
    # ==========================================
    if not query:
        parser = argparse.ArgumentParser(
            description="多智能体学术综述生成系统",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例用法:
  python main.py
  python main.py --query "量子计算在密码学中的应用"
  python main.py --query "AI in drug discovery" --iterations 5
        """
        )
        
        parser.add_argument(
            "--query",
            type=str,
            default="基于神经网络的计算全息 (Deep Learning in Computer-Generated Holography)",
            help="研究问题（默认：基于神经网络的计算全息）"
        )
        
        parser.add_argument(
            "--iterations",
            type=int,
            default=3,
            help="最大迭代次数（Writer-Reviewer 循环，默认：3）"
        )
        
        args = parser.parse_args()
        query = args.query
        max_iterations = args.iterations
    else:
        # 使用配置文件中的课题，命令行参数无效
        max_iterations = 3
    
    # 打印启动日志
    print("\n" + "="*80)
    print("[PhD] 多智能体学术综述生成系统")
    print("="*80)
    print(f"[Query] {query}")
    print(f"[Iterations] {max_iterations}")
    if draft_outline:
        print(f"[Draft Outline] 已提供 ({len(draft_outline)} 字符)")
    else:
        print(f"[Draft Outline] 未提供 (AI 自动生成)")
    print("="*80)
    print("开始执行...\n")
    
    # 运行研究流程
    try:
        asyncio.run(run_research(query, max_iterations, draft_outline))
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n程序异常退出：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
