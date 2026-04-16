"""
Iterative Academic Review Framework - State Management Module

定义全局研究状态 (Knowledge Vault)，供所有 Agent 共享和更新。
使用 TypedDict 确保类型安全，与 LangGraph 兼容。
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from enum import Enum
import operator


class ResearchPhase(str, Enum):
    """研究阶段状态机"""
    PLANNING = "planning"        # 生成大纲，等待人类确认
    RETRIEVING = "retrieving"    # 检索文献，抓取摘要
    DRAFTING = "drafting"        # 对比驱动写作
    REVIEWING = "reviewing"      # 质量打分与驳回
    COMBINING = "combining"      # 拼接与润色
    COMPLETED = "completed"      # 完成


class Paper(BaseModel):
    """学术论文"""
    paper_id: str = Field(..., description="Semantic Scholar 的 Paper ID 或 DOI")
    title: str = Field(..., description="论文标题")
    abstract: str = Field(..., description="摘要")
    authors_list: List[str] = Field(default_factory=list, description="作者列表")
    year: Optional[int] = Field(None, description="发表年份")
    venue: Optional[str] = Field(None, description="发表 venue（期刊/会议名称，预印本可为空）")


class AcademicExtraction(BaseModel):
    """学术观点提取"""
    paper_id: str = Field(..., description="关联论文的 paper_id")
    purpose: str = Field(..., description="研究目的")
    methodology: str = Field(..., description="研究方法")
    conclusion: str = Field(..., description="主要结论")
    limitations: str = Field(default="", description="研究局限性")


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并两个字典
    
    Args:
        left: 左字典
        right: 右字典
        
    Returns:
        合并后的字典（右字典会覆盖左字典中相同的键）
    """
    return {**left, **right}


def merge_extractions(left: List[AcademicExtraction], right: List[AcademicExtraction]) -> List[AcademicExtraction]:
    """
    合并并去重提取的观点
    
    Args:
        left: 左侧提取列表
        right: 右侧提取列表
        
    Returns:
        合并后的列表（基于 paper_id 去重）
    """
    if not left:
        return right
    if not right:
        return left
    
    existing_ids = {ext.paper_id for ext in left}
    merged = list(left)
    for ext in right:
        if ext.paper_id not in existing_ids:
            merged.append(ext)
            existing_ids.add(ext.paper_id)
    return merged


class ResearchState(TypedDict):
    """
    全局研究状态 (Knowledge Vault)
    
    所有 Agent 通过读取和更新此状态来协作完成文献综述。
    每个 Agent 只应更新与其职责相关的字段，不可覆盖其他状态。
    
    使用 Annotated 和 reducer 函数支持多 Agent 并发写入：
    - List 类型字段：使用 operator.add 进行列表合并
    - Dict 类型字段：使用 merge_dicts 进行字典合并
    """
    query: str  # 研究问题
    phase: str  # 当前阶段 (ResearchPhase 的字符串值)
    outline: List[Dict[str, Any]]  # 大纲结构 [{"section": "标题", "subsections": [...]}]
    papers: Annotated[Dict[str, Paper], merge_dicts]  # 已检索的论文库，键为 paper_id
    extractions: Annotated[List[AcademicExtraction], merge_extractions]  # 从论文中提取的观点（去重合并）
    glossary: Annotated[Dict[str, str], merge_dicts]  # 术语表 {术语：解释}
    messages: Annotated[List[Dict[str, Any]], operator.add]  # Agent 对话历史
    draft_sections: Annotated[Dict[str, str], merge_dicts]  # Writer 撰写的草稿段落 {章节 ID: 内容}
    final_report: str  # Editor 最终生成的完整报告
    iteration: int  # 当前迭代次数
    max_iterations: int  # 最大迭代次数限制
    review_feedback: Annotated[List[Dict[str, Any]], operator.add]  # Reviewer 的驳回意见列表
    core_thesis: str  # Supervisor 生成的核心论点
