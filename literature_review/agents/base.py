"""
Iterative Academic Review Framework - Base Agent Module

所有专家 Agent 的基类，提供通用的 LLM 调用、JSON 容错解析等功能。
"""

import json
import logging
import asyncio
import time
import os
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from ..state import ResearchState

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')


class BaseAgent(ABC):
    """
    Agent 基类

    所有专家 Agent 继承此类，实现特定的 process 方法。
    """

    def __init__(
        self,
        name: str,
        role: str,
        llm_api_key: str,
        llm_base_url: str,
        model: str = "qwen-max"
    ):
        self.name = name
        self.role = role
        self.model = model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        
        self.logger = logging.getLogger(f"Agent.{name}")

    @abstractmethod
    async def process(self, state: ResearchState) -> Dict[str, Any]:
        """
        处理状态并返回需要更新的字段

        Args:
            state: 当前研究状态

        Returns:
            需要更新的字段字典（只返回变化的部分）
        """
        pass

    async def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 16000
    ) -> str:
        """
        异步调用 LLM（使用 requests 库直接调用，绕过 OpenAI SDK 的代理问题）

        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            json_mode: 是否强制 JSON 输出
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            LLM 响应文本
        """
        start_time = time.time()

        try:
            # 构建请求体
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_api_key}"
            }

            # 创建 Session，强制绕过代理
            session = requests.Session()
            session.trust_env = True  # 忽略环境变量代理
            proxies = None  # 显式不使用代理

            # 使用 asyncio.to_thread 将同步调用包装为异步（Python 3.9+）
            response = await asyncio.to_thread(
                session.post,
                f"{self.llm_base_url}/chat/completions",
                headers=headers,
                json=payload,
                proxies=proxies,
                timeout=300  # Editor 阶段需要整合大量内容，延长超时至 300 秒
            )

            if response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")

            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            duration = int((time.time() - start_time) * 1000)
            self.logger.info(f"LLM call completed in {duration}ms, response length: {len(content)}")

            return content

        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """安全解析 JSON 响应，处理 markdown 代码块和格式问题"""
        import re

        def _fix_escaped_values(obj: Any, key: str = None) -> Any:
            """
            递归修复字典和列表中的转义字符

            注意：对于 'code' 字段，不处理转义，因为代码中的 \n 是有意义的转义序列
            """
            if isinstance(obj, dict):
                return {k: _fix_escaped_values(v, key=k) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_fix_escaped_values(item, key=key) for item in obj]
            elif isinstance(obj, str):
                if key in ('code', 'fixed_code', 'revised_content'):
                    return obj

                result = obj
                result = result.replace('\\\\n', '\n')
                result = result.replace('\\n', '\n')
                result = result.replace('\\\\r', '\r')
                result = result.replace('\\r', '\r')
                result = result.replace('\\\\t', '\t')
                result = result.replace('\\t', '\t')
                return result
            else:
                return obj

        def try_parse(s: str) -> Optional[Dict]:
            """尝试解析 JSON，包含修复逻辑"""
            s = s.strip()
            if s.startswith('\ufeff'):
                s = s[1:]

            try:
                result = json.loads(s)
                return _fix_escaped_values(result)
            except json.JSONDecodeError:
                pass

            try:
                s = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', '', s)
                s = re.sub(r'//.*?$', '', s, flags=re.MULTILINE)
                s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
                s = re.sub(r',(\s*[}\]])', r'\1', s)
                s = re.sub(r'([}\]])(\s*)([{\[])', r'\1,\2\3', s)
                s = re.sub(r'(\{|\,)\s*(\w+)\s*:', r'\1"\2":', s)
                result = json.loads(s)
                return _fix_escaped_values(result)
            except json.JSONDecodeError:
                pass

            return None

        result = try_parse(response)
        if result:
            self.logger.debug("Direct JSON parse succeeded")
            return result

        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(code_block_pattern, response)
        if match:
            result = try_parse(match.group(1))
            if result:
                self.logger.debug("Extracted JSON from code block")
                return result

        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            result = try_parse(response[start:end+1])
            if result:
                self.logger.debug("Extracted JSON from braces")
                return result

        try:
            import ast
            s = response
            s = re.sub(r'\btrue\b', 'True', s)
            s = re.sub(r'\bfalse\b', 'False', s)
            s = re.sub(r'\bnull\b', 'None', s)
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1:
                result = ast.literal_eval(s[start:end+1])
                if isinstance(result, dict):
                    self.logger.debug("Parsed using ast.literal_eval")
                    return result
        except:
            pass

        self.logger.error(f"JSON parse error, could not extract valid JSON")
        self.logger.warning(f"Raw response (first 800 chars): {response[:800]}")
        return {}

    def add_message(self, state: ResearchState, event_type: str, content: Any) -> None:
        """
        添加消息到状态

        Args:
            state: 研究状态
            event_type: 事件类型
            content: 消息内容
        """
        message = {
            "type": event_type,
            "agent": self.name,
            "content": content
        }
        state["messages"].append(message)


class AgentRegistry:
    """Agent 注册表"""

    _agents: Dict[str, BaseAgent] = {}

    @classmethod
    def register(cls, agent: BaseAgent) -> None:
        """注册 Agent"""
        cls._agents[agent.name] = agent

    @classmethod
    def get(cls, name: str) -> Optional[BaseAgent]:
        """获取 Agent"""
        return cls._agents.get(name)
