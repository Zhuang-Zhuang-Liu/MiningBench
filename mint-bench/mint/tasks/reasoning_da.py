import json
import os
import re
from datetime import datetime
from typing import Any

from mint.prompt import ReasoningDAToolPromptTemplate

from .reasoning import ReasoningTask


class ReasoningDATask(ReasoningTask):
    task_name = "reasoning_da"
    in_context_example_files = {
        "with_tool": "with_py.txt",
        "with_tool_and_feedback": "with_tool_and_feedback.txt",
    }
    ENV_INFO_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data_env_prompt",
        "prefix_map.json",
    )
    _env_info_cache = None

    @classmethod
    def stringify_env_info(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value)

    @classmethod
    def env_info_for_version(cls, env_info_version: str) -> str:
        if cls._env_info_cache is None:
            with open(cls.ENV_INFO_PATH, "r") as f:
                cls._env_info_cache = json.load(f)
        value = cls._env_info_cache.get(env_info_version, "")
        prompt = cls.stringify_env_info(value)
        if prompt and not prompt.endswith("\n"):
            prompt += "\n"
        return prompt

    def __init__(self, id: str, prompt: str, reference: str, **kwargs):
        super().__init__(id=id, prompt=prompt, reference=reference, **kwargs)
        env_info_version = kwargs.get("env_info_version") or kwargs.get("data_env")
        self._env_info = (
            self.env_info_for_version(env_info_version) if env_info_version else ""
        )
        if env_info_version:
            self.metadata["env_info_version"] = env_info_version

    @property
    def env_info(self) -> str:
        return getattr(self, "_env_info", "")

    @property
    def env_state_prompt(self) -> str:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_path = os.path.join(os.path.abspath(os.getcwd()), "workspace/work_dataset")
        return (
            "Environment state:\n"
            f"- <current_time>: {current_time}\n"
            f"- <system_path>: {system_path}"
        )

    def prompt_template(self, use_tool: bool = True):
        if use_tool:
            return ReasoningDAToolPromptTemplate()
        return super().prompt_template(use_tool=use_tool)


class MiningBenchTask(ReasoningDATask):
    """MiningBench rule-mining tasks evaluated by XML answer fields."""

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _normalize_features(value: str) -> str:
        return ",".join(part.strip().lower() for part in value.split(",") if part.strip())

    @staticmethod
    def _normalize_rule(value: str) -> str:
        return re.sub(r"\s+", "", value.strip().lower())

    def extract_answer(self, solution: str) -> str:
        features = self._extract_tag(solution, "answer_features")
        rule = self._extract_tag(solution, "answer_rule")
        return (
            f"<answer_features>{features}</answer_features>"
            f"<answer_rule>{rule}</answer_rule>"
        )

    def success(self, solution: str) -> bool:
        answer = self.extract_answer(solution)
        return (
            self._normalize_features(self._extract_tag(answer, "answer_features"))
            == self._normalize_features(self._extract_tag(self.reference, "answer_features"))
            and self._normalize_rule(self._extract_tag(answer, "answer_rule"))
            == self._normalize_rule(self._extract_tag(self.reference, "answer_rule"))
        )
