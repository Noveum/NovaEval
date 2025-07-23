import csv
import json
from collections.abc import Iterator
from typing import Any, Optional

from novaeval.agents.agent_data import AgentData


class AgentDataset:
    def __init__(self) -> None:
        self.data: list[AgentData] = []

    def _parse_field(self, field: str, value: Any) -> Any:
        list_fields = {"trace", "tools_available", "tool_calls", "tool_call_results"}
        dict_fields = {"parameters_passed"}
        if field in list_fields:
            if isinstance(value, str):
                value = value.strip()
                if value.startswith("[") and value.endswith("]"):
                    try:
                        return json.loads(value)
                    except Exception:
                        return []
            return value if isinstance(value, list) else []
        if field in dict_fields:
            if isinstance(value, str):
                value = value.strip()
                if value.startswith("{") and value.endswith("}"):
                    try:
                        return json.loads(value)
                    except Exception:
                        return {}
            return value if isinstance(value, dict) else {}
        return value

    def ingest_from_csv(
        self,
        file_path: str,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        agent_task: Optional[str] = None,
        system_prompt: Optional[str] = None,
        agent_response: Optional[str] = None,
        trace: Optional[str] = None,
        tools_available: Optional[str] = None,
        tool_calls: Optional[str] = None,
        parameters_passed: Optional[str] = None,
        tool_call_results: Optional[str] = None,
        retrieval_query: Optional[str] = None,
        retrieved_context: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> None:
        field_map = {
            "agent_name": agent_name,
            "agent_role": agent_role,
            "agent_task": agent_task,
            "system_prompt": system_prompt,
            "agent_response": agent_response,
            "trace": trace,
            "tools_available": tools_available,
            "tool_calls": tool_calls,
            "parameters_passed": parameters_passed,
            "tool_call_results": tool_call_results,
            "retrieval_query": retrieval_query,
            "retrieved_context": retrieved_context,
            "metadata": metadata,
        }
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_kwargs = {}
                for field in AgentData.model_fields:
                    col = field_map[field] if field_map[field] is not None else field
                    value = row.get(col, None)
                    data_kwargs[field] = self._parse_field(field, value)
                self.data.append(AgentData(**data_kwargs))

    def ingest_from_json(
        self,
        file_path: str,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        agent_task: Optional[str] = None,
        system_prompt: Optional[str] = None,
        agent_response: Optional[str] = None,
        trace: Optional[str] = None,
        tools_available: Optional[str] = None,
        tool_calls: Optional[str] = None,
        parameters_passed: Optional[str] = None,
        tool_call_results: Optional[str] = None,
        retrieval_query: Optional[str] = None,
        retrieved_context: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> None:
        field_map = {
            "agent_name": agent_name,
            "agent_role": agent_role,
            "agent_task": agent_task,
            "system_prompt": system_prompt,
            "agent_response": agent_response,
            "trace": trace,
            "tools_available": tools_available,
            "tool_calls": tool_calls,
            "parameters_passed": parameters_passed,
            "tool_call_results": tool_call_results,
            "retrieval_query": retrieval_query,
            "retrieved_context": retrieved_context,
            "metadata": metadata,
        }
        with open(file_path, encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                data_kwargs = {}
                for field in AgentData.model_fields:
                    key = field_map[field] if field_map[field] is not None else field
                    value = item.get(key, None)
                    data_kwargs[field] = self._parse_field(field, value)
                self.data.append(AgentData(**data_kwargs))

    def export_to_csv(self, file_path: str) -> None:
        if not self.data:
            return
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(AgentData.model_fields.keys()))
            writer.writeheader()
            for agent in self.data:
                row = agent.model_dump()
                for k, v in row.items():
                    if isinstance(v, (list, dict)) and v is not None:
                        row[k] = json.dumps(v)
                writer.writerow(row)

    def export_to_json(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                [agent.model_dump() for agent in self.data],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def get_data(self) -> list[AgentData]:
        return self.data

    def get_datapoint(self) -> Iterator[AgentData]:
        yield from self.data
