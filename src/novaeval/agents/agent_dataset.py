import csv
import json
import typing
from collections.abc import Iterator
from typing import Any, Optional

from novaeval.agents.agent_data import AgentData


class AgentDataset:
    def __init__(self) -> None:
        self.data: list[AgentData] = []
        # Dynamically determine field types from AgentData model
        self._list_fields = set()
        self._dict_fields = set()
        for field_name, field_info in AgentData.model_fields.items():
            if hasattr(field_info, "annotation"):
                annotation = field_info.annotation
                # Unwrap typing.Optional, typing.Union, etc.
                origin = getattr(annotation, "__origin__", None)
                args = getattr(annotation, "__args__", ())
                # If it's a Union (e.g., Optional), check all args except NoneType
                if origin is typing.Union:
                    for arg in args:
                        if arg is type(None):
                            continue
                        arg_origin = getattr(arg, "__origin__", None)
                        if arg_origin in (list, dict):
                            if arg_origin is list:
                                self._list_fields.add(field_name)
                            elif arg_origin is dict:
                                self._dict_fields.add(field_name)
                        elif arg in (list, dict):
                            if arg is list:
                                self._list_fields.add(field_name)
                            elif arg is dict:
                                self._dict_fields.add(field_name)
                # Handle direct types
                elif origin in (list, dict):
                    if origin is list:
                        self._list_fields.add(field_name)
                    elif origin is dict:
                        self._dict_fields.add(field_name)
                elif annotation in (list, dict):
                    if annotation is list:
                        self._list_fields.add(field_name)
                    elif annotation is dict:
                        self._dict_fields.add(field_name)

    def _parse_field(self, field: str, value: Any) -> Any:
        if field in self._list_fields:
            if isinstance(value, str):
                value = value.strip()
                if value.startswith("[") and value.endswith("]"):
                    try:
                        return json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        return []
            return value if isinstance(value, list) else []
        if field in self._dict_fields:
            if isinstance(value, str):
                value = value.strip()
                if value.startswith("{") and value.endswith("}"):
                    try:
                        return json.loads(value)
                    except (json.JSONDecodeError, TypeError):
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
        try:
            with open(file_path, encoding="utf-8") as f:
                try:
                    items = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in file '{file_path}': {e}")
        except FileNotFoundError:
            raise ValueError(f"File not found: '{file_path}'")
        except PermissionError:
            raise ValueError(f"Permission denied when reading file: '{file_path}'")
        if not isinstance(items, list):
            raise ValueError(
                f"JSON file '{file_path}' must contain an array of objects at the top level."
            )
        for item in items:
            if not isinstance(item, dict):
                continue  # Skip non-dict items
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
