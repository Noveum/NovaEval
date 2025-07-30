import csv
import json
import typing
from collections.abc import Iterator
from typing import Any, Optional

import pandas as pd

from novaeval.agents.agent_data import AgentData, ToolCall


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
        if field not in AgentData.model_fields:
            raise KeyError(f"Field '{field}' not found in AgentData.model_fields.")
        # Get the annotation for the field
        annotation = AgentData.model_fields[field].annotation
        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())
        # Handle Optional[X]
        if origin is typing.Union and type(None) in args:
            non_none_args = [arg for arg in args if arg is not type(None)]
            actual_type = non_none_args[0] if non_none_args else annotation
        else:
            actual_type = annotation
        # Handle list fields
        if field in self._list_fields:
            if isinstance(value, str):
                value = value.strip()
                if value.startswith("[") and value.endswith("]"):
                    try:
                        return json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        return []
            return value if isinstance(value, list) else []
        # Handle dict fields
        if field in self._dict_fields:
            if isinstance(value, str):
                value = value.strip()
                if value.startswith("{") and value.endswith("}"):
                    try:
                        return json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        return {}
            return value if isinstance(value, dict) else {}
        # Handle ToolCall fields
        if actual_type is ToolCall:
            if value is None or value == "":
                return None
            if isinstance(value, ToolCall):
                return value
            if isinstance(value, str):
                value = value.strip()
                try:
                    value_dict = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return None
            elif isinstance(value, dict):
                value_dict = value
            else:
                return None
            try:
                return ToolCall(**value_dict)
            except Exception:
                return None
        # Handle str fields (including turn_id)
        if actual_type is str:
            if value is None:
                return None
            return str(value)
        # Handle bool fields (including agent_exit)
        if actual_type is bool:
            if value is None:
                # Return default value for boolean fields
                field_info = AgentData.model_fields[field]
                if hasattr(field_info, "default") and field_info.default is not None:
                    return field_info.default
                return False  # Fallback default for bool
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                value = value.strip().lower()
                if value in ("true", "1", "yes", "on"):
                    return True
                elif value in ("false", "0", "no", "off"):
                    return False
                else:
                    return False  # Default to False for unrecognized strings
            # Convert numbers to bool (0 = False, anything else = True)
            try:
                return bool(value)
            except (ValueError, TypeError):
                return False
        # Default: return as is
        return value

    def ingest_from_csv(
        self,
        file_path: str,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        expected_tool_call: Optional[str] = None,
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
        exit_status: Optional[str] = None,
        agent_exit: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> None:
        field_map = {
            "user_id": user_id,
            "task_id": task_id,
            "turn_id": turn_id,
            "ground_truth": ground_truth,
            "expected_tool_call": expected_tool_call,
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
            "exit_status": exit_status,
            "agent_exit": agent_exit,
            "metadata": metadata,
        }

        try:
            import pandas as pd

            # Process CSV in chunks to save memory
            chunk_size = 1000  # Adjust based on memory constraints

            for chunk_df in pd.read_csv(
                file_path, encoding="utf-8", chunksize=chunk_size
            ):
                for _, row in chunk_df.iterrows():
                    data_kwargs = {}
                    for field in AgentData.model_fields:
                        col = (
                            field_map[field] if field_map[field] is not None else field
                        )
                        value = row.get(col, None)
                        data_kwargs[field] = self._parse_field(field, value)
                    self.data.append(AgentData(**data_kwargs))

                # Clear chunk from memory
                del chunk_df

        except Exception as e:
            raise ValueError(f"Error reading CSV file '{file_path}': {e!s}")

    def ingest_from_json(
        self,
        file_path: str,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        expected_tool_call: Optional[str] = None,
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
        exit_status: Optional[str] = None,
        agent_exit: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> None:
        field_map = {
            "user_id": user_id,
            "task_id": task_id,
            "turn_id": turn_id,
            "ground_truth": ground_truth,
            "expected_tool_call": expected_tool_call,
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
            "exit_status": exit_status,
            "agent_exit": agent_exit,
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

    def stream_from_csv(
        self,
        file_path: str,
        chunk_size: int = 1000,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        expected_tool_call: Optional[str] = None,
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
        exit_status: Optional[str] = None,
        agent_exit: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> Iterator[list[AgentData]]:
        """
        Stream data from CSV in chunks without loading entire file.
        Returns an iterator of lists, where each list contains chunk_size AgentData objects.

        Args:
            file_path: Path to the CSV file
            chunk_size: Number of rows to process at a time
            user_id, task_id, etc.: Column name mappings for AgentData fields

        Returns:
            Iterator yielding lists of AgentData objects, each list of size <= chunk_size
        """
        field_map = {
            "user_id": user_id,
            "task_id": task_id,
            "turn_id": turn_id,
            "ground_truth": ground_truth,
            "expected_tool_call": expected_tool_call,
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
            "exit_status": exit_status,
            "agent_exit": agent_exit,
            "metadata": metadata,
        }

        try:
            # Process CSV in chunks
            for chunk_df in pd.read_csv(
                file_path, encoding="utf-8", chunksize=chunk_size
            ):
                chunk_data = []

                for _, row in chunk_df.iterrows():
                    data_kwargs = {}
                    for field in AgentData.model_fields:
                        col = (
                            field_map[field] if field_map[field] is not None else field
                        )
                        value = row.get(col, None)
                        data_kwargs[field] = self._parse_field(field, value)
                    chunk_data.append(AgentData(**data_kwargs))

                yield chunk_data

                # Clear memory
                del chunk_data

        except Exception as e:
            raise ValueError(f"Error reading CSV file '{file_path}': {e!s}")

    def stream_from_json(
        self,
        file_path: str,
        chunk_size: int = 1000,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        expected_tool_call: Optional[str] = None,
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
        exit_status: Optional[str] = None,
        agent_exit: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> Iterator[list[AgentData]]:
        """
        Stream data from JSON in chunks without loading entire file.
        Returns an iterator of lists, where each list contains chunk_size AgentData objects.

        Args:
            file_path: Path to the JSON file
            chunk_size: Number of items to process at a time
            user_id, task_id, etc.: Field name mappings for AgentData fields

        Returns:
            Iterator yielding lists of AgentData objects, each list of size <= chunk_size

        Note:
            Expects JSON file to contain an array of objects at the root level
        """
        field_map = {
            "user_id": user_id,
            "task_id": task_id,
            "turn_id": turn_id,
            "ground_truth": ground_truth,
            "expected_tool_call": expected_tool_call,
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
            "exit_status": exit_status,
            "agent_exit": agent_exit,
            "metadata": metadata,
        }

        try:
            import ijson  # Import here to not require it unless method is used

            chunk_data = []
            with open(file_path, "rb") as file:
                # Parse JSON array items one at a time
                parser = ijson.items(file, "item")

                for item in parser:
                    if not isinstance(item, dict):
                        continue

                    data_kwargs = {}
                    for field in AgentData.model_fields:
                        key = (
                            field_map[field] if field_map[field] is not None else field
                        )
                        value = item.get(key, None)
                        data_kwargs[field] = self._parse_field(field, value)

                    chunk_data.append(AgentData(**data_kwargs))

                    # When chunk is full, yield it
                    if len(chunk_data) >= chunk_size:
                        yield chunk_data
                        chunk_data = []

                # Yield remaining data if any
                if chunk_data:
                    yield chunk_data

        except ImportError:
            raise ImportError(
                "ijson package is required for streaming JSON. Install with: pip install ijson"
            )
        except Exception as e:
            raise ValueError(f"Error reading JSON file '{file_path}': {e!s}")

    def get_data(self) -> list[AgentData]:
        return self.data.copy()

    def get_datapoint(self) -> Iterator[AgentData]:
        yield from self.data
