import csv
import json
import os
import tempfile
from unittest.mock import patch

import pytest

from novaeval.agents.agent_data import AgentData
from novaeval.datasets.agent_dataset import AgentDataset, ToolCall


def minimal_agent_data_dict():
    return {
        "user_id": "user42",
        "task_id": "task99",
        "turn_id": "turn7",
        "ground_truth": "expected answer",
        "expected_tool_call": {
            "tool_name": "tool1",
            "parameters": {},
            "call_id": "abc",
        },
        "agent_name": "TestAgent",
        "agent_role": "assistant",
        "agent_task": "answer",
        "system_prompt": "You are helpful.",
        "agent_response": "Hello!",
        "trace": [{"step": "trace info"}],
        "tools_available": [
            {
                "name": "tool1",
                "description": "desc",
                "args_schema": {},
                "return_schema": {},
            }
        ],
        "tool_calls": [{"tool_name": "tool1", "parameters": {}, "call_id": "abc"}],
        "parameters_passed": {"param": "value"},
        "tool_call_results": [
            {
                "call_id": "abc",
                "result": "result1",
                "success": True,
                "error_message": None,
            }
        ],
        "retrieval_query": ["query1"],
        "retrieved_context": [["context1"]],
        "exit_status": "completed",
        "agent_exit": True,
        "metadata": "meta1",
    }


def minimal_agent_data_csv_row():
    d = minimal_agent_data_dict()
    return {
        **{
            k: v
            for k, v in d.items()
            if k
            not in [
                "trace",
                "tools_available",
                "tool_calls",
                "parameters_passed",
                "tool_call_results",
                "expected_tool_call",
                "retrieval_query",
                "retrieved_context",
            ]
        },
        "trace": json.dumps(d["trace"]),
        "tools_available": json.dumps(d["tools_available"]),
        "tool_calls": json.dumps(d["tool_calls"]),
        "parameters_passed": json.dumps(d["parameters_passed"]),
        "tool_call_results": json.dumps(d["tool_call_results"]),
        "expected_tool_call": json.dumps(d["expected_tool_call"]),
        "retrieval_query": json.dumps(d["retrieval_query"]),
        "retrieved_context": json.dumps(d["retrieved_context"]),
    }


def assert_agentdata_equal(actual, expected):
    for k, v in expected.items():
        actual_val = getattr(actual, k)
        if k == "expected_tool_call" and actual_val is not None:
            assert actual_val.model_dump() == v
        elif (
            isinstance(actual_val, list)
            and actual_val
            and hasattr(actual_val[0], "model_dump")
        ):
            assert [x.model_dump() for x in actual_val] == v
        elif hasattr(actual_val, "model_dump"):
            assert actual_val.model_dump() == v
        else:
            assert actual_val == v


def assert_missing_fields_defaults(agent):
    for k in AgentData.model_fields:
        if k not in [
            "agent_name",
            "agent_role",
            "tools_available",
            "tool_calls",
            "parameters_passed",
            "agent_exit",  # Boolean field with default False
        ]:
            val = getattr(agent, k)
            if isinstance(val, list) and val and hasattr(val[0], "model_dump"):
                assert [x.model_dump() for x in val] == []
            elif hasattr(val, "model_dump"):
                assert val.model_dump() == {}
            else:
                assert val is None or val == [] or val == {}

    # Check agent_exit specifically has its default value
    assert agent.agent_exit is False


@pytest.mark.unit
def test_ingest_from_csv_and_export(tmp_path):
    data = minimal_agent_data_csv_row()
    csv_file = tmp_path / "test.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
    ds = AgentDataset()
    ds.ingest_from_csv(str(csv_file))
    assert len(ds.data) == 1
    expected = minimal_agent_data_dict()
    assert_agentdata_equal(ds.data[0], expected)
    # Export to CSV
    export_file = tmp_path / "export.csv"
    ds.export_to_csv(str(export_file))
    with open(export_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        for k, v in minimal_agent_data_csv_row().items():
            if k in [
                "trace",
                "tools_available",
                "tool_calls",
                "parameters_passed",
                "tool_call_results",
            ]:
                assert json.loads(rows[0][k]) == json.loads(v)
            elif k == "agent_exit":
                # Boolean fields are converted to strings in CSV
                assert rows[0][k] == str(v)
            else:
                assert rows[0][k] == v


@pytest.mark.unit
def test_ingest_from_json_and_export(tmp_path):
    data = minimal_agent_data_dict()
    json_file = tmp_path / "test.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump([data], f)
    ds = AgentDataset()
    ds.ingest_from_json(str(json_file))
    assert len(ds.data) == 1
    assert_agentdata_equal(ds.data[0], data)
    # Export to JSON
    export_file = tmp_path / "export.json"
    ds.export_to_json(str(export_file))
    with open(export_file, encoding="utf-8") as f:
        items = json.load(f)
        assert len(items) == 1
        for k, v in data.items():
            assert items[0][k] == v


@pytest.mark.unit
def test_ingest_from_csv_with_field_map(tmp_path):
    data = minimal_agent_data_csv_row()
    csv_file = tmp_path / "test_map.csv"
    custom_cols = {k: f"col_{k}" for k in data}
    row = {f"col_{k}": v for k, v in data.items()}
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    ds = AgentDataset()
    ds.ingest_from_csv(str(csv_file), **custom_cols)
    assert len(ds.data) == 1
    expected = minimal_agent_data_dict()
    assert_agentdata_equal(ds.data[0], expected)


@pytest.mark.unit
def test_ingest_from_json_with_field_map(tmp_path):
    data = minimal_agent_data_dict()
    custom_cols = {k: f"col_{k}" for k in data}
    row = {f"col_{k}": v for k, v in data.items()}
    json_file = tmp_path / "test_map.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump([row], f)
    ds = AgentDataset()
    ds.ingest_from_json(str(json_file), **custom_cols)
    assert len(ds.data) == 1
    expected = minimal_agent_data_dict()
    assert_agentdata_equal(ds.data[0], expected)


@pytest.mark.unit
def test_export_to_csv_empty(tmp_path):
    ds = AgentDataset()
    export_file = tmp_path / "empty.csv"
    ds.export_to_csv(str(export_file))  # Should not raise
    # For empty dataset, should create file with just header
    assert export_file.exists()
    content = export_file.read_text()
    assert (
        content.strip()
        == "user_id,task_id,turn_id,ground_truth,expected_tool_call,agent_name,agent_role,agent_task,system_prompt,agent_response,trace,tools_available,tool_calls,parameters_passed,tool_call_results,retrieval_query,retrieved_context,exit_status,agent_exit,metadata"
    )


@pytest.mark.unit
def test_export_to_json_empty(tmp_path):
    ds = AgentDataset()
    export_file = tmp_path / "empty.json"
    ds.export_to_json(str(export_file))
    with open(export_file, encoding="utf-8") as f:
        data = json.load(f)
        assert data == []


@pytest.mark.unit
def test_get_data_and_get_datapoint():
    ds = AgentDataset()
    data = minimal_agent_data_dict()
    agent = AgentData(**data)
    ds.data.append(agent)
    assert ds.get_data() == [agent]
    assert list(ds.get_datapoint()) == [agent]


@pytest.mark.unit
def test_ingest_from_csv_missing_fields(tmp_path):
    # Only some fields present, but use correct types for those present
    data = {
        "agent_name": "A",
        "agent_role": "B",
        "tools_available": json.dumps(
            [
                {
                    "name": "tool1",
                    "description": "desc",
                    "args_schema": {},
                    "return_schema": {},
                }
            ]
        ),
        "tool_calls": json.dumps(
            [{"tool_name": "tool1", "parameters": {}, "call_id": "abc"}]
        ),
        "parameters_passed": json.dumps({"param": "value"}),
    }
    csv_file = tmp_path / "missing.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
    ds = AgentDataset()
    ds.ingest_from_csv(str(csv_file))
    assert len(ds.data) == 1
    assert ds.data[0].agent_name == "A"
    assert ds.data[0].agent_role == "B"
    assert_missing_fields_defaults(ds.data[0])


@pytest.mark.unit
def test_ingest_from_json_missing_fields(tmp_path):
    data = {
        "agent_name": "A",
        "agent_role": "B",
        "tools_available": [
            {
                "name": "tool1",
                "description": "desc",
                "args_schema": {},
                "return_schema": {},
            }
        ],
        "tool_calls": [{"tool_name": "tool1", "parameters": {}, "call_id": "abc"}],
        "parameters_passed": {"param": "value"},
    }
    json_file = tmp_path / "missing.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump([data], f)
    ds = AgentDataset()
    ds.ingest_from_json(str(json_file))
    assert len(ds.data) == 1
    assert ds.data[0].agent_name == "A"
    assert ds.data[0].agent_role == "B"
    assert_missing_fields_defaults(ds.data[0])


@pytest.mark.unit
def test_parse_field_list_and_dict_edge_cases():
    ds = AgentDataset()
    # List fields: valid JSON string
    assert ds._parse_field("trace", '[{"a":1}]') == [{"a": 1}]
    # List fields: invalid JSON string (kept as string for union fields)
    assert ds._parse_field("trace", "[invalid]") == "[invalid]"
    # List fields: already a list
    assert ds._parse_field("trace", [{"a": 2}]) == [{"a": 2}]
    # List fields: wrong type (non-string values return empty list)
    assert ds._parse_field("trace", 123) == []
    # Dict fields: valid JSON string
    assert ds._parse_field("parameters_passed", '{"x":1}') == {"x": 1}
    # Dict fields: invalid JSON string (kept as string for union fields)
    assert ds._parse_field("parameters_passed", "{invalid}") == "{invalid}"
    # Dict fields: already a dict
    assert ds._parse_field("parameters_passed", {"y": 2}) == {"y": 2}
    # Dict fields: wrong type (non-string values return empty dict)
    assert ds._parse_field("parameters_passed", 123) == {}
    # Non-list/dict field
    assert ds._parse_field("agent_name", "abc") == "abc"


@pytest.mark.unit
def test_parse_field_list_string_no_brackets():
    ds = AgentDataset()
    # For union fields with string, non-JSON strings should be kept as strings
    assert ds._parse_field("trace", "notalist") == "notalist"


@pytest.mark.unit
def test_parse_field_dict_string_no_braces():
    ds = AgentDataset()
    # For union fields with string, non-JSON strings should be kept as strings
    assert ds._parse_field("parameters_passed", "notadict") == "notadict"


@pytest.mark.unit
def test_parse_field_list_field_non_list_non_str():
    ds = AgentDataset()
    # Should return [] for list field if value is not a list or str
    assert ds._parse_field("trace", 42) == []


@pytest.mark.unit
def test_parse_field_dict_field_non_dict_non_str():
    ds = AgentDataset()
    # Should return {} for dict field if value is not a dict or str
    assert ds._parse_field("parameters_passed", 42) == {}


@pytest.mark.unit
def test_ingest_from_csv_invalid_json(tmp_path):
    # List/dict fields with invalid JSON (now kept as strings for union fields)
    data = {
        "agent_name": "A",
        "trace": "[invalid]",
        "tools_available": "[invalid]",
        "tool_calls": "[invalid]",
        "parameters_passed": "{invalid}",
        "tool_call_results": "[invalid]",
    }
    csv_file = tmp_path / "invalid.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
    ds = AgentDataset()
    ds.ingest_from_csv(str(csv_file))
    agent = ds.data[0]
    # Invalid JSON strings are kept as strings for union fields
    assert agent.trace == "[invalid]"
    assert agent.tools_available == "[invalid]"
    assert agent.tool_calls == "[invalid]"
    assert agent.parameters_passed == "{invalid}"
    assert agent.tool_call_results == "[invalid]"


@pytest.mark.unit
def test_ingest_from_json_invalid_types(tmp_path):
    # List/dict fields with wrong types
    data = {
        "agent_name": "A",
        "trace": 123,
        "tools_available": 123,
        "tool_calls": 123,
        "parameters_passed": 123,
        "tool_call_results": 123,
    }
    json_file = tmp_path / "invalid_types.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump([data], f)
    ds = AgentDataset()
    ds.ingest_from_json(str(json_file))
    agent = ds.data[0]
    assert agent.trace == []
    assert agent.tools_available == []
    assert agent.tool_calls == []
    assert agent.parameters_passed == {}
    assert agent.tool_call_results == []


@pytest.mark.unit
def test_export_to_csv_non_serializable(tmp_path):
    ds = AgentDataset()
    # Insert an agent with a non-serializable field (should not fail, just skip serialization)
    agent = AgentData(
        agent_name="A", tools_available=[], tool_calls=[], parameters_passed={}
    )
    ds.data.append(agent)
    export_file = tmp_path / "nons.json"
    ds.export_to_csv(str(export_file))
    # File should exist and be readable
    assert export_file.exists()


@pytest.mark.unit
def test_get_datapoint_empty():
    ds = AgentDataset()
    assert list(ds.get_datapoint()) == []


@pytest.mark.unit
def test_agentdataset_field_type_detection():
    ds = AgentDataset()
    # These are the actual list/dict fields in AgentData
    expected_list_fields = {
        "trace",
        "tools_available",
        "tool_calls",
        "tool_call_results",
        "retrieval_query",
        "retrieved_context",
    }
    expected_dict_fields = {"parameters_passed"}
    assert ds._list_fields == expected_list_fields
    assert ds._dict_fields == expected_dict_fields


@pytest.mark.unit
def test_agentdataset_init_type_detection_edge_cases(monkeypatch):
    import typing
    from types import SimpleNamespace

    # Save original model_fields
    import novaeval.agents.agent_data as agent_data_mod

    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()

    # Create mock fields
    class Dummy:
        pass

    # Not Optional, not list/dict
    model_fields = {
        "plain": SimpleNamespace(annotation=int),
        # Optional but not list/dict
        "opt_str": SimpleNamespace(annotation=typing.Optional[str]),
        # Optional Union with first arg not list/dict
        "opt_union": SimpleNamespace(annotation=typing.Optional[int]),
        # Optional Union with first arg list (should be detected)
        "opt_list": SimpleNamespace(annotation=typing.Optional[list]),
        # Optional Union with first arg dict (should be detected)
        "opt_dict": SimpleNamespace(annotation=typing.Optional[dict]),
    }
    monkeypatch.setattr(agent_data_mod.AgentData, "model_fields", model_fields)
    ds = AgentDataset()
    # Only opt_list and opt_dict should be detected
    assert ds._list_fields == {"opt_list"}
    assert ds._dict_fields == {"opt_dict"}
    # Restore
    monkeypatch.setattr(agent_data_mod.AgentData, "model_fields", orig_model_fields)


@pytest.mark.unit
def test_agentdataset_init_skips_fields_without_annotation(monkeypatch):
    # Should not raise or add to _list_fields/_dict_fields
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()
    model_fields = {
        "plain": SimpleNamespace(),  # No annotation attribute
    }
    monkeypatch.setattr(agent_data_mod.AgentData, "model_fields", model_fields)
    ds = AgentDataset()
    assert ds._list_fields == set()
    assert ds._dict_fields == set()
    monkeypatch.setattr(agent_data_mod.AgentData, "model_fields", orig_model_fields)


@pytest.mark.unit
def test_parse_field_returns_value_for_non_listdict_field():
    ds = AgentDataset()
    # Add a dummy field not in _list_fields or _dict_fields
    with pytest.raises(KeyError, match="not_special"):
        ds._parse_field("not_special", 123)
    with pytest.raises(KeyError, match="not_special"):
        ds._parse_field("not_special", "abc")


@pytest.mark.unit
def test_parse_field_list_field_non_str_non_list():
    ds = AgentDataset()
    # Should return [] for list field if value is not a list or str
    assert ds._parse_field("trace", 42) == []


@pytest.mark.unit
def test_parse_field_dict_field_non_str_non_dict():
    ds = AgentDataset()
    # Should return {} for dict field if value is not a dict or str
    assert ds._parse_field("parameters_passed", 42) == {}


@pytest.mark.unit
def test_export_to_csv_empty_file(tmp_path):
    ds = AgentDataset()
    file_path = tmp_path / "empty.csv"
    ds.export_to_csv(str(file_path))
    # For empty dataset, should create file with just header
    assert file_path.exists()
    content = file_path.read_text()
    assert (
        content.strip()
        == "user_id,task_id,turn_id,ground_truth,expected_tool_call,agent_name,agent_role,agent_task,system_prompt,agent_response,trace,tools_available,tool_calls,parameters_passed,tool_call_results,retrieval_query,retrieved_context,exit_status,agent_exit,metadata"
    )


@pytest.mark.unit
def test_parse_field_invalid_json_list():
    ds = AgentDataset()
    ds._list_fields.add("trace")
    # Invalid JSON string for list (kept as string for union fields)
    assert ds._parse_field("trace", "[notjson]") == "[notjson]"


@pytest.mark.unit
def test_parse_field_invalid_json_dict():
    ds = AgentDataset()
    ds._dict_fields.add("parameters_passed")
    # Invalid JSON string for dict (kept as string for union fields)
    assert ds._parse_field("parameters_passed", "{notjson}") == "{notjson}"


@pytest.mark.unit
def test_agentdataset_init_direct_list_dict_types(monkeypatch):
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()
    # Direct built-in list
    model_fields = {
        "list_field": SimpleNamespace(annotation=list),
        "dict_field": SimpleNamespace(annotation=dict),
        "typing_list_field": SimpleNamespace(annotation=list),
        "typing_dict_field": SimpleNamespace(annotation=dict),
    }
    monkeypatch.setattr(agent_data_mod.AgentData, "model_fields", model_fields)
    ds = AgentDataset()
    assert "list_field" in ds._list_fields
    assert "dict_field" in ds._dict_fields
    assert "typing_list_field" in ds._list_fields
    assert "typing_dict_field" in ds._dict_fields
    monkeypatch.setattr(agent_data_mod.AgentData, "model_fields", orig_model_fields)


@pytest.mark.unit
def test_ingest_from_json_file_not_found():
    ds = AgentDataset()
    with pytest.raises(ValueError) as exc:
        ds.ingest_from_json("/nonexistent/file/path.json")
    assert "File not found" in str(exc.value)


@pytest.mark.unit
def test_ingest_from_json_permission_denied(monkeypatch, tmp_path):
    # Simulate PermissionError by monkeypatching open
    json_file = tmp_path / "perm.json"
    json_file.write_text("[]", encoding="utf-8")

    def raise_permission(*args, **kwargs):
        raise PermissionError

    monkeypatch.setattr("builtins.open", raise_permission)
    ds = AgentDataset()
    with pytest.raises(ValueError) as exc:
        ds.ingest_from_json(str(json_file))
    assert "Permission denied" in str(exc.value)


@pytest.mark.unit
def test_ingest_from_json_invalid_json(tmp_path):
    json_file = tmp_path / "invalid.json"
    json_file.write_text("{not valid json}", encoding="utf-8")
    ds = AgentDataset()
    with pytest.raises(ValueError) as exc:
        ds.ingest_from_json(str(json_file))
    assert "Invalid JSON" in str(exc.value)


@pytest.mark.unit
def test_ingest_from_json_not_a_list(tmp_path):
    json_file = tmp_path / "notalist.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"foo": "bar"}, f)
    ds = AgentDataset()
    with pytest.raises(ValueError) as exc:
        ds.ingest_from_json(str(json_file))
    assert "must contain an array of objects" in str(exc.value)


@pytest.mark.unit
def test_ingest_from_json_skips_non_dict_items(tmp_path):
    # Only dicts should be ingested
    items = [
        {"agent_name": "A", "agent_role": "B"},
        [1, 2, 3],
        "string",
        123,
        {"agent_name": "C", "agent_role": "D"},
    ]
    json_file = tmp_path / "mixed.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(items, f)
    ds = AgentDataset()
    ds.ingest_from_json(str(json_file))
    # Only the two dicts should be ingested
    assert len(ds.data) == 2
    assert ds.data[0].agent_name == "A"
    assert ds.data[1].agent_name == "C"


@pytest.mark.unit
def test_parse_field_raises_keyerror_for_unknown_field():
    ds = AgentDataset()
    with pytest.raises(KeyError, match="not_special"):
        ds._parse_field("not_special", 123)


@pytest.mark.unit
def test_parse_field_expected_tool_call_dict_and_json():
    ds = AgentDataset()
    # As dict (for union fields, non-string values are kept as-is and handled by Pydantic)
    val = ds._parse_field(
        "expected_tool_call", {"tool_name": "t", "parameters": {}, "call_id": "c"}
    )
    assert val == {"tool_name": "t", "parameters": {}, "call_id": "c"}
    # As JSON string (gets parsed to ToolCall)
    val2 = ds._parse_field(
        "expected_tool_call", '{"tool_name": "t2", "parameters": {}, "call_id": "c2"}'
    )
    assert val2.tool_name == "t2"


@pytest.mark.unit
def test_parse_field_turn_id_as_string():
    ds = AgentDataset()
    assert ds._parse_field("turn_id", "abc") == "abc"
    assert ds._parse_field("turn_id", None) is None
    assert ds._parse_field("turn_id", "") == ""


@pytest.mark.unit
def test_parse_field_list_and_dict_various_json():
    ds = AgentDataset()
    # List field
    assert ds._parse_field("trace", "[]") == []
    assert (
        ds._parse_field("trace", "[invalid]") == "[invalid]"
    )  # kept as string for union fields
    assert ds._parse_field("trace", '[{"a":1}]') == [{"a": 1}]
    # Dict field
    assert ds._parse_field("parameters_passed", "{}") == {}
    assert (
        ds._parse_field("parameters_passed", "{invalid}") == "{invalid}"
    )  # kept as string for union fields
    assert ds._parse_field("parameters_passed", '{"x":1}') == {"x": 1}


@pytest.mark.unit
def test_ingest_from_csv_with_extra_columns(tmp_path):
    data = minimal_agent_data_csv_row()
    data["extra_col"] = "extra_val"
    csv_file = tmp_path / "extra.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
    ds = AgentDataset()
    ds.ingest_from_csv(str(csv_file))
    assert len(ds.data) == 1
    expected = minimal_agent_data_dict()
    assert_agentdata_equal(ds.data[0], expected)


@pytest.mark.unit
def test_ingest_from_json_with_extra_keys(tmp_path):
    data = minimal_agent_data_dict()
    data["extra_key"] = "extra_val"
    json_file = tmp_path / "extra.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump([data], f)
    ds = AgentDataset()
    ds.ingest_from_json(str(json_file))
    assert len(ds.data) == 1
    expected = minimal_agent_data_dict()
    assert_agentdata_equal(ds.data[0], expected)


@pytest.mark.unit
def test_export_to_csv_and_json_with_new_fields(tmp_path):
    ds = AgentDataset()
    agent = AgentData(**minimal_agent_data_dict())
    ds.data.append(agent)
    # Export to CSV
    export_file = tmp_path / "export_new.csv"
    ds.export_to_csv(str(export_file))
    with open(export_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        for k, v in minimal_agent_data_csv_row().items():
            if k in [
                "trace",
                "tools_available",
                "tool_calls",
                "parameters_passed",
                "tool_call_results",
                "expected_tool_call",
            ]:
                assert json.loads(rows[0][k]) == json.loads(v)
            elif k == "agent_exit":
                # Boolean fields are converted to strings in CSV
                assert rows[0][k] == str(v)
            else:
                assert rows[0][k] == v
    # Export to JSON
    export_file_json = tmp_path / "export_new.json"
    ds.export_to_json(str(export_file_json))
    with open(export_file_json, encoding="utf-8") as f:
        items = json.load(f)
        assert len(items) == 1
        for k, v in minimal_agent_data_dict().items():
            if k == "expected_tool_call":
                assert items[0][k] == v
            else:
                assert items[0][k] == v


# --- Additional tests for 100% coverage ---


@pytest.mark.unit
def test_parse_field_toolcall_exception(monkeypatch):
    ds = AgentDataset()
    # Patch ToolCall in agent_dataset and AgentData.model_fields annotation
    import novaeval.agents.agent_data as agent_data_mod
    import novaeval.datasets.agent_dataset as agent_dataset_mod

    orig_toolcall = agent_dataset_mod.ToolCall
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()

    class BadToolCall:
        def __init__(self, **kwargs):
            raise Exception("fail")

    monkeypatch.setattr(agent_dataset_mod, "ToolCall", BadToolCall)
    # Patch the annotation in model_fields to our BadToolCall
    from types import SimpleNamespace

    agent_data_mod.AgentData.model_fields["expected_tool_call"] = SimpleNamespace(
        annotation=BadToolCall
    )
    # Should return None if ToolCall init fails
    val = ds._parse_field(
        "expected_tool_call", {"tool_name": "t", "parameters": {}, "call_id": "c"}
    )
    assert val is None
    # Restore
    monkeypatch.setattr(agent_dataset_mod, "ToolCall", orig_toolcall)
    agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_parse_field_toolcall_non_str_non_dict():
    ds = AgentDataset()
    # For union fields, non-string non-dict values fall through to default behavior
    # Since expected_tool_call is a union type, non-string values are returned as-is and will be handled by Pydantic validation
    assert ds._parse_field("expected_tool_call", 123) == 123
    assert ds._parse_field("expected_tool_call", [1, 2, 3]) == [1, 2, 3]


@pytest.mark.unit
def test_parse_field_list_field_typeerror(monkeypatch):
    ds = AgentDataset()
    # Patch json.loads to raise TypeError
    monkeypatch.setattr(
        json, "loads", lambda v: (_ for _ in ()).throw(TypeError("fail"))
    )
    # For union fields, failed JSON parsing keeps the string
    assert ds._parse_field("trace", "[1,2,3]") == "[1,2,3]"


@pytest.mark.unit
def test_parse_field_dict_field_typeerror(monkeypatch):
    ds = AgentDataset()
    # Patch json.loads to raise TypeError
    monkeypatch.setattr(
        json, "loads", lambda v: (_ for _ in ()).throw(TypeError("fail"))
    )
    # For union fields, failed JSON parsing keeps the string
    assert ds._parse_field("parameters_passed", '{"x":1}') == '{"x":1}'


@pytest.mark.unit
def test_parse_field_default_return():
    ds = AgentDataset()
    # Patch AgentData.model_fields to add an int field
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()
    agent_data_mod.AgentData.model_fields["int_field"] = SimpleNamespace(annotation=int)
    # Should return value as is for int field
    assert ds._parse_field("int_field", 42) == 42
    agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_agentdataset_init_all_type_detection_branches(monkeypatch):
    import typing
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()

    # Custom types to simulate __origin__ and __args__
    class FakeList:
        __origin__ = list
        __args__ = (int,)

    class FakeDict:
        __origin__ = dict
        __args__ = (str, int)

    class FakeUnion:
        __origin__ = typing.Union
        __args__ = (list, type(None))

    class FakeUnion2:
        __origin__ = typing.Union
        __args__ = (FakeList, type(None))

    class FakeUnion3:
        __origin__ = typing.Union
        __args__ = (dict, type(None))

    class FakeUnion4:
        __origin__ = typing.Union
        __args__ = (FakeDict, type(None))

    model_fields = {
        "plain_list": SimpleNamespace(annotation=list),
        "plain_dict": SimpleNamespace(annotation=dict),
        "fake_list": SimpleNamespace(annotation=FakeList),
        "fake_dict": SimpleNamespace(annotation=FakeDict),
        "union_list": SimpleNamespace(annotation=FakeUnion),
        "union_list2": SimpleNamespace(annotation=FakeUnion2),
        "union_dict": SimpleNamespace(annotation=FakeUnion3),
        "union_dict2": SimpleNamespace(annotation=FakeUnion4),
    }
    monkeypatch.setattr(agent_data_mod.AgentData, "model_fields", model_fields)
    ds = AgentDataset()
    # All list fields
    assert "plain_list" in ds._list_fields
    assert "fake_list" in ds._list_fields
    assert "union_list" in ds._list_fields
    assert "union_list2" in ds._list_fields
    # All dict fields
    assert "plain_dict" in ds._dict_fields
    assert "fake_dict" in ds._dict_fields
    assert "union_dict" in ds._dict_fields
    assert "union_dict2" in ds._dict_fields
    # Restore
    agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_new_exit_fields_csv_json_ingestion(tmp_path):
    """Test that the new exit_status and agent_exit fields work properly in CSV and JSON ingestion."""
    # Test data with exit fields
    data_dict = {
        "agent_name": "TestAgent",
        "exit_status": "timeout",
        "agent_exit": True,
    }

    # Test CSV ingestion
    csv_file = tmp_path / "exit_test.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data_dict.keys())
        writer.writeheader()
        writer.writerow(data_dict)

    ds = AgentDataset()
    ds.ingest_from_csv(str(csv_file))
    assert len(ds.data) == 1
    agent = ds.data[0]
    assert agent.agent_name == "TestAgent"
    assert agent.exit_status == "timeout"
    assert agent.agent_exit is True

    # Test JSON ingestion
    json_file = tmp_path / "exit_test.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump([data_dict], f)

    ds2 = AgentDataset()
    ds2.ingest_from_json(str(json_file))
    assert len(ds2.data) == 1
    agent2 = ds2.data[0]
    assert agent2.agent_name == "TestAgent"
    assert agent2.exit_status == "timeout"
    assert agent2.agent_exit is True

    # Test export to CSV
    export_csv = tmp_path / "export_exit.csv"
    ds.export_to_csv(str(export_csv))
    with open(export_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["exit_status"] == "timeout"
        assert rows[0]["agent_exit"] == "True"  # Boolean becomes string in CSV

    # Test export to JSON
    export_json = tmp_path / "export_exit.json"
    ds.export_to_json(str(export_json))
    with open(export_json, encoding="utf-8") as f:
        items = json.load(f)
        assert len(items) == 1
        assert items[0]["exit_status"] == "timeout"
        assert items[0]["agent_exit"] is True  # Boolean stays boolean in JSON


@pytest.mark.unit
def test_exit_fields_with_field_mapping(tmp_path):
    """Test that field mapping works for the new exit fields."""
    data = {
        "agent_name": "TestAgent",
        "custom_exit_status": "error",
        "custom_agent_exit": False,
    }

    # Test CSV with field mapping
    csv_file = tmp_path / "mapped_exit.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

    ds = AgentDataset()
    ds.ingest_from_csv(
        str(csv_file), exit_status="custom_exit_status", agent_exit="custom_agent_exit"
    )
    assert len(ds.data) == 1
    agent = ds.data[0]
    assert agent.exit_status == "error"
    assert agent.agent_exit is False


# Test coverage improvements for missing lines


@pytest.mark.unit
def test_stream_from_csv_basic():
    """Test basic stream_from_csv functionality."""
    csv_data = """user_id,task_id,turn_id,ground_truth,agent_name,agent_response
user1,task1,turn1,truth1,agent1,response1
user2,task2,turn2,truth2,agent2,response2
user3,task3,turn3,truth3,agent3,response3"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_file:
        temp_file.write(csv_data)
        temp_file_path = temp_file.name

    try:
        ds = AgentDataset()
        chunks = list(ds.stream_from_csv(temp_file_path, chunk_size=2))

        assert len(chunks) == 2  # 3 items with chunk_size=2 gives 2 chunks
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 1

        # Check first chunk data
        assert chunks[0][0].user_id == "user1"
        assert chunks[0][0].task_id == "task1"
        assert chunks[0][1].user_id == "user2"

        # Check second chunk data
        assert chunks[1][0].user_id == "user3"

    finally:
        os.unlink(temp_file_path)


@pytest.mark.unit
def test_stream_from_csv_with_field_mapping():
    """Test stream_from_csv with custom field mapping."""
    csv_data = """custom_user,custom_task,custom_turn,custom_agent,custom_response
user1,task1,turn1,agent1,response1
user2,task2,turn2,agent2,response2"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_file:
        temp_file.write(csv_data)
        temp_file_path = temp_file.name

    try:
        ds = AgentDataset()
        chunks = list(
            ds.stream_from_csv(
                temp_file_path,
                chunk_size=10,
                user_id="custom_user",
                task_id="custom_task",
                turn_id="custom_turn",
                agent_name="custom_agent",
                agent_response="custom_response",
            )
        )

        assert len(chunks) == 1
        assert len(chunks[0]) == 2
        assert chunks[0][0].user_id == "user1"
        assert chunks[0][0].agent_name == "agent1"

    finally:
        os.unlink(temp_file_path)


@pytest.mark.unit
def test_stream_from_csv_file_error():
    """Test stream_from_csv with file read error."""
    ds = AgentDataset()

    with pytest.raises(ValueError, match="Error reading CSV file"):
        list(ds.stream_from_csv("/nonexistent/file.csv"))


def test_stream_from_csv_memory_management():
    """Test that stream_from_csv properly manages memory by deleting chunks."""
    csv_data = """user_id,task_id,turn_id,agent_name
user1,task1,turn1,agent1
user2,task2,turn2,agent2"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_file:
        temp_file.write(csv_data)
        temp_file_path = temp_file.name

    try:
        ds = AgentDataset()
        chunks = list(ds.stream_from_csv(temp_file_path, chunk_size=1))

        # Should have processed properly without memory issues
        assert len(chunks) == 2
        assert len(chunks[0]) == 1
        assert len(chunks[1]) == 1

    finally:
        os.unlink(temp_file_path)


def test_stream_from_json_basic():
    """Test basic stream_from_json functionality."""
    json_data = [
        {"user_id": "user1", "task_id": "task1", "agent_name": "agent1"},
        {"user_id": "user2", "task_id": "task2", "agent_name": "agent2"},
        {"user_id": "user3", "task_id": "task3", "agent_name": "agent3"},
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as temp_file:
        json.dump(json_data, temp_file)
        temp_file_path = temp_file.name

    try:
        ds = AgentDataset()
        chunks = list(ds.stream_from_json(temp_file_path, chunk_size=2))

        assert len(chunks) == 2  # 3 items with chunk_size=2 gives 2 chunks
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 1

        # Check data
        assert chunks[0][0].user_id == "user1"
        assert chunks[0][1].user_id == "user2"
        assert chunks[1][0].user_id == "user3"

    finally:
        os.unlink(temp_file_path)


@pytest.mark.unit
def test_stream_from_json_with_field_mapping():
    """Test stream_from_json with custom field mapping."""
    json_data = [
        {"custom_user": "user1", "custom_task": "task1", "custom_agent": "agent1"},
        {"custom_user": "user2", "custom_task": "task2", "custom_agent": "agent2"},
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as temp_file:
        json.dump(json_data, temp_file)
        temp_file_path = temp_file.name

    try:
        ds = AgentDataset()
        chunks = list(
            ds.stream_from_json(
                temp_file_path,
                chunk_size=10,
                user_id="custom_user",
                task_id="custom_task",
                agent_name="custom_agent",
            )
        )

        assert len(chunks) == 1
        assert len(chunks[0]) == 2
        assert chunks[0][0].user_id == "user1"
        assert chunks[0][0].agent_name == "agent1"

    finally:
        os.unlink(temp_file_path)


@pytest.mark.unit
def test_stream_from_json_non_dict_items():
    """Test stream_from_json skips non-dict items."""
    json_data = [
        {"user_id": "user1", "agent_name": "agent1"},
        "not a dict",  # Should be skipped
        {"user_id": "user2", "agent_name": "agent2"},
        123,  # Should be skipped
        {"user_id": "user3", "agent_name": "agent3"},
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as temp_file:
        json.dump(json_data, temp_file)
        temp_file_path = temp_file.name

    try:
        ds = AgentDataset()
        chunks = list(ds.stream_from_json(temp_file_path, chunk_size=10))

        # Should only have 3 items (the valid dicts)
        assert len(chunks) == 1
        assert len(chunks[0]) == 3
        assert chunks[0][0].user_id == "user1"
        assert chunks[0][1].user_id == "user2"
        assert chunks[0][2].user_id == "user3"

    finally:
        os.unlink(temp_file_path)


@pytest.mark.unit
def test_stream_from_json_import_error():
    """Test stream_from_json when ijson is not available."""
    ds = AgentDataset()

    with (
        patch(
            "builtins.__import__", side_effect=ImportError("No module named 'ijson'")
        ),
        pytest.raises(ImportError, match="ijson package is required"),
    ):
        list(ds.stream_from_json("dummy.json"))


@pytest.mark.unit
def test_stream_from_json_file_error():
    """Test stream_from_json with file read error."""
    ds = AgentDataset()

    with pytest.raises(ValueError, match="Error reading JSON file"):
        list(ds.stream_from_json("/nonexistent/file.json"))


@pytest.mark.unit
def test_stream_from_json_remaining_data():
    """Test that stream_from_json yields remaining data at the end."""
    json_data = [
        {"user_id": "user1"},
        {"user_id": "user2"},
        {"user_id": "user3"},
        {"user_id": "user4"},
        {"user_id": "user5"},
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as temp_file:
        json.dump(json_data, temp_file)
        temp_file_path = temp_file.name

    try:
        ds = AgentDataset()
        chunks = list(ds.stream_from_json(temp_file_path, chunk_size=3))

        # Should have 2 chunks: [3 items, 2 items]
        assert len(chunks) == 2
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 2  # Remaining data

    finally:
        os.unlink(temp_file_path)


@pytest.mark.unit
def test_parse_field_bool_string_variations():
    """Test boolean field parsing with various string inputs."""
    ds = AgentDataset()

    # Test true values
    assert ds._parse_field("agent_exit", "true") is True
    assert ds._parse_field("agent_exit", "True") is True
    assert ds._parse_field("agent_exit", "TRUE") is True
    assert ds._parse_field("agent_exit", "1") is True
    assert ds._parse_field("agent_exit", "yes") is True
    assert ds._parse_field("agent_exit", "YES") is True
    assert ds._parse_field("agent_exit", "on") is True
    assert ds._parse_field("agent_exit", "ON") is True

    # Test false values
    assert ds._parse_field("agent_exit", "false") is False
    assert ds._parse_field("agent_exit", "False") is False
    assert ds._parse_field("agent_exit", "FALSE") is False
    assert ds._parse_field("agent_exit", "0") is False
    assert ds._parse_field("agent_exit", "no") is False
    assert ds._parse_field("agent_exit", "NO") is False
    assert ds._parse_field("agent_exit", "off") is False
    assert ds._parse_field("agent_exit", "OFF") is False

    # Test unrecognized strings are kept as strings for union fields
    assert ds._parse_field("agent_exit", "maybe") == "maybe"
    assert ds._parse_field("agent_exit", "unknown") == "unknown"


@pytest.mark.unit
def test_parse_field_bool_numeric_conversions():
    """Test boolean field parsing with numeric inputs."""
    ds = AgentDataset()

    # For union fields, numeric values are kept as-is (handled by Pydantic)
    assert ds._parse_field("agent_exit", 1) == 1
    assert ds._parse_field("agent_exit", 0) == 0
    assert ds._parse_field("agent_exit", 42) == 42
    assert ds._parse_field("agent_exit", -1) == -1

    # Non-boolean strings are kept as strings for union fields
    assert ds._parse_field("agent_exit", "not_a_number") == "not_a_number"


@pytest.mark.unit
def test_parse_field_bool_exception_handling():
    """Test boolean field parsing exception handling."""
    ds = AgentDataset()

    # Test with objects that can't be converted to bool
    class UnconvertibleObj:
        def __bool__(self):
            raise ValueError("Cannot convert to bool")

    obj = UnconvertibleObj()
    result = ds._parse_field("agent_exit", obj)
    # For union fields, objects are kept as-is (handled by Pydantic)
    assert result is obj


@pytest.mark.unit
def test_parse_field_bool_none_with_default():
    """Test boolean field parsing with None value and field defaults."""
    ds = AgentDataset()

    # agent_exit field should have a default value
    result = ds._parse_field("agent_exit", None)
    assert result is False  # Should use default


@pytest.mark.unit
def test_parse_field_expected_tool_call_invalid_json():
    """Test parsing expected_tool_call with invalid JSON string."""
    ds = AgentDataset()

    # Test with invalid JSON (kept as string for union fields)
    result = ds._parse_field("expected_tool_call", "invalid json {")
    assert result == "invalid json {"

    # Test with valid JSON but invalid ToolCall structure (kept as string for union fields)
    result = ds._parse_field("expected_tool_call", '{"invalid": "structure"}')
    assert result == '{"invalid": "structure"}'


@pytest.mark.unit
def test_parse_field_expected_tool_call_exception():
    """Test parsing expected_tool_call with exception during ToolCall creation."""
    ds = AgentDataset()

    # Test with JSON that causes ToolCall constructor to fail (kept as string for union fields)
    invalid_toolcall_json = '{"tool_name": null, "parameters": "not_a_dict"}'
    result = ds._parse_field("expected_tool_call", invalid_toolcall_json)
    assert result == '{"tool_name": null, "parameters": "not_a_dict"}'


@pytest.mark.unit
def test_get_data_method():
    """Test get_data method returns a copy of the data."""
    ds = AgentDataset()
    ds.data = [
        AgentData(**minimal_agent_data_dict()),
        AgentData(**minimal_agent_data_dict()),
    ]

    data_copy = ds.get_data()

    # Should be a copy, not the same list
    assert data_copy is not ds.data
    assert len(data_copy) == len(ds.data)
    assert data_copy[0] == ds.data[0]  # But elements should be the same


@pytest.mark.unit
def test_get_datapoint_iterator():
    """Test get_datapoint method returns an iterator."""
    ds = AgentDataset()
    ds.data = [
        AgentData(**minimal_agent_data_dict()),
        AgentData(**minimal_agent_data_dict()),
    ]

    datapoints = list(ds.get_datapoint())

    assert len(datapoints) == 2
    assert all(isinstance(dp, AgentData) for dp in datapoints)
    assert datapoints[0] == ds.data[0]
    assert datapoints[1] == ds.data[1]


@pytest.mark.unit
def test_ingest_from_csv_complex_field_parsing():
    """Test complex field parsing scenarios in ingest_from_csv."""
    # Create CSV with various complex field types
    csv_data = """user_id,task_id,turn_id,trace,tools_available,tool_calls,agent_exit,expected_tool_call
user1,task1,turn1,"[{""step"": 1}]","[{""name"": ""tool1"", ""description"": ""desc"", ""args_schema"": {}, ""return_schema"": {}}]","[{""tool_name"": ""tool1"", ""parameters"": {}, ""call_id"": ""123""}]",1,"{""tool_name"": ""expected"", ""parameters"": {}, ""call_id"": ""123""}"
user2,task2,turn2,"","[]","[]",false,
user3,task3,turn3,invalid_json,invalid_json,invalid_json,maybe,invalid_json"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_file:
        temp_file.write(csv_data)
        temp_file_path = temp_file.name

    try:
        ds = AgentDataset()
        ds.ingest_from_csv(temp_file_path)

        assert len(ds.data) == 3

        # First row should parse correctly
        assert ds.data[0].user_id == "user1"
        assert ds.data[0].agent_exit is True  # "1" should convert to True
        assert isinstance(ds.data[0].trace, list)
        assert isinstance(ds.data[0].tools_available, list)
        assert ds.data[0].expected_tool_call is not None

        # Second row with empty/false values
        assert ds.data[1].agent_exit is False
        assert ds.data[1].trace == []
        assert ds.data[1].expected_tool_call is None

        # Third row with invalid data should handle gracefully
        assert (
            ds.data[2].agent_exit == "maybe"
        )  # Non-boolean strings are kept as strings for union fields
        assert (
            ds.data[2].trace == "invalid_json"
        )  # Invalid JSON kept as string for union fields
        assert (
            ds.data[2].expected_tool_call == "invalid_json"
        )  # Invalid JSON kept as string

    finally:
        os.unlink(temp_file_path)


@pytest.mark.unit
def test_parse_field_tool_call_instance():
    """Test parsing when value is already a ToolCall instance."""
    ds = AgentDataset()
    existing_toolcall = ToolCall(tool_name="test", parameters={}, call_id="123")
    result = ds._parse_field("expected_tool_call", existing_toolcall)
    assert result == existing_toolcall


@pytest.mark.unit
def test_parse_field_boolean_no_default():
    """Test boolean field parsing when field has no default value."""
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create field without default
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()
    agent_data_mod.AgentData.model_fields["test_bool"] = SimpleNamespace(
        annotation=bool
    )

    try:
        ds = AgentDataset()
        result = ds._parse_field("test_bool", None)
        assert result is False  # Should fallback to False
    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_parse_field_bool_with_default_value():
    """Test boolean field parsing when field has a default value."""
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create field with default
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()
    agent_data_mod.AgentData.model_fields["test_bool_with_default"] = SimpleNamespace(
        annotation=bool, default=True
    )

    try:
        ds = AgentDataset()
        # Test None value with default
        result = ds._parse_field("test_bool_with_default", None)
        assert result is True  # Should use the default value
    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_parse_field_bool_without_default_value():
    """Test boolean field parsing when field has no default value."""
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create field without default
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()
    agent_data_mod.AgentData.model_fields["test_bool_no_default"] = SimpleNamespace(
        annotation=bool
    )

    try:
        ds = AgentDataset()
        # Test None value without default
        result = ds._parse_field("test_bool_no_default", None)
        assert result is False  # Should fallback to False
    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_parse_field_bool_without_default_attr():
    """Test boolean field parsing when field has no default attribute."""
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create field without default attribute
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()
    field_without_default = SimpleNamespace(annotation=bool)
    # Remove default attribute if it exists
    if hasattr(field_without_default, "default"):
        delattr(field_without_default, "default")
    agent_data_mod.AgentData.model_fields["test_bool_no_default_attr"] = (
        field_without_default
    )

    try:
        ds = AgentDataset()
        # Test None value without default attribute
        result = ds._parse_field("test_bool_no_default_attr", None)
        assert result is False  # Should fallback to False
    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_parse_field_bool_unrecognized_string():
    """Test boolean field parsing with unrecognized string values."""
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create bool field
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()
    agent_data_mod.AgentData.model_fields["test_bool_unrecognized"] = SimpleNamespace(
        annotation=bool
    )

    try:
        ds = AgentDataset()
        # Test unrecognized string values
        result = ds._parse_field("test_bool_unrecognized", "maybe")
        assert result is False  # Should default to False for unrecognized strings

        result = ds._parse_field("test_bool_unrecognized", "unknown")
        assert result is False  # Should default to False for unrecognized strings
    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_agent_dataset_init_various_type_patterns():
    """Test AgentDataset initialization with various type annotation patterns."""
    import typing
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()

    # Test different type patterns that hit different branches
    class MockListType:
        __origin__ = list
        __args__ = (str,)

    class MockDictType:
        __origin__ = dict
        __args__ = (str, int)

    class MockUnionWithListOrigin:
        __origin__ = typing.Union
        __args__ = (MockListType, type(None))

    class MockUnionWithDictOrigin:
        __origin__ = typing.Union
        __args__ = (MockDictType, type(None))

    class MockUnionWithDirectList:
        __origin__ = typing.Union
        __args__ = (list, type(None))

    class MockUnionWithDirectDict:
        __origin__ = typing.Union
        __args__ = (dict, type(None))

    model_fields = {
        "union_list_origin": SimpleNamespace(annotation=MockUnionWithListOrigin),
        "union_dict_origin": SimpleNamespace(annotation=MockUnionWithDictOrigin),
        "union_direct_list": SimpleNamespace(annotation=MockUnionWithDirectList),
        "union_direct_dict": SimpleNamespace(annotation=MockUnionWithDirectDict),
        "direct_list": SimpleNamespace(annotation=list),
        "direct_dict": SimpleNamespace(annotation=dict),
    }

    agent_data_mod.AgentData.model_fields = model_fields

    try:
        ds = AgentDataset()
        # Verify the different patterns were detected correctly
        assert "union_list_origin" in ds._list_fields
        assert "union_dict_origin" in ds._dict_fields
        assert "union_direct_list" in ds._list_fields
        assert "union_direct_dict" in ds._dict_fields
        assert "direct_list" in ds._list_fields
        assert "direct_dict" in ds._dict_fields
    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_agentdataset_init_direct_types_without_union():
    """Test AgentDataset initialization with direct list/dict types (not in unions)."""
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()

    # Test direct list and dict types (not in unions)
    model_fields = {
        "direct_list_field": SimpleNamespace(annotation=list),
        "direct_dict_field": SimpleNamespace(annotation=dict),
        "typing_list_field": SimpleNamespace(annotation=list[str]),
        "typing_dict_field": SimpleNamespace(annotation=dict[str, int]),
    }

    agent_data_mod.AgentData.model_fields = model_fields

    try:
        ds = AgentDataset()
        # These should be detected as list/dict fields
        assert "direct_list_field" in ds._list_fields
        assert "direct_dict_field" in ds._dict_fields
        assert "typing_list_field" in ds._list_fields
        assert "typing_dict_field" in ds._dict_fields
    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_agentdataset_init_union_without_str():
    """Test AgentDataset initialization with Union types that don't include str."""
    import typing
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()

    # Test Union types without str
    class MockListType:
        __origin__ = list
        __args__ = (str,)

    class MockDictType:
        __origin__ = dict
        __args__ = (str, int)

    model_fields = {
        "union_list_no_str": SimpleNamespace(
            annotation=typing.Union[MockListType, None]
        ),
        "union_dict_no_str": SimpleNamespace(
            annotation=typing.Union[MockDictType, None]
        ),
        "union_direct_list_no_str": SimpleNamespace(
            annotation=typing.Union[list, None]
        ),
        "union_direct_dict_no_str": SimpleNamespace(
            annotation=typing.Union[dict, None]
        ),
    }

    agent_data_mod.AgentData.model_fields = model_fields

    try:
        ds = AgentDataset()
        # These should be detected as list/dict fields even without str in union
        assert "union_list_no_str" in ds._list_fields
        assert "union_dict_no_str" in ds._dict_fields
        assert "union_direct_list_no_str" in ds._list_fields
        assert "union_direct_dict_no_str" in ds._dict_fields
    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_agentdataset_init_union_with_str_and_other_types():
    """Test AgentDataset initialization with Union types that include str and other types."""
    import typing
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()

    # Test Union types with str and other types
    class MockListType:
        __origin__ = list
        __args__ = (str,)

    class MockDictType:
        __origin__ = dict
        __args__ = (str, int)

    model_fields = {
        "union_str_list": SimpleNamespace(
            annotation=typing.Union[str, MockListType, None]
        ),
        "union_str_dict": SimpleNamespace(
            annotation=typing.Union[str, MockDictType, None]
        ),
        "union_str_direct_list": SimpleNamespace(
            annotation=typing.Union[str, list, None]
        ),
        "union_str_direct_dict": SimpleNamespace(
            annotation=typing.Union[str, dict, None]
        ),
    }

    agent_data_mod.AgentData.model_fields = model_fields

    try:
        ds = AgentDataset()
        # These should be detected as union_with_str_fields
        assert "union_str_list" in ds._union_with_str_fields
        assert "union_str_dict" in ds._union_with_str_fields
        assert "union_str_direct_list" in ds._union_with_str_fields
        assert "union_str_direct_dict" in ds._union_with_str_fields
        # And also as list/dict fields
        assert "union_str_list" in ds._list_fields
        assert "union_str_dict" in ds._dict_fields
        assert "union_str_direct_list" in ds._list_fields
        assert "union_str_direct_dict" in ds._dict_fields
    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_ingest_csv_with_string_union_fields(tmp_path):
    """Test CSV ingestion with string union fields."""
    # Create CSV data where some union fields are strings and some are JSON
    data = {
        "user_id": "user1",
        "task_id": "task1",
        "agent_name": "TestAgent",
        "expected_tool_call": "string_representation_of_tool_call",
        "trace": "string_trace_info",
        "tools_available": '[{"name": "tool1", "description": "desc"}]',  # JSON
        "tool_calls": "string_tool_calls",
        "parameters_passed": '{"param1": "value1"}',  # JSON
        "tool_call_results": "string_results",
        "agent_exit": "custom_exit_state",
    }

    csv_file = tmp_path / "string_union_test.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

    ds = AgentDataset()
    ds.ingest_from_csv(str(csv_file))

    assert len(ds.data) == 1
    agent = ds.data[0]

    # Check that string values are preserved as strings
    assert agent.expected_tool_call == "string_representation_of_tool_call"
    assert agent.trace == "string_trace_info"
    assert agent.tool_calls == "string_tool_calls"
    assert agent.tool_call_results == "string_results"
    assert agent.agent_exit == "custom_exit_state"

    # Check that JSON-like strings are parsed
    assert isinstance(agent.tools_available, list)
    assert len(agent.tools_available) == 1
    # tools_available becomes a list of ToolSchema objects when AgentData is constructed
    assert agent.tools_available[0].name == "tool1"
    assert isinstance(agent.parameters_passed, dict)
    assert agent.parameters_passed["param1"] == "value1"


@pytest.mark.unit
def test_ingest_json_with_string_union_fields(tmp_path):
    """Test JSON ingestion with string union fields."""
    data = {
        "user_id": "user1",
        "task_id": "task1",
        "agent_name": "TestAgent",
        "expected_tool_call": "string_representation_of_tool_call",
        "trace": "string_trace_info",
        "tools_available": [{"name": "tool1", "description": "desc"}],  # Actual list
        "tool_calls": "string_tool_calls",
        "parameters_passed": {"param1": "value1"},  # Actual dict
        "tool_call_results": "string_results",
        "agent_exit": "custom_exit_state",
    }

    json_file = tmp_path / "string_union_test.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump([data], f)

    ds = AgentDataset()
    ds.ingest_from_json(str(json_file))

    assert len(ds.data) == 1
    agent = ds.data[0]

    # Check that string values are preserved as strings
    assert agent.expected_tool_call == "string_representation_of_tool_call"
    assert agent.trace == "string_trace_info"
    assert agent.tool_calls == "string_tool_calls"
    assert agent.tool_call_results == "string_results"
    assert agent.agent_exit == "custom_exit_state"

    # Check that actual list/dict values are preserved
    assert isinstance(agent.tools_available, list)
    assert len(agent.tools_available) == 1
    # tools_available becomes a list of ToolSchema objects when AgentData is constructed
    assert agent.tools_available[0].name == "tool1"
    assert isinstance(agent.parameters_passed, dict)
    assert agent.parameters_passed["param1"] == "value1"


@pytest.mark.unit
def test_export_preserves_string_union_values(tmp_path):
    """Test that export methods preserve string union field values."""
    # Create AgentData with mixed string/parsed values
    agent = AgentData(
        user_id="user1",
        task_id="task1",
        agent_name="TestAgent",
        expected_tool_call="string_tool_call",
        trace="string_trace",
        tools_available=[
            {
                "name": "tool1",
                "description": "desc",
                "args_schema": None,
                "return_schema": None,
            }
        ],  # List
        tool_calls="string_tool_calls",
        parameters_passed={"param1": "value1"},  # Dict
        tool_call_results="string_results",
        agent_exit="custom_exit",
    )

    ds = AgentDataset()
    ds.data.append(agent)

    # Test CSV export
    csv_file = tmp_path / "export_string_union.csv"
    ds.export_to_csv(str(csv_file))

    with open(csv_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        row = rows[0]

        # String values should be preserved
        assert row["expected_tool_call"] == "string_tool_call"
        assert row["trace"] == "string_trace"
        assert row["tool_calls"] == "string_tool_calls"
        assert row["tool_call_results"] == "string_results"
        assert row["agent_exit"] == "custom_exit"

        # List/dict values should be JSON serialized - check the structure matches what AgentData produces
        tools_data = json.loads(row["tools_available"])
        assert len(tools_data) == 1
        assert tools_data[0]["name"] == "tool1"
        assert tools_data[0]["description"] == "desc"
        assert json.loads(row["parameters_passed"]) == {"param1": "value1"}

    # Test JSON export
    json_file = tmp_path / "export_string_union.json"
    ds.export_to_json(str(json_file))

    with open(json_file, encoding="utf-8") as f:
        items = json.load(f)
        assert len(items) == 1
        item = items[0]

        # All values should be preserved as-is
        assert item["expected_tool_call"] == "string_tool_call"
        assert item["trace"] == "string_trace"
        assert item["tool_calls"] == "string_tool_calls"
        assert item["tool_call_results"] == "string_results"
        assert item["agent_exit"] == "custom_exit"
        # Check the structure that AgentData actually produces
        assert len(item["tools_available"]) == 1
        assert item["tools_available"][0]["name"] == "tool1"
        assert item["parameters_passed"] == {"param1": "value1"}


@pytest.mark.unit
def test_mixed_string_and_parsed_values_edge_cases():
    """Test edge cases with mixed string and parsed values."""
    ds = AgentDataset()

    # Test empty JSON-like strings
    assert ds._parse_field("tools_available", "[]") == []
    assert ds._parse_field("parameters_passed", "{}") == {}

    # Test whitespace handling
    assert ds._parse_field("tools_available", "  string_value  ") == "string_value"
    assert ds._parse_field("parameters_passed", "  string_value  ") == "string_value"

    # Test JSON parsing with whitespace
    result = ds._parse_field("tools_available", '  [{"name": "tool1"}]  ')
    assert isinstance(result, list)
    assert result[0]["name"] == "tool1"

    # Test malformed JSON-like strings that don't start/end correctly
    assert ds._parse_field("tools_available", "not_json]") == "not_json]"
    assert ds._parse_field("tools_available", "[not_json") == "[not_json"
    assert ds._parse_field("parameters_passed", "not_json}") == "not_json}"
    assert ds._parse_field("parameters_passed", "{not_json") == "{not_json"


@pytest.mark.unit
def test_parse_field_non_union_list_dict_fields():
    """Test parsing list and dict fields that are not union types."""
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create non-union list/dict fields
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()
    agent_data_mod.AgentData.model_fields["pure_list_field"] = SimpleNamespace(
        annotation=list
    )
    agent_data_mod.AgentData.model_fields["pure_dict_field"] = SimpleNamespace(
        annotation=dict
    )

    try:
        ds = AgentDataset()

        # Test list field with non-JSON string
        result = ds._parse_field("pure_list_field", "not_a_list")
        assert result == []  # Should return empty list for non-JSON strings

        # Test list field with valid JSON
        result = ds._parse_field("pure_list_field", "[1, 2, 3]")
        assert result == [1, 2, 3]  # Should parse valid JSON

        # Test list field with invalid JSON
        result = ds._parse_field("pure_list_field", "[invalid")
        assert result == []  # Should return empty list for invalid JSON

        # Test list field with actual list
        result = ds._parse_field("pure_list_field", [4, 5, 6])
        assert result == [4, 5, 6]  # Should keep actual list

        # Test list field with non-list non-string
        result = ds._parse_field("pure_list_field", 42)
        assert result == []  # Should return empty list for non-list non-string

        # Test dict field with non-JSON string
        result = ds._parse_field("pure_dict_field", "not_a_dict")
        assert result == {}  # Should return empty dict for non-JSON strings

        # Test dict field with valid JSON
        result = ds._parse_field("pure_dict_field", '{"a": 1, "b": 2}')
        assert result == {"a": 1, "b": 2}  # Should parse valid JSON

        # Test dict field with invalid JSON
        result = ds._parse_field("pure_dict_field", "{invalid")
        assert result == {}  # Should return empty dict for invalid JSON

        # Test dict field with actual dict
        result = ds._parse_field("pure_dict_field", {"c": 3, "d": 4})
        assert result == {"c": 3, "d": 4}  # Should keep actual dict

        # Test dict field with non-dict non-string
        result = ds._parse_field("pure_dict_field", 42)
        assert result == {}  # Should return empty dict for non-dict non-string

    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_parse_field_toolcall_without_str():
    """Test parsing ToolCall fields that don't include str in their type."""
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create ToolCall field without str
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()
    agent_data_mod.AgentData.model_fields["pure_toolcall_field"] = SimpleNamespace(
        annotation=ToolCall
    )

    try:
        ds = AgentDataset()

        # Test with None value
        result = ds._parse_field("pure_toolcall_field", None)
        assert result is None  # Should return None for None

        # Test with empty string
        result = ds._parse_field("pure_toolcall_field", "")
        assert result is None  # Should return None for empty string

        # Test with existing ToolCall instance
        existing_toolcall = ToolCall(tool_name="test", parameters={}, call_id="123")
        result = ds._parse_field("pure_toolcall_field", existing_toolcall)
        assert result == existing_toolcall  # Should keep existing ToolCall

        # Test with valid JSON string
        result = ds._parse_field(
            "pure_toolcall_field",
            '{"tool_name": "test", "parameters": {}, "call_id": "456"}',
        )
        assert isinstance(result, ToolCall)  # Should create new ToolCall
        assert result.tool_name == "test"
        assert result.call_id == "456"

        # Test with invalid JSON string
        result = ds._parse_field("pure_toolcall_field", "invalid_json")
        assert result is None  # Should return None for invalid JSON

        # Test with valid dict
        result = ds._parse_field(
            "pure_toolcall_field",
            {"tool_name": "test2", "parameters": {"p": "v"}, "call_id": "789"},
        )
        assert isinstance(result, ToolCall)  # Should create new ToolCall
        assert result.tool_name == "test2"
        assert result.parameters == {"p": "v"}

        # Test with invalid dict (missing required fields)
        result = ds._parse_field("pure_toolcall_field", {"invalid": "dict"})
        assert result is None  # Should return None for invalid dict

        # Test with non-dict non-string non-ToolCall
        result = ds._parse_field("pure_toolcall_field", 42)
        assert result is None  # Should return None for other types

    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_parse_field_str_only_fields():
    """Test parsing fields that are only str type (not union)."""
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create str-only field
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()
    agent_data_mod.AgentData.model_fields["pure_str_field"] = SimpleNamespace(
        annotation=str
    )

    try:
        ds = AgentDataset()

        # Test with None value
        result = ds._parse_field("pure_str_field", None)
        assert result is None  # Should return None for None

        # Test with string value
        result = ds._parse_field("pure_str_field", "test_string")
        assert result == "test_string"  # Should keep string

        # Test with non-string value
        result = ds._parse_field("pure_str_field", 42)
        assert result == "42"  # Should convert to string

    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_parse_field_union_str_only():
    """Test parsing fields that are Union[str, None] (only str and None)."""
    import typing
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create Union[str, None] field
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()
    agent_data_mod.AgentData.model_fields["union_str_only_field"] = SimpleNamespace(
        annotation=typing.Union[str, type(None)]
    )

    try:
        ds = AgentDataset()

        # Test with None value
        result = ds._parse_field("union_str_only_field", None)
        assert result is None  # Should return None for None

        # Test with string value
        result = ds._parse_field("union_str_only_field", "test_string")
        assert result == "test_string"  # Should keep string

        # Test with non-string value
        result = ds._parse_field("union_str_only_field", 42)
        assert result == "42"  # Should convert to string

    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_parse_field_union_fields_none_values():
    """Test parsing union fields with None values for different field types."""
    import typing
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create union fields
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()

    # Create fields that simulate the actual union types in AgentData
    agent_data_mod.AgentData.model_fields["union_list_field"] = SimpleNamespace(
        annotation=typing.Union[list, str, type(None)]
    )
    agent_data_mod.AgentData.model_fields["union_dict_field"] = SimpleNamespace(
        annotation=typing.Union[dict, str, type(None)]
    )
    agent_data_mod.AgentData.model_fields["union_toolcall_field"] = SimpleNamespace(
        annotation=typing.Union[ToolCall, str, type(None)]
    )
    agent_data_mod.AgentData.model_fields["union_bool_field"] = SimpleNamespace(
        annotation=typing.Union[bool, str, type(None)]
    )

    try:
        ds = AgentDataset()

        # Test None values for different union field types
        # These should hit the specific None handling branches

        # For list union fields, return empty list
        result = ds._parse_field("union_list_field", None)
        assert result == []

        # For dict union fields, return empty dict
        result = ds._parse_field("union_dict_field", None)
        assert result == {}

        # For ToolCall union fields, return None
        result = ds._parse_field("union_toolcall_field", None)
        assert result is None

        # For bool union fields, return False (default)
        result = ds._parse_field("union_bool_field", None)
        assert result is False

    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_parse_field_union_fields_other_types():
    """Test parsing union fields with non-string non-None values."""
    import typing
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create union fields
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()

    # Create fields that simulate the actual union types in AgentData
    agent_data_mod.AgentData.model_fields["union_list_field"] = SimpleNamespace(
        annotation=typing.Union[list, str, type(None)]
    )
    agent_data_mod.AgentData.model_fields["union_dict_field"] = SimpleNamespace(
        annotation=typing.Union[dict, str, type(None)]
    )
    agent_data_mod.AgentData.model_fields["union_toolcall_field"] = SimpleNamespace(
        annotation=typing.Union[ToolCall, str, type(None)]
    )

    try:
        ds = AgentDataset()

        # Test with non-string non-None values (should fall through to regular parsing)
        # These should hit the "If value is not a string, fall through" branch

        # Test with list value
        result = ds._parse_field("union_list_field", [1, 2, 3])
        assert result == [1, 2, 3]

        # Test with dict value
        result = ds._parse_field("union_dict_field", {"a": 1})
        assert result == {"a": 1}

        # Test with ToolCall value
        toolcall = ToolCall(tool_name="test", parameters={}, call_id="123")
        result = ds._parse_field("union_toolcall_field", toolcall)
        assert result == toolcall

    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_stream_from_json_exit_path():
    """Test the exit path in stream_from_json method."""
    ds = AgentDataset()

    # Test with a file that doesn't exist to trigger the exception handling
    with pytest.raises(ValueError, match="Error reading JSON file"):
        list(ds.stream_from_json("/nonexistent/file.json"))


@pytest.mark.unit
def test_parse_field_nan_handling():
    """Test handling of NaN values from pandas."""
    import math
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create a test field
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()
    agent_data_mod.AgentData.model_fields["test_nan_field"] = SimpleNamespace(
        annotation=str
    )

    try:
        ds = AgentDataset()

        # Test with NaN float value
        result = ds._parse_field("test_nan_field", float("nan"))
        assert result is None  # Should convert NaN to None

        # Test with math.nan
        result = ds._parse_field("test_nan_field", math.nan)
        assert result is None  # Should convert NaN to None

    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_parse_field_remaining_edge_cases():
    """Test remaining edge cases in _parse_field method."""
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create test fields
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()

    # Create fields for different test scenarios
    agent_data_mod.AgentData.model_fields["test_field_with_default"] = SimpleNamespace(
        annotation=bool, default=True
    )
    agent_data_mod.AgentData.model_fields["test_field_no_default"] = SimpleNamespace(
        annotation=bool
    )

    try:
        ds = AgentDataset()

        # Test bool field with default value (line 128)
        result = ds._parse_field("test_field_with_default", None)
        assert result is True  # Should use the default value

        # Test bool field without default value (line 194)
        result = ds._parse_field("test_field_no_default", None)
        assert result is False  # Should fallback to False

        # Test bool field with unrecognized string (lines 274, 278, 280)
        result = ds._parse_field("test_field_no_default", "maybe")
        assert result is False  # Should default to False for unrecognized strings

        result = ds._parse_field("test_field_no_default", "unknown")
        assert result is False  # Should default to False for unrecognized strings

        # Test bool field with conversion exception (lines 213-214, 227-228)
        class UnconvertibleObj:
            def __bool__(self):
                raise ValueError("Cannot convert to bool")

        obj = UnconvertibleObj()
        result = ds._parse_field("test_field_no_default", obj)
        assert result is False  # Should fallback to False on exception

    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_parse_field_final_edge_cases():
    """Test final edge cases in _parse_field method to achieve 95%+ coverage."""
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create test fields
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()

    # Create fields for specific test scenarios
    agent_data_mod.AgentData.model_fields["bool_field_with_default"] = SimpleNamespace(
        annotation=bool, default=True
    )
    agent_data_mod.AgentData.model_fields["bool_field_no_default"] = SimpleNamespace(
        annotation=bool
    )
    agent_data_mod.AgentData.model_fields["bool_field_no_default_attr"] = (
        SimpleNamespace(annotation=bool)
    )

    try:
        ds = AgentDataset()

        # Test bool field with default value (line 128)
        result = ds._parse_field("bool_field_with_default", None)
        assert result is True  # Should use the default value

        # Test bool field without default value (line 194)
        result = ds._parse_field("bool_field_no_default", None)
        assert result is False  # Should fallback to False

        # Test bool field with unrecognized string (lines 274, 278, 280)
        result = ds._parse_field("bool_field_no_default", "maybe")
        assert result is False  # Should default to False for unrecognized strings

        result = ds._parse_field("bool_field_no_default", "unknown")
        assert result is False  # Should default to False for unrecognized strings

        # Test bool field with conversion exception (lines 213-214, 227-228)
        class UnconvertibleObj:
            def __bool__(self):
                raise ValueError("Cannot convert to bool")

        obj = UnconvertibleObj()
        result = ds._parse_field("bool_field_no_default", obj)
        assert result is False  # Should fallback to False on exception

        # Test bool field without default attribute (lines 175-178)
        # Remove default attribute if it exists
        if hasattr(
            agent_data_mod.AgentData.model_fields["bool_field_no_default_attr"],
            "default",
        ):
            delattr(
                agent_data_mod.AgentData.model_fields["bool_field_no_default_attr"],
                "default",
            )

        result = ds._parse_field("bool_field_no_default_attr", None)
        assert result is False  # Should fallback to False

    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_stream_from_json_final_exit_path():
    """Test the final exit path in stream_from_json method."""
    ds = AgentDataset()

    # Test with a file that doesn't exist to trigger the exception handling (line 632)
    with pytest.raises(ValueError, match="Error reading JSON file"):
        list(ds.stream_from_json("/nonexistent/file.json"))


@pytest.mark.unit
def test_agentdataset_init_final_missing_branches():
    """Test AgentDataset initialization with specific type patterns to cover missing branches."""
    import typing
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()

    # Test specific patterns that hit the missing branches
    class MockListType:
        __origin__ = list
        __args__ = (str,)

    class MockDictType:
        __origin__ = dict
        __args__ = (str, int)

    # Test Union types with specific patterns that hit missing branches
    class MockUnionWithStrAndList:
        __origin__ = typing.Union
        __args__ = (str, MockListType, type(None))

    class MockUnionWithStrAndDict:
        __origin__ = typing.Union
        __args__ = (str, MockDictType, type(None))

    # Test Union types without str that hit missing branches
    class MockUnionWithoutStrList:
        __origin__ = typing.Union
        __args__ = (MockListType, type(None))

    class MockUnionWithoutStrDict:
        __origin__ = typing.Union
        __args__ = (MockDictType, type(None))

    model_fields = {
        # Test branches 55->50, 60->50 (Union with str and list/dict)
        "union_str_list": SimpleNamespace(annotation=MockUnionWithStrAndList),
        "union_str_dict": SimpleNamespace(annotation=MockUnionWithStrAndDict),
        # Test branches 69->64, 74->64 (Union without str and list/dict)
        "union_no_str_list": SimpleNamespace(annotation=MockUnionWithoutStrList),
        "union_no_str_dict": SimpleNamespace(annotation=MockUnionWithoutStrDict),
        # Test branches 81->22, 86->22 (direct list/dict types)
        "direct_list": SimpleNamespace(annotation=list),
        "direct_dict": SimpleNamespace(annotation=dict),
    }

    agent_data_mod.AgentData.model_fields = model_fields

    try:
        ds = AgentDataset()

        # Verify the fields were detected correctly
        assert "union_str_list" in ds._list_fields
        assert "union_str_dict" in ds._dict_fields
        assert "union_str_list" in ds._union_with_str_fields
        assert "union_str_dict" in ds._union_with_str_fields

        assert "union_no_str_list" in ds._list_fields
        assert "union_no_str_dict" in ds._dict_fields

        assert "direct_list" in ds._list_fields
        assert "direct_dict" in ds._dict_fields

    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_parse_field_specific_missing_branches():
    """Test specific missing branches in _parse_field method."""
    import typing
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create test fields
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()

    # Create fields for specific test scenarios
    agent_data_mod.AgentData.model_fields["toolcall_only_field"] = SimpleNamespace(
        annotation=ToolCall  # ToolCall without str in union
    )
    agent_data_mod.AgentData.model_fields["bool_only_field"] = SimpleNamespace(
        annotation=bool  # bool without str in union
    )
    agent_data_mod.AgentData.model_fields["str_only_union_field"] = SimpleNamespace(
        annotation=typing.Union[str, type(None)]  # Union with only str and None
    )

    try:
        ds = AgentDataset()

        # Test ToolCall field without str in union (lines 213-214, 227-228)
        # Test with None value
        result = ds._parse_field("toolcall_only_field", None)
        assert result is None

        # Test with empty string
        result = ds._parse_field("toolcall_only_field", "")
        assert result is None

        # Test with invalid JSON string
        result = ds._parse_field("toolcall_only_field", "invalid json")
        assert result is None

        # Test with valid JSON but invalid ToolCall data
        result = ds._parse_field("toolcall_only_field", '{"invalid": "data"}')
        assert result is None

        # Test with non-string non-dict value
        result = ds._parse_field("toolcall_only_field", 123)
        assert result is None

        # Test str-only union field (lines 274, 278, 280)
        result = ds._parse_field("str_only_union_field", None)
        assert result is None

        result = ds._parse_field("str_only_union_field", 123)
        assert result == "123"

        # Test bool-only field (lines 194)
        result = ds._parse_field("bool_only_field", None)
        assert result is False  # Should use default

    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_parse_field_final_missing_branches():
    """Test final missing branches in _parse_field method."""
    import math
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    # Save original and create test fields
    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()

    # Create fields for specific test scenarios
    agent_data_mod.AgentData.model_fields["test_field"] = SimpleNamespace(
        annotation=str
    )

    try:
        ds = AgentDataset()

        # Test NaN handling (line 194)
        nan_value = float("nan")
        result = ds._parse_field("test_field", nan_value)
        assert result is None  # NaN should be converted to None

        # Test with math.nan
        result = ds._parse_field("test_field", math.nan)
        assert result is None  # math.nan should also be converted to None

        # Test with a custom object that has "nan" string representation
        class NanObject:
            def __str__(self):
                return "nan"

        nan_obj = NanObject()
        result = ds._parse_field("test_field", nan_obj)
        assert result == "nan"  # Should be converted to string

    finally:
        agent_data_mod.AgentData.model_fields = orig_model_fields


@pytest.mark.unit
def test_stream_from_json_exit_path_final():
    """Test the exit path in stream_from_json method."""
    ds = AgentDataset()

    # Test with a file that doesn't exist to hit the exception handler
    with pytest.raises(ValueError, match="Error reading JSON file"):
        list(ds.stream_from_json("/nonexistent/file.json"))


@pytest.mark.unit
def test_ingest_from_csv_file_not_found():
    ds = AgentDataset()
    with pytest.raises(ValueError) as exc:
        ds.ingest_from_csv("/nonexistent/file/path.csv")
    assert "CSV file not found" in str(exc.value)


@pytest.mark.unit
def test_ingest_from_csv_permission_denied(monkeypatch, tmp_path):
    # Simulate PermissionError by monkeypatching pandas.read_csv
    csv_file = tmp_path / "perm.csv"
    csv_file.write_text("agent_name,agent_role\nA,B", encoding="utf-8")

    def raise_permission(*args, **kwargs):
        raise PermissionError

    monkeypatch.setattr("pandas.read_csv", raise_permission)
    ds = AgentDataset()
    with pytest.raises(ValueError) as exc:
        ds.ingest_from_csv(str(csv_file))
    assert "Permission denied" in str(exc.value)


@pytest.mark.unit
def test_ingest_from_csv_empty_data_error(monkeypatch, tmp_path):
    # Simulate EmptyDataError
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("", encoding="utf-8")

    def raise_empty_data(*args, **kwargs):
        import pandas as pd

        raise pd.errors.EmptyDataError("No data")

    monkeypatch.setattr("pandas.read_csv", raise_empty_data)
    ds = AgentDataset()
    with pytest.raises(ValueError) as exc:
        ds.ingest_from_csv(str(csv_file))
    assert "empty or contains no data" in str(exc.value)


@pytest.mark.unit
def test_ingest_from_csv_parser_error(monkeypatch, tmp_path):
    # Simulate ParserError
    csv_file = tmp_path / "malformed.csv"
    csv_file.write_text("invalid,csv,format", encoding="utf-8")

    def raise_parser_error(*args, **kwargs):
        import pandas as pd

        raise pd.errors.ParserError("Parser error")

    monkeypatch.setattr("pandas.read_csv", raise_parser_error)
    ds = AgentDataset()
    with pytest.raises(ValueError) as exc:
        ds.ingest_from_csv(str(csv_file))
    assert "Error parsing CSV file" in str(exc.value)


@pytest.mark.unit
def test_stream_from_csv_file_not_found():
    ds = AgentDataset()
    with pytest.raises(ValueError) as exc:
        list(ds.stream_from_csv("/nonexistent/file/path.csv"))
    assert "Error reading CSV file" in str(exc.value)


@pytest.mark.unit
def test_parse_field_union_str_only_edge_case():
    ds = AgentDataset()
    # Test the case where a field is Union[str, None] only
    # This tests the specific condition: len([a for a in args if a is not type(None)]) == 1
    result = ds._parse_field("user_id", "test_value")
    assert result == "test_value"


@pytest.mark.unit
def test_parse_field_union_str_only_none():
    ds = AgentDataset()
    # Test the case where a field is Union[str, None] and value is None
    result = ds._parse_field("user_id", None)
    assert result is None


@pytest.mark.unit
def test_parse_field_bool_conversion_exception():
    ds = AgentDataset()

    # Test bool conversion with exception
    class UnconvertibleObj:
        def __bool__(self):
            raise ValueError("Cannot convert")

    result = ds._parse_field("agent_exit", UnconvertibleObj())
    # Since agent_exit is in union with str, it should return the object as-is
    assert isinstance(result, UnconvertibleObj)


@pytest.mark.unit
def test_agentdataset_init_modern_union_syntax(monkeypatch):
    # Test handling of modern | union syntax
    from types import SimpleNamespace

    import novaeval.agents.agent_data as agent_data_mod

    orig_model_fields = agent_data_mod.AgentData.model_fields.copy()

    class ModernUnionType:
        __args__ = (str, int, type(None))
        __origin__ = None

    model_fields = {
        "modern_union_field": SimpleNamespace(annotation=ModernUnionType),
    }

    monkeypatch.setattr(agent_data_mod.AgentData, "model_fields", model_fields)
    ds = AgentDataset()

    # Should not raise any exceptions
    assert hasattr(ds, "_union_with_str_fields")

    monkeypatch.setattr(agent_data_mod.AgentData, "model_fields", orig_model_fields)


@pytest.mark.unit
def test_validate_retrieval_fields_matching_lengths():
    """Test that _validate_retrieval_fields passes when lengths match."""
    ds = AgentDataset()

    # Test with matching lengths
    data_kwargs = {
        "retrieval_query": ["query1", "query2"],
        "retrieved_context": [["context1"], ["context2"]],
    }

    # Should not raise any exception
    ds._validate_retrieval_fields(data_kwargs)


@pytest.mark.unit
def test_validate_retrieval_fields_mismatched_lengths():
    """Test that _validate_retrieval_fields raises ValueError when lengths don't match."""
    ds = AgentDataset()

    # Test with mismatched lengths
    data_kwargs = {
        "retrieval_query": ["query1", "query2"],
        "retrieved_context": [["context1"]],  # Only one context for two queries
    }

    with pytest.raises(
        ValueError,
        match="Length mismatch: retrieval_query has 2 queries but retrieved_context has 1 context lists",
    ):
        ds._validate_retrieval_fields(data_kwargs)


@pytest.mark.unit
def test_validate_retrieval_fields_none_values():
    """Test that _validate_retrieval_fields skips validation when fields are None."""
    ds = AgentDataset()

    # Test with None values
    data_kwargs = {"retrieval_query": None, "retrieved_context": ["context1"]}

    # Should not raise any exception
    ds._validate_retrieval_fields(data_kwargs)

    # Test with both None
    data_kwargs = {"retrieval_query": None, "retrieved_context": None}

    # Should not raise any exception
    ds._validate_retrieval_fields(data_kwargs)


@pytest.mark.unit
def test_validate_retrieval_fields_non_list_values():
    """Test that _validate_retrieval_fields skips validation when fields are not lists."""
    ds = AgentDataset()

    # Test with non-list values
    data_kwargs = {"retrieval_query": "not a list", "retrieved_context": ["context1"]}

    # Should not raise any exception
    ds._validate_retrieval_fields(data_kwargs)

    # Test with both non-list
    data_kwargs = {
        "retrieval_query": "not a list",
        "retrieved_context": "also not a list",
    }

    # Should not raise any exception
    ds._validate_retrieval_fields(data_kwargs)


@pytest.mark.unit
def test_validate_retrieval_fields_empty_lists():
    """Test that _validate_retrieval_fields passes with empty lists."""
    ds = AgentDataset()

    # Test with empty lists
    data_kwargs = {"retrieval_query": [], "retrieved_context": []}

    # Should not raise any exception
    ds._validate_retrieval_fields(data_kwargs)


@pytest.mark.unit
def test_ingest_from_csv_with_retrieval_validation(tmp_path):
    """Test that CSV ingestion calls validation for retrieval fields."""
    # Create CSV data with retrieval fields - all rows have matching lengths
    csv_data = """user_id,task_id,agent_name,retrieval_query,retrieved_context
user1,task1,agent1,"[""query1"", ""query2""]","[[""context1""], [""context2""]]"
user2,task2,agent2,"[""query3""]","[[""context3""]]"
user3,task3,agent3,"[""query4"", ""query5""]","[[""context4""], [""context5""]]"
"""

    csv_file = tmp_path / "retrieval_test.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        f.write(csv_data)

    ds = AgentDataset()

    # All rows should work fine since they have matching lengths
    ds.ingest_from_csv(str(csv_file))

    # Should have 3 valid rows
    assert len(ds.data) == 3
    assert ds.data[0].user_id == "user1"
    assert ds.data[1].user_id == "user2"
    assert ds.data[2].user_id == "user3"


@pytest.mark.unit
def test_ingest_from_json_with_retrieval_validation(tmp_path):
    """Test that JSON ingestion calls validation for retrieval fields."""
    # Create JSON data with retrieval fields
    json_data = [
        {
            "user_id": "user1",
            "task_id": "task1",
            "agent_name": "agent1",
            "retrieval_query": ["query1", "query2"],
            "retrieved_context": [["context1"], ["context2"]],
        },
        {
            "user_id": "user2",
            "task_id": "task2",
            "agent_name": "agent2",
            "retrieval_query": ["query3"],
            "retrieved_context": [["context3"]],
        },
    ]

    json_file = tmp_path / "retrieval_test.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    ds = AgentDataset()

    # Both rows should work fine since they have matching lengths
    ds.ingest_from_json(str(json_file))

    # Should have 2 valid rows
    assert len(ds.data) == 2
    assert ds.data[0].user_id == "user1"
    assert ds.data[1].user_id == "user2"


@pytest.mark.unit
def test_ingest_from_json_with_retrieval_validation_error(tmp_path):
    """Test that JSON ingestion raises validation error for mismatched retrieval fields."""
    # Create JSON data with mismatched retrieval fields
    json_data = [
        {
            "user_id": "user1",
            "task_id": "task1",
            "agent_name": "agent1",
            "retrieval_query": ["query1", "query2"],
            "retrieved_context": [["context1"]],  # Mismatched lengths
        }
    ]

    json_file = tmp_path / "retrieval_error_test.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    ds = AgentDataset()

    # Should raise validation error
    with pytest.raises(
        ValueError,
        match="Length mismatch: retrieval_query has 2 queries but retrieved_context has 1 context lists",
    ):
        ds.ingest_from_json(str(json_file))


@pytest.mark.unit
def test_ingest_from_csv_retrieval_field_mapping(tmp_path):
    """Test that CSV ingestion with field mapping works for retrieval fields."""
    # Create CSV with custom column names for retrieval fields
    csv_data = """custom_user,custom_task,custom_agent,custom_query,custom_context
user1,task1,agent1,"[""query1""]","[[""context1""]]"
user2,task2,agent2,"[""query2"", ""query3""]","[[""context2""], [""context3""]]"
"""

    csv_file = tmp_path / "retrieval_mapping_test.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        f.write(csv_data)

    ds = AgentDataset()

    # Use field mapping for retrieval fields
    ds.ingest_from_csv(
        str(csv_file),
        user_id="custom_user",
        task_id="custom_task",
        agent_name="custom_agent",
        retrieval_query="custom_query",
        retrieved_context="custom_context",
    )

    assert len(ds.data) == 2
    assert ds.data[0].user_id == "user1"
    assert ds.data[0].retrieval_query == ["query1"]
    assert ds.data[0].retrieved_context == [["context1"]]
    assert ds.data[1].user_id == "user2"
    assert ds.data[1].retrieval_query == ["query2", "query3"]
    assert ds.data[1].retrieved_context == [["context2"], ["context3"]]


@pytest.mark.unit
def test_ingest_from_json_retrieval_field_mapping(tmp_path):
    """Test that JSON ingestion with field mapping works for retrieval fields."""
    # Create JSON with custom field names for retrieval fields
    json_data = [
        {
            "custom_user": "user1",
            "custom_task": "task1",
            "custom_agent": "agent1",
            "custom_query": ["query1"],
            "custom_context": [["context1"]],
        },
        {
            "custom_user": "user2",
            "custom_task": "task2",
            "custom_agent": "agent2",
            "custom_query": ["query2", "query3"],
            "custom_context": [["context2"], ["context3"]],
        },
    ]

    json_file = tmp_path / "retrieval_mapping_test.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    ds = AgentDataset()

    # Use field mapping for retrieval fields
    ds.ingest_from_json(
        str(json_file),
        user_id="custom_user",
        task_id="custom_task",
        agent_name="custom_agent",
        retrieval_query="custom_query",
        retrieved_context="custom_context",
    )

    assert len(ds.data) == 2
    assert ds.data[0].user_id == "user1"
    assert ds.data[0].retrieval_query == ["query1"]
    assert ds.data[0].retrieved_context == [["context1"]]
    assert ds.data[1].user_id == "user2"
    assert ds.data[1].retrieval_query == ["query2", "query3"]
    assert ds.data[1].retrieved_context == [["context2"], ["context3"]]


@pytest.mark.unit
def test_validate_retrieval_fields_mixed_types():
    """Test validation with mixed types in retrieval fields."""
    ds = AgentDataset()

    # Test with one list and one non-list
    data_kwargs = {
        "retrieval_query": ["query1", "query2"],
        "retrieved_context": "not a list",
    }

    # Should not raise any exception (skips validation)
    ds._validate_retrieval_fields(data_kwargs)

    # Test with one list and one None
    data_kwargs = {"retrieval_query": ["query1", "query2"], "retrieved_context": None}

    # Should not raise any exception (skips validation)
    ds._validate_retrieval_fields(data_kwargs)


@pytest.mark.unit
def test_validate_retrieval_fields_single_item_lists():
    """Test validation with single item lists."""
    ds = AgentDataset()

    # Test with single item lists
    data_kwargs = {"retrieval_query": ["query1"], "retrieved_context": [["context1"]]}

    # Should not raise any exception
    ds._validate_retrieval_fields(data_kwargs)


@pytest.mark.unit
def test_validate_retrieval_fields_large_lists():
    """Test validation with larger lists."""
    ds = AgentDataset()

    # Test with larger lists
    queries = [f"query{i}" for i in range(10)]
    contexts = [[f"context{i}"] for i in range(10)]

    data_kwargs = {"retrieval_query": queries, "retrieved_context": contexts}

    # Should not raise any exception
    ds._validate_retrieval_fields(data_kwargs)

    # Test with mismatched larger lists
    data_kwargs = {
        "retrieval_query": queries,
        "retrieved_context": contexts[:-1],  # One less context
    }

    with pytest.raises(
        ValueError,
        match="Length mismatch: retrieval_query has 10 queries but retrieved_context has 9 context lists",
    ):
        ds._validate_retrieval_fields(data_kwargs)
