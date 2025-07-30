import csv
import json
import os
import tempfile
from unittest.mock import patch

import pytest

from novaeval.agents.agent_data import AgentData
from novaeval.agents.agent_dataset import AgentDataset, ToolCall


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
        "retrieval_query": "query1",
        "retrieved_context": "context1",
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
            ]
        },
        "trace": json.dumps(d["trace"]),
        "tools_available": json.dumps(d["tools_available"]),
        "tool_calls": json.dumps(d["tool_calls"]),
        "parameters_passed": json.dumps(d["parameters_passed"]),
        "tool_call_results": json.dumps(d["tool_call_results"]),
        "expected_tool_call": json.dumps(d["expected_tool_call"]),
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
    assert not export_file.exists() or export_file.read_text() == ""


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
    # List fields: invalid JSON string
    assert ds._parse_field("trace", "[invalid]") == []
    # List fields: already a list
    assert ds._parse_field("trace", [{"a": 2}]) == [{"a": 2}]
    # List fields: wrong type
    assert ds._parse_field("trace", 123) == []
    # Dict fields: valid JSON string
    assert ds._parse_field("parameters_passed", '{"x":1}') == {"x": 1}
    # Dict fields: invalid JSON string
    assert ds._parse_field("parameters_passed", "{invalid}") == {}
    # Dict fields: already a dict
    assert ds._parse_field("parameters_passed", {"y": 2}) == {"y": 2}
    # Dict fields: wrong type
    assert ds._parse_field("parameters_passed", 123) == {}
    # Non-list/dict field
    assert ds._parse_field("agent_name", "abc") == "abc"


@pytest.mark.unit
def test_parse_field_list_string_no_brackets():
    ds = AgentDataset()
    # Should return [] for list field if string does not start/end with brackets
    assert ds._parse_field("trace", "notalist") == []


@pytest.mark.unit
def test_parse_field_dict_string_no_braces():
    ds = AgentDataset()
    # Should return {} for dict field if string does not start/end with braces
    assert ds._parse_field("parameters_passed", "notadict") == {}


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
    # List/dict fields with invalid JSON
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
    assert agent.trace == []
    assert agent.tools_available == []
    assert agent.tool_calls == []
    assert agent.parameters_passed == {}
    assert agent.tool_call_results == []


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
    # File should not exist or be empty
    assert not file_path.exists() or file_path.read_text() == ""


@pytest.mark.unit
def test_parse_field_invalid_json_list():
    ds = AgentDataset()
    ds._list_fields.add("trace")
    # Invalid JSON string for list
    assert ds._parse_field("trace", "[notjson]") == []


@pytest.mark.unit
def test_parse_field_invalid_json_dict():
    ds = AgentDataset()
    ds._dict_fields.add("parameters_passed")
    # Invalid JSON string for dict
    assert ds._parse_field("parameters_passed", "{notjson}") == {}


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
    # As dict
    val = ds._parse_field(
        "expected_tool_call", {"tool_name": "t", "parameters": {}, "call_id": "c"}
    )
    assert val.tool_name == "t"
    # As JSON string
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
    assert ds._parse_field("trace", "[invalid]") == []
    assert ds._parse_field("trace", '[{"a":1}]') == [{"a": 1}]
    # Dict field
    assert ds._parse_field("parameters_passed", "{}") == {}
    assert ds._parse_field("parameters_passed", "{invalid}") == {}
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
    import novaeval.agents.agent_dataset as agent_dataset_mod

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
    # Should return None if value is not str or dict for ToolCall
    assert ds._parse_field("expected_tool_call", 123) is None
    assert ds._parse_field("expected_tool_call", [1, 2, 3]) is None


@pytest.mark.unit
def test_parse_field_list_field_typeerror(monkeypatch):
    ds = AgentDataset()
    # Patch json.loads to raise TypeError
    monkeypatch.setattr(
        json, "loads", lambda v: (_ for _ in ()).throw(TypeError("fail"))
    )
    assert ds._parse_field("trace", "[1,2,3]") == []


@pytest.mark.unit
def test_parse_field_dict_field_typeerror(monkeypatch):
    ds = AgentDataset()
    # Patch json.loads to raise TypeError
    monkeypatch.setattr(
        json, "loads", lambda v: (_ for _ in ()).throw(TypeError("fail"))
    )
    assert ds._parse_field("parameters_passed", '{"x":1}') == {}


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


def test_stream_from_json_file_error():
    """Test stream_from_json with file read error."""
    ds = AgentDataset()

    with pytest.raises(ValueError, match="Error reading JSON file"):
        list(ds.stream_from_json("/nonexistent/file.json"))


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

    # Test unrecognized strings default to False
    assert ds._parse_field("agent_exit", "maybe") is False
    assert ds._parse_field("agent_exit", "unknown") is False


def test_parse_field_bool_numeric_conversions():
    """Test boolean field parsing with numeric inputs."""
    ds = AgentDataset()

    # Test numeric conversions
    assert ds._parse_field("agent_exit", 1) is True
    assert ds._parse_field("agent_exit", 0) is False
    assert ds._parse_field("agent_exit", 42) is True  # Non-zero is True
    assert ds._parse_field("agent_exit", -1) is True  # Non-zero is True

    # Test invalid numeric conversions
    assert ds._parse_field("agent_exit", "not_a_number") is False


def test_parse_field_bool_exception_handling():
    """Test boolean field parsing exception handling."""
    ds = AgentDataset()

    # Test with objects that can't be converted to bool
    class UnconvertibleObj:
        def __bool__(self):
            raise ValueError("Cannot convert to bool")

    obj = UnconvertibleObj()
    result = ds._parse_field("agent_exit", obj)
    assert result is False  # Should default to False on exception


def test_parse_field_bool_none_with_default():
    """Test boolean field parsing with None value and field defaults."""
    ds = AgentDataset()

    # agent_exit field should have a default value
    result = ds._parse_field("agent_exit", None)
    assert result is False  # Should use default


def test_parse_field_expected_tool_call_invalid_json():
    """Test parsing expected_tool_call with invalid JSON string."""
    ds = AgentDataset()

    # Test with invalid JSON
    result = ds._parse_field("expected_tool_call", "invalid json {")
    assert result is None

    # Test with valid JSON but invalid ToolCall structure
    result = ds._parse_field("expected_tool_call", '{"invalid": "structure"}')
    assert result is None


def test_parse_field_expected_tool_call_exception():
    """Test parsing expected_tool_call with exception during ToolCall creation."""
    ds = AgentDataset()

    # Test with JSON that causes ToolCall constructor to fail
    invalid_toolcall_json = '{"tool_name": null, "parameters": "not_a_dict"}'
    result = ds._parse_field("expected_tool_call", invalid_toolcall_json)
    assert result is None


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
        assert ds.data[2].agent_exit is False  # "maybe" should convert to False
        assert ds.data[2].trace is None or ds.data[2].trace == []
        assert ds.data[2].expected_tool_call is None

    finally:
        os.unlink(temp_file_path)


def test_parse_field_tool_call_instance():
    """Test parsing when value is already a ToolCall instance."""
    ds = AgentDataset()
    existing_toolcall = ToolCall(tool_name="test", parameters={}, call_id="123")
    result = ds._parse_field("expected_tool_call", existing_toolcall)
    assert result == existing_toolcall


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


def test_ingest_from_csv_pandas_exception():
    """Test ingest_from_csv with pandas-related exception."""
    ds = AgentDataset()

    # Test with a nonexistent file to trigger exception handling
    with pytest.raises(ValueError, match="Error reading CSV file"):
        ds.ingest_from_csv("/nonexistent/file/path.csv")
