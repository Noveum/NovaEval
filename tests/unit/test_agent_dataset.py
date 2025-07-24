import csv
import json

import pytest

from novaeval.agents.agent_data import AgentData
from novaeval.agents.agent_dataset import AgentDataset


def minimal_agent_data_dict():
    return {
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
            ]
        },
        "trace": json.dumps(d["trace"]),
        "tools_available": json.dumps(d["tools_available"]),
        "tool_calls": json.dumps(d["tool_calls"]),
        "parameters_passed": json.dumps(d["parameters_passed"]),
        "tool_call_results": json.dumps(d["tool_call_results"]),
    }


def assert_agentdata_equal(actual, expected):
    for k, v in expected.items():
        actual_val = getattr(actual, k)
        if (
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
        ]:
            val = getattr(agent, k)
            if isinstance(val, list) and val and hasattr(val[0], "model_dump"):
                assert [x.model_dump() for x in val] == []
            elif hasattr(val, "model_dump"):
                assert val.model_dump() == {}
            else:
                assert val is None or val == [] or val == {}


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
            else:
                assert rows[0][k] == v


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


def test_export_to_csv_empty(tmp_path):
    ds = AgentDataset()
    export_file = tmp_path / "empty.csv"
    ds.export_to_csv(str(export_file))  # Should not raise
    assert not export_file.exists() or export_file.read_text() == ""


def test_export_to_json_empty(tmp_path):
    ds = AgentDataset()
    export_file = tmp_path / "empty.json"
    ds.export_to_json(str(export_file))
    with open(export_file, encoding="utf-8") as f:
        data = json.load(f)
        assert data == []


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


def test_parse_field_list_string_no_brackets():
    ds = AgentDataset()
    # Should return [] for list field if string does not start/end with brackets
    assert ds._parse_field("trace", "notalist") == []


def test_parse_field_dict_string_no_braces():
    ds = AgentDataset()
    # Should return {} for dict field if string does not start/end with braces
    assert ds._parse_field("parameters_passed", "notadict") == {}


def test_parse_field_list_field_non_list_non_str():
    ds = AgentDataset()
    # Should return [] for list field if value is not a list or str
    assert ds._parse_field("trace", 42) == []


def test_parse_field_dict_field_non_dict_non_str():
    ds = AgentDataset()
    # Should return {} for dict field if value is not a dict or str
    assert ds._parse_field("parameters_passed", 42) == {}


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


def test_get_datapoint_empty():
    ds = AgentDataset()
    assert list(ds.get_datapoint()) == []


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


def test_parse_field_returns_value_for_non_listdict_field():
    ds = AgentDataset()
    # Add a dummy field not in _list_fields or _dict_fields
    val = ds._parse_field("not_special", 123)
    assert val == 123
    val = ds._parse_field("not_special", "abc")
    assert val == "abc"


def test_export_to_csv_empty_file(tmp_path):
    ds = AgentDataset()
    file_path = tmp_path / "empty.csv"
    ds.export_to_csv(str(file_path))
    # File should not exist or be empty
    assert not file_path.exists() or file_path.read_text() == ""


def test_parse_field_invalid_json_list():
    ds = AgentDataset()
    ds._list_fields.add("trace")
    # Invalid JSON string for list
    assert ds._parse_field("trace", "[notjson]") == []


def test_parse_field_invalid_json_dict():
    ds = AgentDataset()
    ds._dict_fields.add("parameters_passed")
    # Invalid JSON string for dict
    assert ds._parse_field("parameters_passed", "{notjson}") == {}


def test_parse_field_list_field_non_str_non_list():
    ds = AgentDataset()
    ds._list_fields.add("trace")
    # Not a str or list
    assert ds._parse_field("trace", 42) == []


def test_parse_field_dict_field_non_str_non_dict():
    ds = AgentDataset()
    ds._dict_fields.add("parameters_passed")
    # Not a str or dict
    assert ds._parse_field("parameters_passed", 42) == {}


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
