import csv
import json

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
    # All other fields should be default or None
    for k in AgentData.model_fields:
        if k not in [
            "agent_name",
            "agent_role",
            "tools_available",
            "tool_calls",
            "parameters_passed",
        ]:
            val = getattr(ds.data[0], k)
            if isinstance(val, list) and val and hasattr(val[0], "model_dump"):
                assert [x.model_dump() for x in val] == []
            elif hasattr(val, "model_dump"):
                assert val.model_dump() == {}
            else:
                assert val is None or val == [] or val == {}


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
    for k in AgentData.model_fields:
        if k not in [
            "agent_name",
            "agent_role",
            "tools_available",
            "tool_calls",
            "parameters_passed",
        ]:
            val = getattr(ds.data[0], k)
            if isinstance(val, list) and val and hasattr(val[0], "model_dump"):
                assert [x.model_dump() for x in val] == []
            elif hasattr(val, "model_dump"):
                assert val.model_dump() == {}
            else:
                assert val is None or val == [] or val == {}


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
