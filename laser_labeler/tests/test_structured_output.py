import json
from pathlib import Path

import pytest
from PIL import Image

from config.settings import load_settings
from core.labeler.label_image import label_image
from core.labeler.schemas import pipeline_schema
from core.models import INCIDENCES, LASER_KINDS
from core.providers.base import Message
from core.providers.openai_compat import OpenAICompatProvider
from core.providers.registry import build_provider
from core.schema.taxonomy import load_taxonomy
from tests.fakes import PEN_RUBBER, ScriptedProvider, pipeline_json, text
from tests.synth import finish, rect, scene, spot
from tests.test_providers import SPEC, oa_client, oa_reply

jsonschema = pytest.importorskip("jsonschema")
TAX = load_taxonomy(Path(__file__).resolve().parent.parent / "taxonomy")
SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


def sent(provider_kwargs, schema=SCHEMA, tools=None):
    client, s = oa_client(oa_reply())
    OpenAICompatProvider("m", client=client, **provider_kwargs).chat([Message("user", "q")], tools, schema)
    return s.calls[0]


def test_response_format_sent_only_when_enabled_and_asked():
    kw = sent({"structured_output": True})
    assert kw["response_format"] == {"type": "json_schema", "json_schema": {"name": "reply", "schema": SCHEMA, "strict": True}}
    assert "response_format" not in sent({})                              # flag off
    assert "response_format" not in sent({"structured_output": True}, schema=None)
    assert "response_format" not in sent({"structured_output": True}, tools=[SPEC])  # tools win


def test_registry_passes_the_flag():
    block = {"provider": "openai_compat", "model": "m", "base_url": "http://x/v1", "structured_output": True}
    assert build_provider(block, None).structured_output is True
    assert build_provider({**block, "structured_output": False}, None).structured_output is False


def test_pipeline_schema_accepts_good_replies_and_rejects_bad_ones():
    schema = pipeline_schema(["c1", "c2"], TAX)
    jsonschema.Draft202012Validator.check_schema(schema)
    obj = {"general_object": "pen", "sub_object": "ballpoint", "material": "rubber", "finish": None,
           "incidence": "first", "confidence": 0.8}
    good = {"status": "ok", "questions": [], "candidates": [
        {"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0, "from_id": None, "object": obj, "notes": ""},
        {"id": "c2", "is_laser": False, "kind": None, "order": None, "from_id": None, "object": None, "notes": "glare"}]}
    jsonschema.validate(good, schema)
    for mutate in (lambda d: d["candidates"][0]["object"].update(material="unobtainium"),
                   lambda d: d["candidates"][0].update(id="c9"),
                   lambda d: d["candidates"][0].update(kind="zap"),
                   lambda d: d["candidates"][0]["object"].update(incidence="sideways"),
                   lambda d: d["candidates"][0].pop("notes")):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)
    jsonschema.validate(json.loads(json.dumps(good)) | {"candidates": [
        {**good["candidates"][0], "object": {**obj, "material": "unknown"}}, good["candidates"][1]]}, schema)


def test_schema_follows_taxonomy_and_kinds():
    schema = pipeline_schema(["c1"], TAX)
    item = schema["properties"]["candidates"]["items"]["properties"]
    surface = item["object"]["anyOf"][1]["properties"]
    assert set(TAX.materials) < set(surface["material"]["enum"]) and "unknown" in surface["material"]["enum"]
    assert set(surface["incidence"]["enum"]) == set(INCIDENCES)
    assert set(LASER_KINDS) < set(item["kind"]["enum"])


def test_pipeline_sends_schema_only_when_setting_is_on(tmp_path):
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 140, 110, sigma=2.5, amp=0.5, color=(1, 1, 1))
    path = tmp_path / "a.png"
    Image.fromarray(finish(img)).save(path)
    reply = text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0, "object": PEN_RUBBER}]))
    seen = {}
    for flag in (False, True):
        settings = load_settings()
        settings["provider"]["structured_output"] = flag
        provider = ScriptedProvider([reply])
        label_image(path, provider=provider, taxonomy=TAX, settings=settings, data_dir=tmp_path / f"d{flag}", mode="pipeline")
        seen[flag] = provider.schemas
    assert seen[False] == [None]
    assert seen[True][0]["required"] == ["status", "candidates", "questions"]
