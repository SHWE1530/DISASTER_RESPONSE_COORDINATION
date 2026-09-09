"""Unit tests for Stage 03 NLP Pipeline."""

import sys
from pathlib import Path
import pytest
import json

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

import importlib.util
spec = importlib.util.spec_from_file_location("stage03_nlp_engineer", base_dir / "03_nlp_engineer.py")
nlp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nlp)


def test_model_loading_and_training():
    # Test 11: model loading & artifact initialization
    artifacts = nlp._get_artifacts()
    assert "urgency_model" in artifacts
    assert "hazard_model" in artifacts
    assert "ner_model" in artifacts


def test_urgency_classification():
    # Test 1: urgency classification
    res_high = nlp.predict_urgency("CRITICAL EMERGENCY! 20 people trapped in severe flooding. Send rescue boat immediately!")
    assert res_high in ["HIGH", "CRITICAL"]

    res_low = nlp.predict_urgency("Minor rainfall update for information only, normal conditions.")
    assert res_low in ["LOW", "MEDIUM"]


def test_hazard_classification():
    # Test 2: hazard classification
    res_flood = nlp.predict_hazard("Continuous heavy rainfall causing severe flood in low-lying area.")
    assert res_flood in ["Flood", "Heavy Rainfall", "Waterlogging"]

    res_fire = nlp.predict_hazard("Wildfire and forest fire spreading near rural forest areas.")
    assert res_fire in ["Forest Fire", "Wildfire", "Fire"]


def test_ner_extraction():
    # Test 3: NER extraction
    text = "Rescue requested near industrial area in Chennai. 8 people affected. Please send rescue boat."
    entities = nlp.extract_entities(text)
    assert "location" in entities
    assert "resource_needed" in entities
    assert "headcount" in entities


def test_analyze_text_schema():
    # Test 4: analyze_text output structure
    text = "Flooding reported near main bridge in Guwahati. 15 people affected. Send NDRF flood rescue team."
    res = nlp.analyze_text(text)
    assert res["text"] == text
    assert res["urgency_level"] in nlp.URGENCY_CLASSES
    assert 0.0 <= res["urgency_confidence"] <= 1.0
    assert isinstance(res["hazard_type"], str)
    assert 0.0 <= res["hazard_confidence"] <= 1.0
    assert "location" in res["entities"]
    assert "resource_needed" in res["entities"]
    assert "headcount" in res["entities"]


def test_process_batch():
    # Test 5: process_batch functionality
    batch = [
        "Road blockage reported near highway junction. Send tow truck.",
        "URGENT! 12 people trapped in landslide. Send ambulance and rescue team.",
    ]
    results = nlp.process_batch(batch)
    assert len(results) == 2
    assert results[0]["urgency_level"] in nlp.URGENCY_CLASSES
    assert results[1]["urgency_level"] in nlp.URGENCY_CLASSES


def test_empty_input():
    # Test 6: empty input robustness
    res = nlp.analyze_text("")
    assert res["urgency_level"] is None
    assert res["urgency_confidence"] == 0.0
    assert res["hazard_type"] is None
    assert res["hazard_confidence"] == 0.0
    assert res["entities"]["location"] is None
    assert res["entities"]["resource_needed"] == []
    assert res["entities"]["headcount"] is None


def test_whitespace_input():
    # Test 7: whitespace input robustness
    res = nlp.analyze_text("   \n\t   ")
    assert res["urgency_level"] is None
    assert res["urgency_confidence"] == 0.0
    assert res["hazard_type"] is None
    assert res["hazard_confidence"] == 0.0
    assert res["entities"]["location"] is None
    assert res["entities"]["resource_needed"] == []
    assert res["entities"]["headcount"] is None


def test_noisy_text():
    # Test 8: noisy social media text
    noisy = "URGENT!!! @user #flood https://example.com/help Water is rising in Mumbai! 5 people stranded!!!"
    res = nlp.analyze_text(noisy)
    assert res["urgency_level"] in ["MEDIUM", "HIGH", "CRITICAL"]


def test_multiple_resources():
    # Test 9: multiple resources extraction
    text = "Flooding in low-lying area. Response team requested: water pump and life jackets."
    res = nlp.extract_entities(text)
    assert len(res["resource_needed"]) >= 1


def test_headcount_extraction():
    # Test 10: headcount extraction accuracy
    text = "Rescue operation required. 26 people trapped/affected, urgent evacuation needed."
    res = nlp.extract_entities(text)
    assert res["headcount"] == 26


def test_lowercase_location_extraction():
    # Regression: social text often uses lowercase city names like 'mumbai'.
    text = "there was a flood and 1000 people were affected and it happened in mumbai"
    res = nlp.extract_entities(text)
    assert res["location"] is not None
    assert any(str(loc).lower() == "mumbai" for loc in (res["location"] if isinstance(res["location"], list) else [res["location"]]))


def test_social_text_prefers_real_place_over_hazard_words():
    # Fix: do not treat generic hazard words like 'flood' as the location slot.
    text = "there was a flood and 1000 people were affected in mumbai to the flood"
    res = nlp.extract_entities(text)
    locations = res["location"] if isinstance(res["location"], list) else [res["location"]]
    assert any(str(loc).lower() == "mumbai" for loc in locations)
    assert not any(str(loc).lower() == "flood" for loc in locations)


def test_bio_alignment_validation():
    # Test 12: BIO token/tag alignment validation
    tokens, tags, stats = nlp.load_bio_ner_datasets()
    assert stats["used_records"] > 0
    assert len(tokens) == len(tags)
    for seq_toks, seq_tags in zip(tokens[:100], tags[:100]):
        assert len(seq_toks) == len(seq_tags)


def test_stage03_integration_adapter():
    # Regression test for the app-level adapter contract expected by the existing Flask app.
    import importlib.util
    base_dir = Path(__file__).resolve().parent.parent
    adapter_path = base_dir / "05_integration_engineer.py"
    assert adapter_path.exists(), "Stage03 integration adapter file should exist"

    spec = importlib.util.spec_from_file_location("stage03_integration_engineer", adapter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    engine = module.NLPIntegrationEngine()
    result = engine.analyze("Three people are trapped in a flooded building near the railway bridge and need immediate rescue.")

    assert result["urgency"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert result["hazard_type"]
    assert result["location"] or result["entities"]["location"]
