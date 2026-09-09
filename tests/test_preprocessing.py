import pytest
from src.preprocessing import clean_text


def test_clean_text_happy_path():
    raw_text = "The product works AMAZINGLY well! I highly recommend buying it."
    cleaned = clean_text(raw_text)
    assert isinstance(cleaned, str)
    assert "product" in cleaned
    assert "work" in cleaned or "well" in cleaned
    assert "recommend" in cleaned
    assert "!" not in cleaned


def test_clean_text_html_tags():
    raw_text = "<p>This is a <b>great</b> item!</p><br />Loved it."
    cleaned = clean_text(raw_text)
    assert "<p>" not in cleaned
    assert "<br />" not in cleaned
    assert "great" in cleaned or "loved" in cleaned


def test_clean_text_urls_and_mentions():
    raw_text = "Check this out @john_doe at http://example.com/item! Awesome experience."
    cleaned = clean_text(raw_text)
    assert "@john_doe" not in cleaned
    assert "http" not in cleaned
    assert "example.com" not in cleaned
    assert "awesome" in cleaned or "experience" in cleaned


def test_clean_text_empty_input():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""


def test_clean_text_punctuation_only():
    raw_text = "!@#$%^&*()_+-=[]{}|;:'\",.<>/?"
    assert clean_text(raw_text) == ""


def test_clean_text_negation_contractions():
    raw_text = "I don't like this, it isn't working and I can't recommend it."
    cleaned = clean_text(raw_text)
    assert "not" in cleaned or "cannot" in cleaned
    assert "like" in cleaned or "work" in cleaned

