from pathlib import Path

from src.text_classifier.data import get_label_mapping, encode_labels


def test_get_label_mapping():
    labels = ["positive", "negative", "neutral"]
    mapping = get_label_mapping(labels)

    assert mapping["positive"] == 0
    assert mapping["negative"] == 1
    assert mapping["neutral"] == 2
    assert len(mapping) == 3


def test_encode_labels():
    labels = ["positive", "negative", "neutral"]
    mapping = get_label_mapping(labels)

    encoded = encode_labels(labels, mapping)
    assert encoded == [0, 1, 2]


def test_encode_labels_with_duplicates():
    labels = ["positive", "negative", "positive", "negative"]
    mapping = get_label_mapping(labels)

    encoded = encode_labels(labels, mapping)
    assert encoded == [0, 1, 0, 1]
