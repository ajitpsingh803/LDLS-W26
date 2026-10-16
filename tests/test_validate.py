"""The validator must accept generated documents and reject the mistakes it is meant to catch.

Needs the model and build/bin/ldls_tokenize (skipped otherwise)."""

from __future__ import annotations

import copy

import pytest

from ldls import DEFAULT_MODEL, DEFAULT_TOKENIZER_BIN, REPO_ROOT
from ldls.validate import ValidationError, validate_doc

pytestmark = pytest.mark.skipif(not (DEFAULT_MODEL.exists() and DEFAULT_TOKENIZER_BIN.exists()),
                                reason="model or tokenizer binary missing")

SRC = (REPO_ROOT / "data" / "markdown" / "01_research_paper.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def doc():
    from ldls.build_doc import build_document
    from ldls.tokenizer import LlamaCppTokenizer
    return build_document(SRC, "markdown", LlamaCppTokenizer(), "01_research_paper")


def first_with_tokens(d):
    return next(n for n in d["nodes"] if len(n["tokens"]) > 3)


def test_generated_document_is_valid(doc):
    assert validate_doc(doc, SRC) == []


def test_rejects_position_gap(doc):
    d = copy.deepcopy(doc)
    n = first_with_tokens(d)
    n["positions"] = [p + 1 for p in n["positions"]]
    with pytest.raises(ValidationError):
        validate_doc(d)


def test_rejects_duplicate_position_in_a_sequence(doc):
    # the mistake of the first draft schema: a child restarting positions at 0 inside its parent's sequence
    d = copy.deepcopy(doc)
    n = first_with_tokens(d)
    n["positions"] = list(range(len(n["tokens"])))
    with pytest.raises(ValidationError):
        validate_doc(d)


def test_rejects_seq_ids_that_disagree_with_unit(doc):
    d = copy.deepcopy(doc)
    n = next(x for x in d["nodes"] if len(x["seq_ids"]) == 1 and x["tokens"])
    n["seq_ids"] = [0] if n["seq_ids"] != [0] else [1]
    with pytest.raises(ValidationError):
        validate_doc(d)


def test_rejects_text_that_does_not_reconstruct_source(doc):
    d = copy.deepcopy(doc)
    first_with_tokens(d)["text"] += "x"
    with pytest.raises(ValidationError):
        validate_doc(d, SRC)


def test_rejects_too_many_sequences(doc):
    d = copy.deepcopy(doc)
    d["layout"]["merge_rules"]["max_sequences"] = 2
    with pytest.raises(ValidationError):
        validate_doc(d)


def test_retokenization_matches(doc):
    from ldls.tokenizer import LlamaCppTokenizer
    validate_doc(doc, SRC, LlamaCppTokenizer())
