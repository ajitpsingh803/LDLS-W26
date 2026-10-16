"""Llama-3 chat framing shared by every domain.

The prefix is stored once and belongs to every sequence. The document's own tokens follow it.
A question is appended at the end of ONE sequence with QUESTION_TEMPLATE.
"""

SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the question using only the document. "
    "Reply with a short, direct answer."
)

PREFIX = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"{SYSTEM_PROMPT}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "<document>\n"
)

QUESTION_TEMPLATE = (
    "\n</document>\n\n"
    "Question: {question}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)
