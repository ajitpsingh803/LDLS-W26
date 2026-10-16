# LDLS Assignment 0

Ajit Pratap Singh. Original implementation credited to Rupinder Kaur and Abhishek.

We tested whether the answer to a question about one section stays the same after another section is edited or removed. The input was five Markdown documents. Both main tests passed on all five. The city guide had a small difference in an additional test: the answer gained a full stop. The independent batched prompts also differed from the separate runs.

## Scope

We used Markdown as our domain. The five sample documents are synthetic and contain 464 to 790 model tokens each, so all are below the assignment's 2,000-token limit. The assignment allows one domain or a pair of domains, so one Markdown parser is enough for this team's part.

The supplied HTML and LaTeX examples use one driver for both parsers. One version also combines five documents per domain into a corpus. Their JSON contains text segments and estimated positions. Our Markdown JSON contains actual model token IDs and explicit positions, so the files cannot be passed directly between the two drivers. They would need an adapter or an agreed schema. We read these examples but did not rerun them.

## Results from the local run

The run used WSL2 Ubuntu, an Intel Xeon E5-2620 v4 and eight CPU threads. The model was Llama-3.2-1B-Instruct Q4_K_M with greedy decoding. We rebuilt llama.cpp and the driver at commit 318a471 using the model already on the machine. CMake 4.2.3 and g++ 15.2.0 were available. All 28 unit tests passed, and the five parsed documents passed validation. The full commit and model checksum are in results/environment.txt.

- T1 (edit B, ask A): 5/5 pass.
- T2 (evict C, ask A): 5/5 pass; 48–129 occupied KV cells freed.
- Positive control (edit A): 5/5 pass.
- Optional keep-only-A-path comparison: 4/5 pass. City guide returns `$12.` rather than `$12`.
- Independent batched prompts versus separate runs: 0/2 exact matches, at steps 3 and 24.
- Shared-prefix comparison: 2/2 exact matches; 56 cells instead of 72.
- Same-shape neighbour perturbation: zero logit difference in the observed sequence.

Prefill took about the same time in both modes: 4.95 s for structured attention and 4.92 s for flat attention. The difference showed up after editing B. Patching the affected sequences took 484 ms, while a full re-prefill took 5.00 s. Removing C took about 50 microseconds. These are medians across the five documents from one run. We did not measure variation over repeated trials. The extra logit tracing was disabled for these timings.

Removing C freed between 48 and 129 occupied KV cells. Each cell holds 32 KiB here, giving 1.5 to 4.0 MiB of space that the cache can reuse. This measures occupied cells; it does not show a drop in process RAM or the allocated cache buffer. Keeping only A's path left between 157 and 275 cells. That extra test passed on four documents and failed on the city guide.

Full measurements: [results/summary.md](results/summary.md). Report: [report/Writeup.pdf](report/Writeup.pdf).
One-slide report: [report/slide.pptx](report/slide.pptx), [PDF](report/slide.pdf), [PNG](report/slide.png).

## Run in WSL Ubuntu

The local working copy is `/home/t/ldls`. Its existing model and virtual environment were reused.
From PowerShell, open Ubuntu using `wsl -d Ubuntu`, then:

```bash
cd ~/ldls
source .venv/bin/activate
python -m pytest -q
python -m ldls.tokenizer "Hello Markdown"
```

For a fresh copy of this submitted repo, run these commands from its root directory:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/setup_llama_cpp.sh
bash scripts/download_model.sh     # skips an existing model
bash scripts/build.sh
python -m ldls.build_doc --domain markdown data/markdown/*.md --out-dir build/docs
python -m ldls.validate build/docs/*.json --source-dir data/markdown --retokenize
python -m pytest -q
bash scripts/run_foundation.sh
python tests/run_doc_tests.py --domain markdown --threads 8
python report/make_report.py
```

The Python report script regenerates summary JSON and Markdown. The supplied PowerPoint is editable;
update it if you rerun with different measurements. Build artifacts, `.venv`, the 771 MiB model and
llama.cpp sources are excluded from the submission. Scripts fetch the pinned source and model.
`results/requirements-resolved.txt` records the installed Python package versions.

## Tokenizer import fix

The tokenizer originally used a relative import and had no CLI. Starting it as `python3 tokenizer.py`
inside `ldls/` lost its package context. It now supports both module and file execution:

```bash
# Preferred, from repository root
python -m ldls.tokenizer "Hello Markdown"
# Also supported
python ldls/tokenizer.py "Hello Markdown"
# From inside ldls/: python tokenizer.py "Hello Markdown"
```

Both tested commands returned `[9906, 74292]`. To parse documents, use `python -m ldls.build_doc`;
the tokenizer CLI only prints model token IDs.

## City guide failure

The city question asks how much a Maritime Museum ticket costs. The normal answer is "$12". With only A's path left in the cache, the answer becomes "$12.". The price is still correct, but the extra full stop makes the exact-token test fail. The remaining cache size is correct at 166 cells. The first two generated tokens match. At step 2, the normal run prefers stopping over a full stop by 0.208 logits. In the trimmed-cache run, the full stop wins by 0.036.

The model is close to a tie at this step, so a small change in the logits is enough to change the output. Floating-point differences after the cache layout changes are a possible cause. We have not isolated the exact operation responsible. The main T1 and T2 tests still pass for this document. We left the extra test marked as failed because it requires identical tokens.

Reproduce the saved diagnostic separately from timing runs:

```bash
python tests/run_doc_tests.py --specs tests/specs/markdown/05_city_guide.json \
  --trace-generation --results-dir results/diagnostics --strict
python tests/run_batched_test.py --strict
```

Both strict commands return a nonzero exit code on this machine. Default test-runner exit codes cover
the mechanism checks; they are not a claim that every exact-match comparison passes. The independent
prompt requirement remains unmet. See [docs/city_diagnosis.md](docs/city_diagnosis.md).

## Implementation

The parser reads headings, paragraphs, lists and code blocks using markdown-it-py. It writes a JSON tree containing tokens, positions, seq_ids, node_type and parent. The token IDs come from the model's own llama.cpp tokenizer. Small sections are merged using a 40-token threshold, with at most 32 sequences by default. Each leaf unit gets a sequence ID. Its sequence includes the tokens on the path from the root to that unit. Sibling sections can reuse position numbers because they belong to different sequences.

The C++ driver reads the JSON and fills the KV cache using these positions and sequence IDs. The attention mask restricts the question to its sequence and earlier positions. Tokens from parent sections can be shared across sequences, which requires a unified KV cache. The code trace is in docs/llama_cpp_trace.md. It covers llama.h, the batched example and the mask construction.

- `schema/`: schema, merge rules and a small worked example.
- `ldls/parsers/markdown.py`: Markdown parser.
- `ldls/layout.py`: sequence assignment and positions.
- `driver/src/`: shared C++ driver and exact tokenizer.
- `data/markdown/`: five source documents.
- `parsed_docs/`: model-tokenized outputs for all five documents.
- `tests/`: unit tests, specifications, experiment runners.
- `docs/llama_cpp_trace.md`: API, batched example, mask and eviction trace.
- `results/`: raw measurements, logs, diagnostics and environment record.
- `report/`: write-up and one-slide report.

The results cover five small synthetic documents. A question can read one root-to-leaf path, so a question needing information from two sibling sections would need a different arrangement. Prefill still processes every token, which explains why it was not faster. In this run, the useful difference was the lower cost of updating one section and the ability to reuse freed KV cells.

## Credits

Original implementation: Rupinder Kaur and Abhishek, as credited in the supplied project. Submission and local reproduction: Ajit Pratap Singh. The project uses llama.cpp, markdown-it-py and Meta Llama 3.2, with the Q4_K_M GGUF from bartowski.

## Assignment checklist

See [SUBMISSION_NOTES.md](SUBMISSION_NOTES.md) for the requirement-to-file mapping and open items.
