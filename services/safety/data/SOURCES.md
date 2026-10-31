# Safety training data: sources and licenses

`intent.jsonl` holds 362 labeled queries for the query intent check (feature 3, Section C #24) and its fallback classifier (`models/intent.json`). Each row is `{"text", "label", "source", "license"}`, so you can trace every example back to where it came from.

The labels are the app's five intent classes: `news_question`, `off_topic`, `harmful_request`, `system_probe`, `jailbreak`. The public datasets were **sampled** (a fixed seed) and **relabeled** into these classes. The original labels and categories are kept in each row's `source`, for example `XSTest (safe, homonyms)`.

| Source | License | Rows | Used as | Home |
|---|---|---|---|---|
| HarmBench, text behaviors ("standard") | MIT | 40 | `harmful_request` | https://github.com/centerforaisafety/HarmBench |
| JailbreakBench JBB-Behaviors (harmful / benign) | MIT | 19 / 20 | `harmful_request` / `off_topic` | https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors |
| MultiJail (it, jv, ko, sw, th, vi, zh) | MIT | 15 | `harmful_request` | https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail |
| XSTest (safe / unsafe contrast prompts) | CC-BY-4.0 | 30 / 20 | `off_topic` / `harmful_request` | https://github.com/paul-rottger/xstest |
| OR-Bench hard-1k | CC-BY-4.0 | 20 | `off_topic` | https://huggingface.co/datasets/bench-llm/or-bench |
| jackhhao/jailbreak-classification | Apache-2.0 | 25 | `jailbreak` | https://huggingface.co/datasets/jackhhao/jailbreak-classification |
| deepset/prompt-injections (label 1) | Apache-2.0 | 25 | `jailbreak` | https://huggingface.co/datasets/deepset/prompt-injections |
| This project: past queries, attack test (#7c), queries written for feature 3 (40 of them in Section B: news asked as questions) | ours | 148 | all five classes | — |

**Attribution (CC-BY-4.0):** XSTest by Röttger et al., "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models" (2024). OR-Bench by Cui et al., "OR-Bench: An Over-Refusal Benchmark for Large Language Models" (2024). Changes made: a sample of their prompts, relabeled into the five classes above.

**Why harmful prompts are in the repo:** a classifier can only learn to refuse harmful requests from examples of them. They come from published safety benchmarks built for exactly this use. The app never answers them: they are only embedded and used to train the classifier. This was decided in feature 3, A3 §2.3 (2026-10-31).

**Not here, on purpose:** the labeled article chunks for the injection check. They are real news articles. That's fine for local testing, but the text isn't ours to publish, so they stay in `.local/safety_data/curated/chunks.jsonl`. A classifier trained on them was measured in Section B but isn't shipped: it caught too few planted instructions.
