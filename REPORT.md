# Project Report: AgriGemma4.9ja

**Developer:** Victor Chukwuebuka Nwaruwe  
**Team ID:** team-agrigemma49ja  
**Track:** Agriculture  
**Claims:** Budget Laptop + African Alpha

---

## 1. Executive Summary

AgriGemma4.9ja is an offline, localized Retrieval-Augmented Generation (RAG) system built for Nigerian smallholder farmers. It runs entirely on-device on an 8 GB RAM laptop, requiring no internet once the model is downloaded.

By pairing a quantized Gemma 4 E2B IT model with offline NAERLS agricultural records, the system delivers practical crop, livestock, soil, pest, and weather advice in English and Nigerian Pidgin. It also handles greetings gracefully in Igbo, Yoruba, Hausa, and Pidgin.

---

## 2. Model Selection and Design Alternatives

Multiple models were evaluated against the ADTC Standard Laptop constraint.

### Candidate Models Tested

| Model | Quantization | Approx Size | Accuracy (arc_easy) | TPS | Notes |
|-------|-------------|-------------|---------------------|-----|-------|
| Llama 3.2 3B | Q4_K_M | 3.43 GB | 72% | lower | First model tested |
| Gemma 4 E2B IT | Q4_K_M | ~4.3 GB | 74% | moderate | Good but heavier |
| Gemma 4 E2B IT | Q3_K_M | 2.4 GB | 74% | 12.4 | Best balance |
| Aya | — | — | failed | — | Recommended but failed |
| TinyLlama | — | — | hallucinated | — | Not viable |

### Why Gemma 4 E2B IT Q3_K_M Won

1. Same accuracy as Q4 but smaller and faster. Q3_K_M kept 74% accuracy while reducing model size to 2.4 GB and improving throughput to 12.4 TPS.

2. Strong instruction following. Gemma 4 E2B IT follows structured agricultural prompts better than smaller models, which is critical when combining RAG evidence with system rules.

3. Multilingual capability. The model supports English and Nigerian Pidgin well enough for practical advisory, and recognizes common expressions in other Nigerian languages.

4. RAG compatibility. The model context window and instruction sensitivity made it suitable for injecting retrieved NAERLS evidence directly into the prompt.

5. Failure of smaller alternatives. Aya and TinyLlama did not work well in practice, showing that theoretical recommendations do not always survive real testing.

---

## 3. Verified Performance Telemetry

Measured with the official adtc-profiler on Ubuntu 22.04.5 LTS, Intel Core i5-8365U, 7.6 GB RAM available.

| Metric | Value |
|--------|-------|
| Generation throughput | 12.4 tokens/second |
| Peak RAM | 3,011.63 MB |
| First token latency | 20,120.63 ms |
| Accuracy (arc_easy, 50 samples) | 0.74 |
| CPU p99 | ~51.7% |
| Thermal throttling | None detected |

**Model details:**

- Architecture: Gemma 4 E2B IT
- Quantization: GGUF Q3_K_M
- Parameter count: 4,647,450,147 (~4.65B)
- Model file size: 2.4 GB

---

## 4. Hardware and Software Optimization

### Runtime Configuration

- llama.cpp CPU-only inference with n_threads=4
- n_batch=512 for efficient prompt processing
- n_ctx=4096 to balance context and memory
- OpenMP thread variables explicitly set to 4

### Memory Management

- Q3_K_M quantization keeps the model at 2.4 GB on disk
- Peak runtime memory of 3.01 GB leaves over 4 GB headroom
- No memory-mapped swapping or OOM risk during evaluation

### Inference Stability

- Temperature locked at 0.20 for deterministic, factual responses
- repeat_penalty=1.12 to reduce looping
- top_p=0.9 for controlled diversity

### Guard Rails

A pre-inference guard layer:

- Recognizes greetings in English, Pidgin, Igbo, Yoruba, Hausa
- Detects non-agricultural topics and politely redirects
- Prevents hallucinations on out-of-scope queries
- Asks for clarification on very short or vague messages

---

## 5. RAG System Design

### 1. NAERLS Verified Corpus

- File: naerls_verified.csv
- Records: 671
- Derived from official NAERLS extension documents covering maize, cassava, rice, livestock, aquaculture, and more.

### 2. Concise Master Knowledge Base

- File: master_agro_kb.json
- Entries: 78
- Written as direct farmer-friendly answers
- Covers crops, livestock, soil, weather, post-harvest, marketing, and local language aliases
- Includes 20 Pidgin-specific entries

### Retrieval Mechanism

- Inverted-index keyword search
- Query expansion using a Pidgin term dictionary
- Curated JSON entries are prioritized when they match
- Scoring threshold prevents irrelevant matches

---

## 6. Domain Application Validation

### Test Prompt 1 — Pest Identification

A maize farmer in Kaduna reports sawdust-like frass in the whorl and leaf damage. What is the likely pest and what practical first response should the farmer take?

Result: Correctly identified Fall Armyworm risk and recommended early-stage control using neem or approved insecticide.

### Test Prompt 2 — Pidgin Disease Query

Abeg, my cassava leaf dey yellow and curl. Wetin fit cause am, and wetin I fit do first make e no spread?

Result: Correctly identified Cassava Mosaic Disease (CMD), recommended removing and burning infected plants, and advised planting TME 419 or NR 8082 resistant varieties.

---

## 7. Constraints Addressed

| Constraint | Solution |
|------------|----------|
| 8 GB RAM limit | Q3_K_M quantization |
| No internet during inference | Fully offline RAG |
| CPU-only | Physical core binding |
| Language barrier | English + Pidgin |
| Thermal penalty risk | Low RAM usage |

---

## 8. Repository Structure

agrigemma4.9ja/
- app.py
- master_agro_kb.json
- naerls_verified.csv
- naerls_sources/
- metadata.json
- download_model.sh
- README.md
- REPORT.md
- submission.json
- eval_suite.py
- tools/

---

## 9. Limitations and Future Work

- Model accuracy degrades on complex multi-step reasoning under Q3_K_M
- Pidgin support is practical but not as fluent as English
- Retrieval uses keyword matching, not semantic embeddings
- Future: add vector retrieval, expand Pidgin corpus, fine-tune on NAERLS data
