# AgriGemma4.9ja

**Offline RAG‑powered agricultural advisory system for Nigerian smallholder farmers.**

AgriGemma4.9ja is a local‑first language model application built for the Africa Deep Tech Challenge 2026 (ADTC). It runs entirely on‑device using a quantized Gemma 4 E2B IT model and offline retrieval over Nigerian agricultural records (NAERLS). It provides practical crop, livestock, soil, pest, and weather advice in English and Nigerian Pidgin, with graceful greeting support for Igbo, Yoruba, and Hausa.

> **🏆 ADTC 2026 Submission**  
> **Team:** team-agrigemma49ja  
> **Track:** Agriculture  
> **Bonuses Claimed:** African Use Case (+10 pts) + African Language (+15% panel score)

---

## Problem

Smallholder farmers in Nigeria face:

- Limited access to extension officers – Nigeria has just **1 extension officer for every 5,000 to 10,000 farmers**, compared to the FAO recommendation of 1:400 to 1:800.
- High cost and poor connectivity for cloud‑based AI – rural farmers often lack reliable internet and electricity.
- Language barriers — most agricultural advice is in English, yet many farmers speak Pidgin or other local languages.
- Time‑sensitive pest and disease outbreaks – crop losses have **tripled to over 20%** of plots in the past five years.

Cloud‑hosted LLMs are not practical for rural farmers. AgriGemma4.9ja solves this by running **fully offline** on commodity hardware, requiring no internet connection after the initial model download.

---

## Hardware Target

Built for the ADTC Standard Laptop profile:

- Intel Core i5 10th–12th gen (developed on i5‑8365U)
- 8 GB RAM
- Integrated graphics only
- Ubuntu 22.04 LTS

Model inference is CPU‑only through llama.cpp, with no GPU required.

---

## Model Selection & Empirical Testing

We extensively tested multiple models on the ADTC Standard Laptop profile to find the best balance of speed, memory, accuracy, and African language support. All tests were run on the same hardware (Intel Core i5-8365U, 8 GB RAM, Ubuntu 22.04) to ensure fair comparison.

| Model | Quant | TPS | RAM | Accuracy | Sperf | Seff | Pidgin Support |
|-------|-------|-----|-----|----------|-------|------|----------------|
| **Gemma 4 E2B IT** | **Q3_K_M** | **12.4** | **3,011 MB** | **74%** | **82.67** | **56.97** | ✅ Native |
| Qwen2.5-1.5B | Q4_K_M | 13.06 | 1,695 MB | 76% | 87.07 | 75.79 | ⚠️ Via RAG |
| Phi-3.5-mini | Q4_K_M | 5.18 | 3,825 MB | 82% | 34.53 | 45.36 | ❌ No |
| Phi-3.5-mini | Q3_K_M | 6.08 | 3,218 MB | 78% | 40.53 | 54.03 | ❌ No |
| Llama 3.2 3B | Q3_K_M | 3.57 | 2,370 MB | 64% | 23.80 | 66.14 | ❌ No |

**Why Gemma 4 E2B IT Q3_K_M was selected:**

1. **Native Pidgin Support** – Essential for the African Language Bonus (+15% panel score). The model can understand and generate natural Pidgin responses, unlike other models tested.
2. **Best Balance** – 12.4 TPS, 3.0 GB RAM, and 74% accuracy is the optimal trade-off for ADTC scoring.
3. **Quantization-robust** – Retained full accuracy from Q4 to Q3 (unlike Llama 3.2, which dropped 8 points).
4. **Proven RAG Integration** – Fully tested with NAERLS corpus and Pidgin guardrails.
5. **Lower Implementation Risk** – Unlike Qwen, which would require additional work for Pidgin support, Gemma works out of the box.

---

## Performance

| Metric | Value |
|--------|-------|
| Generation speed | **12.4 tokens/sec** |
| Peak RAM | **3,012 MB** |
| First token latency | 20,121 ms |
| Accuracy (arc_easy, 50 samples) | **0.74 (74%)** |
| CPU p99 | ~51.7% |
| Thermal throttling | **None detected** |

**ADTC Self-Reported Scores:**
- **Sperf (Performance):** 82.67
- **Seff (Efficiency):** 56.97

---

## RAG System

### 1. NAERLS Verified Corpus
- **File:** `naerls_verified.csv`
- **Records:** 671
- Derived from official NAERLS extension documents covering maize, cassava, rice, livestock, aquaculture, and more.

### 2. Concise Master Knowledge Base
- **File:** `master_agro_kb.json`
- **Entries:** 78
- Written as direct farmer‑friendly answers covering crops, livestock, soil, weather, post-harvest, and marketing.
- Includes 20 Pidgin‑specific entries and local language aliases.

### Retrieval Mechanism
- Inverted‑index keyword search for fast retrieval
- Query expansion using a Pidgin term dictionary
- Curated JSON entries are prioritized when they match
- Scoring threshold prevents irrelevant matches

---

## Language Support

- **English:** full support for all agricultural queries
- **Nigerian Pidgin:** full support for agricultural queries, with natural conversational responses
- **Igbo, Yoruba, Hausa:** graceful greeting and redirect responses

The guard layer:
- Recognises greetings in five languages (English, Pidgin, Igbo, Yoruba, Hausa)
- Detects non‑agricultural topics and politely redirects
- Prevents hallucinations on out‑of‑scope questions
- Asks for clarification on vague messages

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Vicgrace01/agrigemma4.9ja.git
cd agrigemma4.9ja
```

### 2. Download the model

```bash
bash download_model.sh
```

### 3. Set up a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install gradio llama-cpp-python
```

### 4. Run the application

```bash
python app.py
```

The Gradio interface will launch locally at `http://localhost:7860`. No internet is required after the model download.

---

## Repository Structure

```
agrigemma4.9ja/
├── app.py                  # Main Gradio application with RAG + guard
├── automated_gauntlet.py   # Automated testing suite
├── build_master_kb.py      # Knowledge base builder
├── download_model.sh       # Downloads the GGUF model
├── eval_results.json       # Evaluation output
├── eval_suite.py           # 50‑case self‑evaluation suite
├── language_test_results.md # Language support test results
├── LICENSE                 # MIT License
├── master_agro_kb.json     # 78 curated agricultural advisory entries
├── metadata.json           # ADTC submission metadata
├── model/                  # Model directory (excluded from git)
│   └── gemma-4-E2B-it-Q3_K_M.gguf  # 2.4 GB model file
├── naerls_database.csv     # Raw NAERLS data
├── naerls_sources/         # Original NAERLS source documents
├── naerls_verified.csv     # 671 NAERLS‑derived records
├── README.md               # Project overview
├── REPORT.md               # Technical write‑up
├── submission.json         # ADTC submission data
├── test_gemma.py           # Model test script
├── test_languages.py       # Multilingual test script
└── tools/                  # Corpus building utilities
```

---

## Evaluation

Run the self‑evaluation suite:

```bash
python eval_suite.py
```

This tests 50 cases across greetings, crops, livestock, pests, soil, weather, post‑harvest, and more.

---

## Constraints Addressed

| Constraint | Solution |
|------------|----------|
| 8 GB RAM limit | Q3_K_M quantization reduces model to 2.4 GB and peak RAM to 3.0 GB |
| No internet during inference | Fully offline RAG with local knowledge base |
| CPU‑only | Physical core binding with n_threads=4 |
| Language barrier | English + Pidgin + multilingual greetings |
| Thermal penalty risk | Low RAM usage and efficient threading keep temperatures safe |
| Limited mobile data for testing | Prioritized model candidates; only downloaded promising models |
| Unstable internet | Resume-capable downloads with curl --continue-at - |

---

## Credits

- **Model:** Google Gemma 4 E2B IT
- **Agricultural corpus:** NAERLS (National Agricultural Extension and Research Liaison Services)
- **Inference:** llama.cpp
- **Interface:** Gradio

---

## License

This project is submitted for the Africa Deep Tech Challenge 2026.
