
# AgriGemma4.9ja

**Offline RAG-powered agricultural advisory system for Nigerian smallholder farmers.**

AgriGemma4.9ja is a local-first language model application built for the Africa Deep Tech Challenge 2026 (ADTC). It runs entirely on-device using a quantized Gemma 4 E2B IT model and offline retrieval over Nigerian agricultural records (NAERLS). It provides practical crop, livestock, soil, pest, and weather advice in English and Nigerian Pidgin, with graceful greeting support for Igbo, Yoruba, and Hausa.

---

## Problem

Smallholder farmers in Nigeria face:

- Limited access to extension officers
- High cost and poor connectivity for cloud-based AI
- Language barriers — most advice is not in local languages
- Time-sensitive pest and disease outbreaks

Cloud-hosted LLMs are not practical for rural farmers. AgriGemma4.9ja solves this by running fully offline on commodity hardware.

---

## Hardware Target

Built for the ADTC Standard Laptop profile:

- Intel Core i5 10th–12th gen (developed on i5-8365U)
- 8 GB RAM
- Integrated graphics only
- Ubuntu 22.04 LTS

Model inference is CPU-only through llama.cpp.

---

## Model Selection

Multiple models were evaluated against the ADTC constraint:

| Model | Quantization | Approx Size | Accuracy (arc_easy) | TPS | Notes |
|-------|-------------|-------------|---------------------|-----|-------|
| Llama 3.2 3B | Q4_K_M | 3.43 GB | 72% | lower | First model tested |
| Gemma 4 E2B IT | Q4_K_M | ~4.3 GB | 74% | moderate | Good but heavier |
| Gemma 4 E2B IT | Q3_K_M | 2.4 GB | 74% | 12.4 | Best balance |
| Aya | — | — | failed | — | Recommended but failed |
| TinyLlama | — | — | hallucinated | — | Not viable |

Why Gemma 4 E2B IT Q3_K_M won:

1. Same accuracy as Q4 but smaller and faster — 74% accuracy with 2.4 GB model size
2. Strong instruction following — critical for combining RAG evidence with system rules
3. Multilingual capability — practical English and Nigerian Pidgin support
4. RAG compatibility — suitable for injecting retrieved NAERLS evidence

---

## Performance

| Metric | Value |
|--------|-------|
| Generation speed | 12.4 tokens/sec |
| Peak RAM | 3,012 MB |
| First token latency | 20,121 ms |
| Accuracy (arc_easy, 50 samples) | 0.74 |
| CPU p99 | ~51.7% |
| Thermal throttling | None |

---

## RAG System

### 1. NAERLS Verified Corpus
- File: naerls_verified.csv
- Records: 671
- Derived from official NAERLS extension documents

### 2. Concise Master Knowledge Base
- File: master_agro_kb.json
- Entries: 78
- Written as direct farmer-friendly answers
- Includes 20 Pidgin-specific entries

### Retrieval Mechanism
- Inverted-index keyword search
- Query expansion using a Pidgin term dictionary
- Curated JSON entries prioritized when they match
- Scoring threshold prevents irrelevant matches

---

## Language Support

- English: full support
- Nigerian Pidgin: full support for agricultural queries
- Igbo, Yoruba, Hausa: graceful greeting and redirect responses

The guard layer:

- Recognizes greetings in five languages
- Detects non-agricultural topics and politely redirects
- Prevents hallucinations on out-of-scope questions
- Asks for clarification on vague messages

---

## Quick Start

### 1. Clone the repository

    git clone https://github.com/Vicgrace01/agrigemma4.9ja.git
    cd agrigemma4.9ja

### 2. Download the model

    bash download_model.sh

### 3. Set up a virtual environment

    python3 -m venv .venv
    source .venv/bin/activate
    pip install gradio llama-cpp-python

### 4. Run the application

    python app.py

The Gradio interface will launch locally. No internet is required after the model download.

---

## Repository Structure

   agrigemma4.9ja/
├── app.py # Main Gradio application with RAG + guard
├── automated_gauntlet.py # Automated testing suite
├── build_master_kb.py # Knowledge base builder
├── download_model.sh # Downloads the GGUF model
├── eval_results.json # Evaluation output
├── eval_suite.py # 50-case self-evaluation suite
├── language_test_results.md # Language support test results
├── LICENSE # MIT License
├── master_agro_kb.json # 78 curated agricultural advisory entries
├── metadata.json # ADTC submission metadata
├── model/ # Model directory (excluded from git)
│ └── gemma-4-E2B-it-Q3_K_M.gguf # 2.4 GB model file
├── naerls_database.csv # Raw NAERLS data
├── naerls_sources/ # Original NAERLS source documents
├── naerls_verified.csv # 671 NAERLS-derived records
├── README.md # Project overview
├── REPORT.md # Technical write-up
├── submission.json # ADTC submission data
├── test_gemma.py # Model test script
├── test_languages.py # Multilingual test script
└── tools/ # Corpus building utilities

✅ Full ADTC Compliance Checklist
Requirement	Status	Notes
Repository is PUBLIC	✅	https://github.com/Vicgrace01/agrigemma4.9ja
README.md exists	✅	Complete with all sections
REPORT.md exists	✅	Technical report complete
LICENSE exists	✅	MIT License
.gitignore exists	✅	Excludes .gguf files
No large files in git	✅	Model not committed
metadata.json exists	✅	All fields filled
download_model.sh exists	✅	Downloads model to model/
model/*.gguf excluded	✅	In .gitignore
submission.json complete	✅	All fields filled
test_prompts (exactly 2)	✅	Agriculture domain
african_alpha_claim	✅	true
budget_laptop_claim	✅	true
model.runtime	✅	llama.cpp
model.quantization	✅	GGUF Q3_K_M
_runtime.model_path	✅	model/gemma-4-E2B-it-Q3_K_M.gguf
📊 Performance Metrics Verification
Metric	Your Value	Source	Status
Model file size	2.4 GB	ls -lh model/	✅
Peak RAM	3,011.63 MB	submission.json	✅
Inference speed	12.4 TPS	submission.json	✅
Accuracy	0.74 (74%)	submission.json	✅
Thermal throttling	None	submission.json	✅
🎯 Test Prompts Verification
Prompt ID	Prompt	Domain	Language
tp_001	A maize farmer in Kaduna reports sawdust-like frass in the whorl and leaf damage. What is the likely pest and what practical first response should the farmer take?	Agriculture	English
tp_002	Abeg, my cassava leaf dey yellow and curl. Wetin fit cause am, and wetin I fit do first make e no spread?	Agriculture	Nigerian Pidgin
✅ Both prompts are in the Agriculture domain
✅ Both are practical, real-world queries
✅ One is in Nigerian Pidgin (African language bonus)

📝 metadata.json Check
Field	Value	Status
team_id	team-agrigemma49ja	✅
domain	agriculture	✅
language_scope	["en", "pcm"]	✅
african_alpha_claim	true	✅
budget_laptop_claim	true	✅
submitter.name	Victor Chukwuebuka Nwaruwe	✅
submitter.email	victornwaruwe@gmail.com	✅
submitter.github_handle	vicgrace01	✅
cross_disciplinary_pairing	offline agricultural knowledge retrieval	✅
test_prompts	2 prompts	✅
model.runtime	llama.cpp	✅
model.quantization	GGUF Q3_K_M	✅
model.parameters_estimate	4.65B	✅
✅ Final Summary
Category	Status
Required Files	✅ All present
Documentation	✅ Complete
Performance Data	✅ Verified
Template Compliance	✅ 100%
African Bonus	✅ Claimed
Language Bonus	✅ Claimed
Budget Laptop	✅ Claimed
No Model in Git	✅ Verified
🚀 Ready for DevPost Submission!
Your GitHub Repository: https://github.com/Vicgrace01/agrigemma4.9ja

Final Steps:
Update README with correct repository structure (copy the updated section above)

Record 2-minute demo video

Submit on DevPost: https://adtc-2026.devpost.com/

Good luck, Victor! AgriGemma4.9ja is fully compliant and ready! 🚀🌾


---

## Evaluation

Run the self-evaluation suite:

    python eval_suite.py

This tests 50 cases across greetings, crops, livestock, pests, soil, weather, post-harvest, and more.

---

## Constraints Addressed

| Constraint | Solution |
|------------|----------|
| 8 GB RAM limit | Q3_K_M quantization |
| No internet during inference | Fully offline RAG |
| CPU-only | Physical core binding |
| Language barrier | English + Pidgin + multilingual greetings |
| Thermal penalty risk | Low RAM usage, efficient threading |

---

## Credits

- Model: Google Gemma 4 E2B IT
- Agricultural corpus: NAERLS
- Inference: llama.cpp
- Interface: Gradio

---

## License

This project is submitted for the Africa Deep Tech Challenge 2026.

