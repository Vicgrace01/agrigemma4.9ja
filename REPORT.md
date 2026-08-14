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

## 2. Problem Definition

### 2.1 The Scale of the Challenge

Nigeria is Africa's largest agricultural economy, with agriculture contributing approximately **25% of national GDP** and employing roughly **70% of the workforce**. Smallholder farmers are the backbone of this system, producing an estimated **85% of the food consumed in the country** [1].

### 2.2 Escalating Crop Losses

Partial crop losses have **more than tripled over the past five years**, rising from approximately **6% to over 20% of agricultural plots** between 2018/19 and 2023/24 [1].

**Key Statistics:**

- Partial crop losses: 6% → 20%+ of plots (2018/19 to 2023/24) [1]
- One-third of total losses attributed to weather/environmental factors [1]
- Two-thirds of partial losses attributed to weather shocks [1]
- Average fertilizer price increase of 20% in 2025 [1]
- Conflict's contribution to crop losses nearly doubled, reaching one-fifth of national losses by 2023/24 [1]

### 2.3 Climate Shocks and Environmental Stress

Weather and environmental factors remain the **single largest contributor to crop losses nationwide**, accounting for more than **one-third of total losses** and nearly **two-thirds of partial losses** in 2023/24 [1]. Field crops such as maize and cassava bear the brunt due to their direct exposure to rainfall variability.

### 2.4 Conflict and Insecurity

Between 2018/19 and 2023/24, **conflict's contribution to total crop losses nearly doubled**, reaching roughly **one-fifth of national losses** by 2023/24 [1]. In the northeast, northwest, and parts of the northcentral region, insecurity now rivals climate as a driver of production loss.

### 2.5 The Extension Agent Gap

Nigeria's agricultural extension system faces a critical shortage of personnel. According to FAO and IFPRI data, Nigeria currently has **1 extension officer serving between 5,000 and 10,000 farmers**. In some states, this ratio is as extreme as **1:24,000 to 1:30,000** [2][3].

| Organization | Recommended Ratio |
|--------------|-------------------|
| FAO | 1:400 to 1:800 |
| World Bank | 1:800 to 1:1,000 |
| **Nigeria Current** | **1:5,000 to 1:10,000** |

**This means Nigeria's extension system is operating at 10-20 times below recommended capacity.** [2]

### 2.6 Language Barriers

Most agricultural information is disseminated in English, yet many smallholder farmers communicate primarily in local languages. Nigerian Pidgin serves as a widely understood lingua franca, but content in Pidgin remains scarce.

### 2.7 Target Users

The primary target users are Nigerian smallholder farmers and extension officers who need offline, localized agricultural advisory in familiar languages.

---

## 3. The Opportunity: AI for African Realities

The Africa Deep Tech Challenge 2026 was created to address exactly this type of problem: **building useful AI applications that run on the hardware Africans already own** [4]. The target is the ADTC Standard Laptop—an 8 GB RAM machine with integrated graphics, representative of what sits on millions of desks across the continent.

Mobile advisory services using digital tools have shown promise. In pilot programs across Africa, mobile apps providing real-time forecasts and planting advice have **reduced crop losses by up to 20%** [5]. However, most of these solutions rely on cloud connectivity—a significant blocker in rural areas with limited internet access and unreliable electricity.

**AgriGemma4.9ja bridges this gap** by bringing offline, on-device AI advisory directly to farmers, eliminating cloud dependencies while delivering localized, practical agricultural knowledge in familiar languages.

---

## 4. Design Decisions

### 4.1 Model Selection

Multiple models were evaluated against the ADTC Standard Laptop constraint: 8 GB RAM, CPU-only inference, and no cloud dependencies [4].

**Candidate Models Tested:**

| Model | Quantization | Approx Size | Accuracy | TPS | Notes |
|-------|-------------|-------------|----------|-----|-------|
| Llama 3.2 3B | Q4_K_M | 3.43 GB | 72% | lower | First model tested |
| Gemma 4 E2B IT | Q4_K_M | ~4.3 GB | 74% | moderate | Good but heavier |
| **Gemma 4 E2B IT** | **Q3_K_M** | **2.4 GB** | **74%** | **12.4** | **Best balance** |
| Aya | — | — | failed | — | Recommended but failed |
| TinyLlama | — | — | hallucinated | — | Not viable |

### 4.2 Why Gemma 4 E2B IT Q3_K_M Was Selected

1. **Same accuracy as Q4 but smaller and faster.** Q3_K_M kept 74% accuracy while reducing model size to 2.4 GB and improving throughput to 12.4 TPS.

2. **Strong instruction following.** Follows structured agricultural prompts better than smaller models.

3. **Multilingual capability.** Supports English and Nigerian Pidgin for practical advisory.

4. **RAG compatibility.** Suitable for injecting retrieved NAERLS evidence directly into the prompt.

5. **Failure of smaller alternatives.** Aya and TinyLlama did not work well in practice.

### 4.3 Quantization Selection

Q3_K_M was chosen over Q4_K_M because:
- Same accuracy (74%) at half the model size
- Reduced memory footprint from ~4.3 GB to 2.4 GB
- Improved inference speed to 12.4 TPS
- Maintained stability and response quality

---

## 5. Constraints Addressed

| Constraint | Solution |
|------------|----------|
| 8 GB RAM limit | Q3_K_M quantization |
| No internet during inference | Fully offline RAG |
| CPU-only | Physical core binding |
| Language barrier | English + Pidgin |
| Thermal penalty risk | Low RAM usage, efficient threading |
| **Limited mobile data for testing** | **Prioritized model candidates; only downloaded promising models** |
| **Unstable internet** | **Resume-capable downloads with curl --continue-at -** |

---

## 6. Tools and Technologies

| Tool | Purpose | Justification |
|------|---------|---------------|
| llama.cpp | Model inference | Only runtime accepted by ADTC |
| Python 3.10 | Application logic | Ubiquitous, well-supported |
| Gradio | Web interface | Lightweight, local-first UI |
| Hugging Face | Model hosting | Public, free, reliable |
| NAERLS | Agricultural corpus | Official Nigerian extension records |

---

## 7. Performance Benchmarks

Measured with the official adtc-profiler on Ubuntu 22.04.5 LTS, Intel Core i5-8365U, 5.8 GB RAM available [6].

| Metric | Value |
|--------|-------|
| Generation throughput | 12.4 tokens/second |
| Peak RAM | 3,011.63 MB |
| Steady-state RAM | 2,901.06 MB |
| First token latency | 20,120.63 ms |
| Accuracy (arc_easy, 50 samples) | 0.74 |
| CPU p99 | ~50.5% |
| Thermal throttling | None detected |

**Model Details:**
- Architecture: Gemma 4 E2B IT
- Quantization: GGUF Q3_K_M
- Parameter count: 4,647,450,147 (~4.65B)
- Model file size: 2.4 GB
- Context length: 131,072 tokens

---

## 8. RAG System Design

### 8.1 NAERLS Verified Corpus

- File: naerls_verified.csv
- Records: 671
- Source: Official NAERLS extension documents [7]

### 8.2 Master Knowledge Base

- File: master_agro_kb.json
- Entries: 78
- Format: Direct farmer-friendly answers
- Localization: 20 Pidgin-specific entries

### 8.3 Retrieval Mechanism

- Inverted-index keyword search for fast retrieval
- Query expansion using a Pidgin term dictionary
- Curated JSON entries prioritized when they match
- Scoring threshold prevents irrelevant matches

---

## 9. Hardware and Software Optimization

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_threads | 4 | CPU core binding for optimal throughput |
| n_batch | 512 | Efficient prompt processing |
| n_ctx | 4096 | Balances context and memory |
| Temperature | 0.20 | Deterministic, factual responses |
| repeat_penalty | 1.12 | Reduces looping |

**Guard Rails:**
- Recognizes greetings in English, Pidgin, Igbo, Yoruba, Hausa
- Detects non-agricultural topics and politely redirects
- Prevents hallucinations on out-of-scope queries

---

## 10. Domain Application Validation

**Test Prompt 1 — Pest Identification**

*Prompt:* A maize farmer in Kaduna reports sawdust-like frass in the whorl and leaf damage. What is the likely pest and what practical first response should the farmer take?

*Result:* Correctly identified Fall Armyworm risk and recommended early-stage control using neem or approved insecticide.

**Test Prompt 2 — Pidgin Disease Query**

*Prompt:* Abeg, my cassava leaf dey yellow and curl. Wetin fit cause am, and wetin I fit do first make e no spread?

*Result:* Correctly identified Cassava Mosaic Disease (CMD), recommended removing and burning infected plants, and advised planting TME 419 or NR 8082 resistant varieties.

---

## 11. Conclusion

AgriGemma4.9ja demonstrates that useful agricultural AI can run entirely offline on 8 GB laptops common across Africa. By combining model quantization, llama.cpp optimization, and local RAG over Nigerian agricultural records, we deliver practical crop and livestock advice without cloud dependencies or high-end hardware.

The solution directly addresses the challenges facing Nigerian smallholder farmers, who produce **85% of the nation's food** yet face escalating crop losses from climate shocks, conflict, and limited access to extension services [1]. With only **1 extension officer for every 5,000 to 10,000 farmers**—compared to the FAO-recommended ratio of 1:400 to 1:800—the need for scalable, offline advisory tools has never been more urgent [2][3].

This approach is replicable across other domains and languages, making it a scalable model for AI access in resource-constrained environments across Africa.

---

## 12. References

1. World Bank Blogs. (2026). "From loss to resilience in Nigeria: turning a growing agricultural challenge into action." World Bank.

2. FAO. (2022). "Extension and advisory services in Nigeria." Food and Agriculture Organization.

3. IFPRI. (2021). "Agricultural extension in Nigeria: Challenges and opportunities."

4. Africa Deep Tech Challenge 2026. (2026). "The Laptop LLM Challenge." CompeteHub / ADTC.

5. Adolwa, I.S., et al. (2025). "Delivering nutrient management impact through farmer-centric research." Agricultural Systems 229: 104416.

6. Africa Deep Tech Foundation. (2026). "ADTC Profiler." GitHub.

7. NAERLS. (2025). "Agricultural Performance Survey (APS) 2025." Federal Ministry of Agriculture, Nigeria.

---

**End of Report**
