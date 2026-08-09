# AgriGemma: Offline LLM for African Agriculture

## Problem
Smallholder farmers in Enugu State often lack reliable internet access. To deploy an agricultural advisory tool—especially one meant to interface with the SoilFi startup utilizing IoT soil telemetry for agricultural loans—the intelligence must operate completely offline at the edge. 

## Design Decisions
Initially, an 8-billion parameter model was evaluated, but it resulted in 0.0 tokens per second due to memory bandwidth bottlenecks and swap file thrashing. The architecture was pivoted to `Llama-3.2-3B-Instruct` quantized to `Q4_K_M`. At roughly 3.2 billion parameters, it retains sufficient reasoning for localized agricultural advice while remaining highly performant on commodity hardware.

## Constraints
The target hardware profile is a heavily constrained laptop (Intel i5, 8 GB RAM, no dedicated GPU). The memory allocation was strictly monitored to ensure the OS overhead was not compromised, preventing Out-Of-Memory (OOM) failures and thermal throttling.

## Benchmarks (Local Evaluation)
- **Hardware:** Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz, 5.8 GB Allocated RAM
- **Throughput:** 7.38 Tokens Per Second (Generation)
- **Peak Memory (RSS):** 3.43 GB (Leaves ~2.3 GB overhead for OS)
- **Thermal Status:** No CPU throttling observed during inference.
