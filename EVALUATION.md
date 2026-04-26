# Anvil — Evaluation Results

**Date:** 2026-04-26 · phi3.5:3.8b (local) · claude-sonnet-4-6 (toolsmith) · all-MiniLM-L6-v2 (embedder).

## Experiment 1 — Baseline (threshold = 0.7)

| Path | Count | % |
|---|---|---|
| `local_direct` | 8 | 40% |
| `generated` | 6 | 30% |
| `fallback` | 6 | 30% |
| `cache_hit` | **0** | **0%** |

Latency: 0.3s – 419s, median ~16s.

| # | Query | Path | Tool | Lat |
|---|---|---|---|---|
| 1 | 17 × 23 | local_direct | — | 5.3s |
| 2 | capital of Mongolia | local_direct | — | 0.3s |
| 3 | 2 + 2 | generated | evaluate_math_expression | 26.4s |
| 4 | weather in Tokyo | generated | get_current_weather | 32.7s |
| 5 | time in Sydney | fallback | — | 47.2s |
| 6 | top news today | generated | get_top_news_headlines | 69.8s |
| 7 | rain in Paris | fallback | — | 15.0s |
| 8 | time in São Paulo | local_direct | — | 2.3s |
| 9 | population of Iceland | local_direct | — | 0.7s |
| 10 | ISS location | generated | get_iss_location | 18.1s |
| 11 | convert 100 USD→EUR | local_direct | — | 5.8s |
| 12 | BTC price | generated | get_crypto_price | 15.1s |
| 13 | population of Norway | local_direct | — | 2.2s |
| 14 | ETH price | generated | get_crypto_price (overwrite) | 24.8s |
| 15 | Minecraft | fallback | — | 16.0s |
| 16 | train transformer | fallback | — | 171.8s |
| 17 | real-time chat app | fallback | — | 34.0s |
| 18 | send email | local_direct | — | 2.4s |
| 19 | summarize Wikipedia QE | local_direct | — | 16.3s |
| 20 | meaning of life | fallback | — | 419.0s |

**Takeaways:**
- 0% cache-hit rate; every existing tool got overwritten instead of reused.
- Threshold 0.7 too high — `all-MiniLM-L6-v2` doesn't produce >0.7 for short paraphrased queries.
- 4 live-data queries answered from stale training data (q9, q11, q13, plus q19 didn't fetch).
- Infeasibility branch worked 4/4 but latency 16s–419s.
- Phi UNKNOWN-detection brittle (q8: `UNKNOWN` mid-text leaked through).

---

## Architecture change — hybrid RAG

Removed `similarity_threshold`; pass top-3 candidates directly to `route_tools`. `route_tools` returns NONE when no tool fits, so cosine filtering was redundant gating.

Files: `src/anvil/server.py`, `src/anvil/web.py`, `tests/test_integration.py`.

---

## Experiment 2 — Cache-reuse test

| # | Query | Path | Tool | Sim |
|---|---|---|---|---|
| 1 | weather in Tokyo | **cache_hit** | get_current_weather | 0.329 |
| 2 | time in Sydney | **cache_hit** | get_current_time_in_timezone | 0.367 |
| 3 | top news today | local_direct | — (phi step-1) | — |
| 4 | BTC price | **cache_hit** | get_crypto_price | 0.564 |
| 5 | ISS location | local_direct | — (phi step-1) | — |
| 6 | rain in Paris | local_direct | — (phi step-1) | — |
| 7 | ETH price | **cache_hit** | get_crypto_price | 0.559 |
| 8 | time in São Paulo | **cache_hit** | get_current_time_in_timezone | 0.306 |
| 9 | Iceland population (ctrl) | local_direct | — | — |
| 10 | Minecraft (ctrl) | fallback | — | — |

**Takeaways:**
- 5/5 (100%) cache-hit on queries that reached the router.
- 3 misses are all phi-step-1 hallucinations, not routing failures.
- Every successful match had `sim < 0.6`; three under 0.4. The old 0.7 floor would have blocked all of them.

| Query | Before | After |
|---|---|---|
| weather Tokyo | regenerated 32.7s | cache_hit 23.1s |
| time Sydney | fallback 47.2s | cache_hit 14.6s |
| BTC price | generated 15.1s | cache_hit 16.3s |
| ETH price | regenerated 24.8s | cache_hit 14.7s |
| time São Paulo | hallucinated 2.3s | cache_hit 15.5s |

---

## Experiment 3 — Verified 20-prompt eval

Verifier outcomes: **16 PASS · 3 WARN · 1 FAIL.**

| Path | Count |
|---|---|
| `local_direct` | 12 |
| `cache_hit` | 7 |
| `fallback` | 1 |
| `generated` | 0 |

Latency: 0.2s – 35.5s. No 100s+ outliers.

| # | Query | Path | Lat | Verify | Output |
|---|---|---|---|---|---|
| 1 | 17 × 23 | local_direct | 4.6s | ✅ | 391 |
| 2 | 2 + 2 | cache_hit | 10.8s | ✅ | 4 |
| 3 | 12 squared | local_direct | 0.6s | ✅ | 144 |
| 4 | capital of Mongolia | local_direct | 0.3s | ✅ | Ulaanbaatar |
| 5 | symbol for gold | local_direct | 0.2s | ✅ | Au |
| 6 | who wrote Hamlet | local_direct | 0.3s | ✅ | Shakespeare |
| 7 | weather in London | cache_hit | 20.8s | ✅ | 14.7°C clear |
| 8 | time in Tokyo | cache_hit | 14.7s | ✅ | 04:08 AM (UTC+9 ✓) |
| 9 | BTC price | cache_hit | 15.2s | ✅ | $78,253 |
| 10 | ETH price | cache_hit | 25.7s | ✅ | $2,361 |
| 11 | ISS location | local_direct | 2.5s | ⚠️ | refusal leaked |
| 12 | rain in Paris | cache_hit | 14.8s | ✅ | 20.4°C overcast (synth correctly noted "not raining") |
| 13 | time in São Paulo | cache_hit | 15.1s | ✅ | 04:10 PM (UTC-3 ✓) |
| 14 | Iceland population | local_direct | 1.0s | ✅ | 368,645 *(stale)* |
| 15 | Norway population | local_direct | 2.7s | ⚠️ | "5.4 million" (regex limit) |
| 16 | Minecraft | fallback | 35.5s | ✅ | canned message |
| 17 | train LLM from scratch | local_direct | 32.8s | ❌ | phi wrote tutorial |
| 18 | send email | local_direct | 2.6s | ⚠️ | drafted email instead of refusing |
| 19 | summarize Wikipedia QE | local_direct | 16.1s | ✅* | summary from training (didn't fetch) |
| 20 | meaning of life | local_direct | 2.0s | ✅ | brief answer (vs 419s in run 1) |

**Takeaways:**
- 7/7 cache-hits correct; 0 spurious regenerations.
- All numerically verifiable answers correct: math, facts, time-zone offsets, price ranges.
- Synthesizer reasoned over tool output well (q12 Paris).
- All 4 misses are step-1 phi hallucinations:
  - q11: trailing `UNKNOWN` not caught by parser.
  - q14, q15: stale data hedged with *"as per my knowledge cutoff"*, parser accepted.
  - q17: phi wrote tutorial instead of UNKNOWN; identical-intent query was fallback in Experiment 1 → pure phi non-determinism.
  - q18: phi drafted email — action-vs-query gap.

---

## Experiment 4 — Wrong-tool-via-grounded-args regression

Discovered case: query "What is the current KRW price compared to USD today?" routed to `get_crypto_price` because the description says "cryptocurrency in various fiat currencies" and the query mentions KRW/USD. The parameter extractor invented `coin_id="bitcoin"` to satisfy the schema. The synthesizer dressed up the BTC/KRW number as an answer to the FX question.

Two guards added to address this class of failure:

**(a) Extraction-grounding guard** in `extract_parameters` — reject if any required parameter's value isn't a substring (or part-after-`/_-` split) of the user's query. Catches "invented" values.

**(b) Verification gate** in `handle_query` — after `route_tools` picks a tool, ask phi: *"does this tool's described capability allow you to answer this task?"* (YES/NO). If NO, abandon the cache hit and fall through to the toolsmith. Catches semantic mismatch.

### Test (6 prompts: 1 regression, 3 controls, 2 ambiguous)

| # | Query | Original | + Grounding guard | + Verification gate |
|---|---|---|---|---|
| 1 | KRW vs USD (regression) | cache_hit BTC/KRW (wrong) | cache_hit FX-shaped (lucky) | **still cache_hit** ❌ |
| 2 | BTC price (control) | ✓ | ✓ | ✓ |
| 3 | weather Tokyo (control) | ✓ | ✓ | ✓ |
| 4 | time Sydney (control) | ✓ | ✓ | local_direct (phi step-1, unrelated) |
| 5 | "current crypto price" (no coin) | cache_hit BTC | generated degenerate tool | **fallback** ✓ |
| 6 | "weather right now" (no city) | cache_hit "Dunkirk, UK"(!) | **fallback** ✓ | **fallback** ✓ |

**Takeaways:**
- Wrong-answer rate on problem set: **3/3 → 1/3**.
- All 3 legit controls preserved.
- q1 is a known unresolvable-with-phi3.5 case: tool description ("crypto in fiat currencies") and query (KRW/USD) share enough vocabulary that phi3.5 rationalizes the fit and the grounding guard passes because both currency strings appear verbatim in the query.
- Escalating q1 to a bigger verification model (e.g. Claude for the YES/NO gate only) is left as a deferred option.

---

## Net result

| Metric | Baseline | After all fixes |
|---|---|---|
| Cache-hit rate (overall) | 0% | 35% |
| Cache-hit rate (queries that reached router) | n/a | 100% |
| Verified-PASS rate (Experiment 3) | n/a | 80% |
| Wrong-answer rate on adversarial set | 3/3 | 1/3 |
| Latency p50 | ~16s | ~5s |
| Latency p100 | 419s | 35.5s |

**Remaining limitations:**
1. `answer_directly` lets phi answer questions it shouldn't — stale-data hedging, refusal-text leaks, action drafting, infeasibility tutorials. Fix: tighten UNKNOWN-detection / expand `_UNCERTAIN_PHRASES` / add action-scaffolding detector. LoRA deferred.
2. Wrong-tool-via-grounded-args (q1 KRW/USD) — phi3.5 too small to reliably distinguish overlapping-vocabulary tool descriptions from semantically distinct user intents. Fix: escalate verification gate to a bigger model.
