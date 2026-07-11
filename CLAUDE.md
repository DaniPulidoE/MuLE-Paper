# MuLE — DPO Strategy & Pipeline Guide

MuLE tests whether **Direct Preference Optimisation (DPO)** can achieve multilingual
reasoning alignment at a fraction of the cost of the GRPO-based M-Thinker project.
The base model is the M-Thinker Cold-Start SFT 1.5B (`XueZhang-bjtu/1.5B-cold-start-SFT`).
Preference pairs are self-generated in **en / fr / pt**; **es** is held out as an
out-of-domain test language. Benchmarks: MMATH and PolyMath (metrics: LC, Acc, LC&Acc).

This document records the **current** pair-construction + training strategy and
explains what changed relative to the first (underperforming) version.

---

## 1. Why the first version underperformed

The initial run gained accuracy only on easy questions and **lost** non-English
language consistency (LC). Root causes, in order of impact:

1. **Likelihood displacement.** LC-contrast pairs (chosen = correct in-language,
   rejected = correct-but-English) are *near-duplicates* — same math, different
   language — and the chosen was selected for maximal chrF similarity to the very
   answers used as rejected. Vanilla sigmoid DPO on near-duplicate pairs drives
   *both* likelihoods down and leaks probability mass to out-of-distribution
   (mixed-language) text. **Confirmed on W&B**: `rewards/chosen` declined while
   `rewards/margins` grew — the trainer "won" by pushing rejected down faster than
   chosen, while making the model *less* likely to produce its own correct answers.
2. **Noisy LC labels.** The filter used whole-text `langdetect.detect()` (majority
   vote), which certifies code-switched traces as consistent, so the model was
   trained toward outputs the eval then penalised. langdetect's pt/es confusion
   also mislabelled clean Portuguese as Spanish and pushed some of the model's best
   pt answers into the *rejected* pool (the "Portuguese poisoning").
3. **The escape-hatch mechanism.** For these models, code-switching to English is
   how capability is *recruited* under difficulty (evidence: SFT-7B is a stronger
   reasoner than SFT-1.5B yet code-switches far more, LC 0.11–0.19 at medium+).
   The old rejected priority suppressed the English escape hatch **without**
   supplying in-language competence, producing a thrash-then-switch equilibrium:
   longer traces, more backtracking, LC *below* baseline.
4. **Offline DPO's hard-tier blind spot.** DPO can only sharpen behaviours already
   present in the sample pool. Hard questions with no correct in-language sample
   contribute nothing, so gains concentrate at low difficulty.
5. **Format artefact.** The old `format_reward` required `\boxed{}` within 800 chars
   of `</think>`; the model's median summary is ~1.4k chars, so ~85% of well-formed
   answers "failed" format for a reason the eval never checks.

**Key data (from `iter1_questions.jsonl`, 2,425 questions × 3 langs × 8 answers):**
- Per-answer accuracy: **en 60.3% / fr 33.5% / pt 31.7%**; only **5.4%** of
  questions are unsolved in *all* languages (the pool is too easy, not over-filtered).
- Real format compliance ~72% (not 15%); `format | acc==1` ≈ **98%** once the
  800-char rule is dropped.
- LC-contrast rejected (correct-English) answers are **~4× longer** than chosen
  (median ~14.9k vs ~3.6k chars) → a length confound rode on every LC pair.
- Accuracy-contrast opportunities outnumber LC-contrast ~**3:1** for fr/pt, yet the
  old priority-fill spent slots on the scarcer, riskier LC pairs first.

---

## 2. Pipeline & file map

```
RL Train Answers Generation/generate_dpo_data.py   # vLLM, N=8/lang, temp 0.9
DPO Pair Generation/dataset_filter.py              # scoring + pair construction  <-- main strategy
DPO Pair Generation/rewards.py                     # acc / LC scoring (M-Thinker)
DPO Pair Generation/compare_lid.py                 # LID calibration harness
DPO Pair Generation/format_stats.py                # per-language format decomposition
DPO train/dpo_train_no_precomp.py                  # QLoRA + Unsloth + TRL DPOTrainer
evaluation/eval_tools/MMATH/cal-MMATH-acc.py       # the authoritative eval scorer
PolyMath Answers Statistics/polymath_data.py       # PolyMath stats (shares backtrack regex)
```

The filter is aligned to the **eval** (`cal-MMATH-acc.py`), not just to the training
reward (`rewards.py`). Where the two disagree, the filter takes the strict intersection.

---

## 3. Pair-construction strategy (current `dataset_filter.py`)

### 3.1 Two contrast types, deliberately mixed
- **Accuracy-contrast**: chosen and rejected both in-language, differ in correctness.
  Isolates the *math* signal.
- **LC-contrast**: chosen and rejected both correct, differ in language (rejected =
  correct-but-English). Isolates the *language* signal. Requires the anchored loss
  (§5) to be safe.

### 3.2 Language-consistency detection — windowed, per-section, 3-way
`language_consistency_label(text, lang) -> {1, 0, -1}`
- **Detector**: fastText `lid.176.bin`, probabilities **renormalised over
  {en, fr, pt, es}** so short windows stay decidable. (Chosen over lingua after
  calibration — see §7.)
- **Windows**: 350 chars, stride 175 (overlap so a real switch spans ≥2 windows).
- **Confidence gate**: renormalised top-1 < 0.8 → window is "unsure" (`None`).
- **Run-length confirmation**: an off-language window counts only inside a run of
  ≥2 consecutive windows (isolated hits = detector noise). A single-window section
  (short answer part) has no neighbour, so one confident hit counts.
- **Per-section** (mirrors the eval): split once at `</think>`; **both** the think
  and answer sections must be consistent. Combine: any section `0` → `0`;
  else any section `-1` → `-1`; else `1`. This closes the gap where a clean think
  section hid a short off-language answer section.
- **Lexical English check** (`EN_EXCLUSIVE`, whole text): ≥3 English-exclusive
  function words ("the/wait/which/…") demote a `1` to `-1` (word-level switches
  below window granularity). Never forces a confident negative.
- **`_strip_math` is byte-identical to the MMATH eval** — do not diverge it, or the
  filter's labels stop matching the metric.

The three-way label routes answers:
- `1` consistent → **chosen-eligible**.
- `0` confident negative → usable as **LC-contrast rejected**.
- `-1` ambiguous → **excluded from both pools** (never imitated, never pushed down).
  Errors here cost only yield; errors in `1`/`0` corrupt training. That asymmetry is
  the whole design.

### 3.3 Format — eval-aligned, hard-required for chosen
`format_reward` now = **exactly one `</think>`** (the eval's `split` needs exactly two
parts) **AND `len > 1000` AND `\boxed{}` after the close**. The 800-char promptness
rule is gone. Format is a **hard requirement for chosen** (costs ~2% of the correct
pool; guarantees every chosen is scoreable by the benchmark). Correct in-language
answers with broken format go to neither pool (`excluded_correct_bad_format`).

### 3.4 Chosen selection
`acc == 1 AND lc == 1 AND format == 1 AND rep < CHOSEN_REP_MAX (0.15)`. The rep
cap is a **hard gate**, not just a rank penalty: with the RPO anchor a chosen
answer is actively imitated, and on sole-correct-answer questions ranking can't
skip a degenerate loop. Correct-but-looping answers go to neither pool
(`excluded_correct_high_rep`). Eligible candidates ranked by the **D2 score**
(from the DPO+GRPO implementation spec):
`candidate_score = chrF_alignment − LAMBDA_REP·100·rep + MU_UNIQ·uniq`
- `rep(a)`: duplicate word-10-gram rate, [0,1] — language-neutral looping measure.
- `uniq(a)`: distinct 10-grams / 1000 — novel content mass (offline only; never
  use as an online RL reward, it is paddable).
- Defaults `LAMBDA_REP=0.3`, `MU_UNIQ=1.0`; tune on the validation split.
- The backtrack penalty is **gone** (bt ≈ 0 for every lc==1 candidate, so it was
  blind where it applied; bt stays computed/logged as a diagnostic, English-only
  comparisons). The format term stays gone (hard gate).
- Both metrics computed on full text with `<think>` tags stripped, LaTeX kept
  (`rep_uniq()`, n=10 — keep the order fixed everywhere).

### 3.5 Rejected selection — `select_rejected_mixed` (replaces `select_rejected`)
Quota-based, **not** priority-fill:
- **en, or k == 1**: accuracy-contrast preferred, LC-contrast fallback (easy
  questions rarely need language pressure — low-tier LC was already ~0.98).
- **fr/pt, k ≥ 2**: **guarantee one accuracy-contrast AND one LC-contrast** when both
  pools are non-empty; fill the remainder accuracy-first, junk pool (`other_rejected`)
  last.
- **LC picks are length-matched** to the specific chosen they zip with (removes the
  4× length confound; random sampling taught "shorter = better").
- Realised mixture is logged per language: `pairs_{en,fr,pt}_{acc_contrast,lc_contrast,other}`.

### 3.6 Pair type 2 — length-matched anti-looping pairs (spec D3/D4)
The real in-language failure mode is degenerate repetition, not backtracking
(SFT/MuLE in-language rep ≈0.18–0.24 vs healthy English ≈0.015; confirmed on
iter1: `rep_mean_pt` ≈ 0.22 all-answers vs ≈0.08 chosen-eligible). Per
(question, lang), at most **one extra pair**:
- Pool A (chosen): `acc==1 ∧ lc==1 ∧ format==1 ∧ rep < 0.08`
- Pool B (rejected): `acc==0 ∧ lc==1 ∧ rep > 0.15` (looped and failed, in-language)
- Length-matched within ±25% (chars) so the dominant contrast is looping vs
  progress; the closest-length (A,B) pair is taken, else
  `antiloop_skipped_length_mismatch` is incremented.
- Needs no chrF pivot, so it also fires on pivot-language questions that pair
  type 1 skips. Logged as `pairs_{lang}_antiloop`.
- **D4 census** (in `stats.json` under `antiloop_census` + printed): eligible
  long clean chosen candidates (`rep<0.08`, >3k words ≈ 4k tokens) per
  (lang, difficulty). Cells under 50 are recorded as shortfalls, never
  fabricated — a shortfall bounds what offline DPO can achieve there.

### 3.7 Prompt / think-tag handling
`_dedup_leading_think`: the chat template's generation prompt ends with `<think>\n`
and stored candidates also carry a prepended `<think>\n`. The completion's leading
tag is stripped **only when the prompt already ends with one**, so training sequences
never contain a doubled `<think>\n<think>\n` the model never sees at inference. (The
old `dpo_dataset.jsonl` had this doubling in 100% of pairs; it was previously fixed
out-of-band into `dpo_dataset_fixed.jsonl` — now fixed inside the pipeline.)

### 3.8 Backtrack signals
`BACKTRACK_SIGNALS` is the **expanded multilingual set** (en/fr/pt/es), synced with
`PolyMath Answers Statistics/polymath_data.py`, plus a **round-3 data-mined block**
(e.g. fr `vérifie`/`je vérifie` ≈ 38k missed hits, en `let's see` family, pt
`verificando`, comma-free `non non`/`não não`). Counts are **not comparable across
rounds** — never mix round-2 and round-3 numbers in one table.

---

## 4. Changes vs the previous version (summary)

| Area | Before | After |
|---|---|---|
| LC detector | whole-text `langdetect.detect()` | windowed fastText, renormalised {en,fr,pt,es} |
| LC granularity | whole text | per-section (think + answer), both must pass |
| LC label | binary | 3-way (1 / 0 / −1), ambiguous excluded from both pools |
| pt/es confusion | mislabels pt→es, poisons rejected | run-length + renorm; ~0% es-windows in pure pt |
| Format rule | `\boxed{}` within 800 chars of `</think>` | exactly-one `</think>` + len>1000 + boxed after |
| Format for chosen | soft `WEIGHT_FORMAT` bonus | hard requirement |
| Rejected selection | priority-fill (fr/pt LC-first) | quota mix, guarantee both types, length-matched LC |
| Mixture visibility | none | per-language pair-type counters in `stats.json` |
| Think tags | doubled `<think>` in training seq | `_dedup_leading_think` conditional strip |
| Backtrack regex | small old set | expanded + round-3 data-mined |
| Training loss | `loss_type="sigmoid"` | `["sigmoid","sft"]`, weights `[1,1]` (anchored) |

**Refactor bugs fixed along the way**: `n_answers = 3*n_answers` (index overflow) →
separate `n_answers_total`; `get_k` referenced deleted `N_ANSWERS` global → parameter;
difficulty log was overwriting the dataset file → separate `--difficulty_log_file`.

---

## 5. Training configuration (`dpo_train_no_precomp.py`)

The central training change is the **anchored loss** (already applied):

```python
loss_type=["sigmoid", "sft"],   # DPO + NLL-on-chosen (Iterative RPO / RPO family)
loss_weights=[1.0, 1.0],
```

**Why**: the SFT (NLL) term pins the chosen likelihood up unconditionally, so the
shared tokens of near-duplicate LC pairs get net-positive gradient and the
discriminative pressure concentrates on the tokens that actually differ (the
language). This is the direct fix for the likelihood displacement confirmed on W&B.

**Verify it worked**: `rewards/chosen` should stay ≥ 0 and `logps/chosen` should not
decline over training. If `rewards/chosen` still falls, the anchor is not effective.

Note: an anchor on **dirty** labels trains the model to imitate mislabelled
code-switched chosen — so the clean per-section LID labels (§3.2) and the anchor must
ship together, not separately.

Other config unchanged: QLoRA 4-bit, LoRA r=16, `beta` from CLI (RPO literature
suggests a lower beta ~0.01 is worth trying), 1 epoch, `max_length=16384`,
`truncation_mode="keep_end"`.

**Filename warning — it DOES precompute despite `_no_precomp`.** The script hardcodes
`precompute_ref_log_probs = True` (line ~173, passed to `DPOConfig`), so the name is
stale and means the opposite of the config. With `ref_model=None` on a LoRA model,
TRL derives the reference by **disabling the adapter** and, because precompute is on,
runs one full inference pass over the whole dataset up front to cache
`reference_{chosen,rejected}_logps`, then skips the reference path during training.
The `sft` term needs no reference, so precompute still correctly serves the `sigmoid`
term. To actually disable it, set `precompute_ref_log_probs = False` (ref computed
per step via adapter toggling — leaner memory upfront, slower steps). Either rename
the file or flip the flag so the two stop contradicting each other.

**Suggested ablation (one extra run, cheap, paper-worthy)**: old dirty dataset +
anchored loss only. Prediction: `rewards/chosen` stops falling and the LC regression
shrinks even without label fixes — decomposes the failure into displacement vs label
noise.

---

## 6. Why NOT iterative DPO (for now)

Iterative/on-policy DPO (regenerate answers from the improved policy each round,
rebuild pairs, retrain) was considered and **deliberately deferred**:

1. **It erodes MuLE's thesis.** MuLE's entire value proposition is achieving
   alignment at *a fraction of GRPO's compute*. The dominant cost is generation
   (N=8 × 3 languages × all questions with vLLM); each iteration pays it again. Two
   or three rounds approach GRPO's cost envelope, dissolving the comparison the paper
   is built on.
2. **It confounds the contribution being measured.** A single well-constructed round
   cleanly isolates what the *pair-construction framework* buys (labels, mixture,
   anchor). Multi-round policy drift mixes that with online-exploration dynamics,
   making it impossible to attribute the result to the offline design.
3. **It cannot cross the sampling-frontier barrier within the current question set.**
   Offline DPO — iterated or not — only sharpens behaviours the model can already
   sample. At hard tiers there are no correct in-language samples to promote
   (§1.4), so iteration yields diminishing returns *until harder questions and/or
   translated hard-tier chosen are added*. Compute is better spent on question
   difficulty and the anchor than on more rounds.
4. **Honest paper framing**: capability gains at hard tiers require online
   exploration (GRPO) — left explicitly as future work. MuLE's claim is narrower and
   defensible: it **fixes consistency cheaply** by getting the offline pair design
   right. That is a statement about what preference data *can contain*, and iterating
   the same data does not change it.

If iteration is revisited later, the preconditions are: harder source questions,
adaptive sampling (N=32+) on low-consensus questions, and translated-English→target
chosen for questions solved only in English (~430 fr/pt available in iter1).

---

## 7. Calibration & analysis scripts

- **`compare_lid.py`** — runs langdetect / lingua / fastText through the *same*
  windowed per-section decision logic over pure vs mostly-English buckets. Result:
  **fastText chosen** — best pt yield (81.5% vs lingua 79.5%), 0% es-window confusion
  in pure pt (vs 3.8%), fastest; all three catch 99–100% of confident negatives, so
  the decision logic dominates the detector choice. Per-section labelling costs only
  ~1–2% of yield, all of it moving to the *ambiguous* bucket (never to false
  negatives). Re-run after any threshold change: `--detectors fasttext`.
- **`format_stats.py`** — per-language decomposition of every format condition, plus
  the `</think>`→`\boxed{}` gap distribution that justified dropping the 800-char rule.

---

## 8. Open items / future work

- Add the anchored-loss **ablation run** and check the `rewards/chosen` curve.
- ~~Consider dropping `WEIGHT_BACKTRACK`~~ **Done** — replaced by the D2
  rep/uniq score (§3.4) per the DPO+GRPO implementation spec; bt is diagnostic-only.
- **D5 metric-reporting fixes** in the PolyMath statistics scripts: add avg rep
  and (for non-English) English-text fraction to summary tables; never aggregate
  backtracks across languages. Not yet applied (`polymath_data.py`).
- Acceptance checks still pending on the **full** dataset rebuild: reproduce the
  reference rep values per regime; review the D4 census; confirm chosen length
  shifts up while chosen rep stays <0.08; review `pairs_over_cap_frac` in
  stats.json (pairs exceeding the 16384-token trainer cap train on an amputated
  prompt under `truncation_mode="keep_end"` — the filter now counts them per side).
- **Harder questions + translated hard-tier chosen** — the only route to medium/high
  accuracy gains for an offline method.
- Confirm the **eval backend is pinned** (langdetect vs fastText in
  `cal-MMATH-acc.py`) and that all reported numbers used the same one.
- After regenerating pairs, sanity-check: 0% doubled think-tags; fr/pt mixture ≈
  60–70% accuracy-contrast / 25–35% LC-contrast; `pairs_*_other` only a few percent.

---

## Conventions
- `_strip_math` in `dataset_filter.py` must stay byte-identical to the one in
  `cal-MMATH-acc.py`.
- The filter's format/LC checks are the strict **intersection** of the eval scorer
  and the M-Thinker reward, since those two disagree with each other.
- Large model files (`lid.176.bin`) and `Datos/` are git-ignored.
