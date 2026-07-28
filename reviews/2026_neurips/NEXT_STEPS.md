# NeurIPS 2026 Rebuttal — Prioritized Next Steps

Actionable checklist. Rationale and reviewer mapping live in [`REBUTTAL_PLAN.md`](REBUTTAL_PLAN.md).

Scores: **8RFz = 3** (conf 4, Quality 2, Originality 2) · **1wx9 = 4** (conf 4) · **aPBL = 3** (conf 3).
AC named 8RFz's loss-vs-correctness objection as the pivotal critique.

**Cluster:** `snapskampere1` → `/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization`,
`source mem_scoring_vs_sampling_env/bin/activate` (uv venv). See [`docs/INFRASTRUCTURE.md`](../../docs/INFRASTRUCTURE.md).

---

## Do first — blocking or long wall-clock

- [ ] **1. Launch Math Verify on the 138 overtrained checkpoints.** *The* rebuttal experiment; inference
      only, no training. τ=0 alone is ~14 GPU-h; all three temperatures ~41 GPU-h.
      ```bash
      python scripts/audit_inventory.py --gaps-only   # regenerate the exact target list
      # then write sweeps/eval_pt/math/model=qwen3-{34,62,93,344}M-{2,4,8,16}xOT.yaml
      # from the ot_1 files, swapping the ot_N model names
      ```
      Worth 30 min first: make `eval_language_model.py` loop temperatures inside one vLLM load
      (startup dominates → ~3× cheaper).

- [ ] **2. Locate the pass@k artifacts on skampere1.** Only item with real data-loss risk.
      `scripts/score_pass_at_k.py` writes `results.csv` / `summary.md`; neither is on the workstation.
      ```bash
      find /lfs/skampere1/0/rschaef -path '*pass_at_k*' -name '*.csv' 2>/dev/null
      ```

- [ ] **3. Format sanity check before building on the SFT result.** Confirm the flat ~1–2% post-SFT
      Math Verify floor is genuine failure, **not** a collapsed `\boxed{}` emission rate. Pull the
      `response` column for sweep `2zpwcnek` and measure parse rate vs. the pretrained runs. If the
      format rate dropped, the number is an artifact and item 4 changes character. **Do this before
      writing any rebuttal text.** Applies to item 1's results too.

## Zero compute — parallelizable across co-authors

- [ ] **4. Fold the SFT Math Verify result into the paper.** Data already exists (sweep `2zpwcnek`,
      `notebooks/13_*`); manuscript currently cites only the loss version. Post-SFT accuracy is flat
      ~1–2% across all contamination levels vs. ~100% at R ≥ 316 pretrained — a ~60× collapse, and it
      rules out the stealth-contamination corollary. Answers 8RFz Q1 (SFT half).
- [ ] **5. Put pass@k in the paper** (0.000% / 808,797 samples, uncontaminated 344M). Zero-capability
      baseline; reframes "small models" from weakness to design feature.
- [ ] **6. Bootstrap CIs on every Math Verify number.** Per-problem scores are already in W&B history.
      Say plainly this is not multi-seed; commit to seeds for camera-ready.
- [ ] **7. Promote `notebooks/16_*` (SFT → perturbed MATH, 14/17) to a figure.** Currently one prose
      clause at `04_further_training.tex:64`.
- [ ] **8. Add an "Original" column to Table 1** (`03_pretraining.tex:14–67`) so the ~100% → ~0%
      collapse is legible without cross-referencing Fig. 1.
- [ ] **9. Normalize Fig. 5 as `score(τ)/score(τ=0)`** to kill 8RFz's temperature confound. Restrict
      "truth serum" claims to τ ≤ 1 and concede τ ∈ {1.29, 1.5} degrades everything.
- [ ] **10. Bootstrap/profile CIs on E(0) + alternative functional forms.** Make explicit that
      contaminated losses are *measured*; only the asymptote is extrapolated.
- [ ] **11. Report contaminated-token fraction per replica count.** Pure arithmetic; answers "contrived."

## Writing only

- [ ] **12. Fix the originality framing.** Delete "the first targeted examination of contamination in
      generative tasks" (`06_discussion.tex:4`) — indefensible given you cite Kocyigit et al. 2025.
      Replace with the full-lifecycle claim. Add Palavalli 2024, Mehrbakhsh 2024, both Dekoninck 2024,
      Godey 2025 (**none are in `references_rylan.bib`**); engage Jiang 2024. Add an explicit
      replicates/conflicts paragraph, reconciled via "models trained from scratch have no competence to
      bridge surface form" — with pass@k = 0 as direct evidence. Best shot at 8RFz's two low sub-scores.
- [ ] **13. SFT hyperparameters appendix.** aPBL W4; straight concession.
- [ ] **14. Rephrase/perturbation validation appendix.** Mostly a port from
      `reviews/2026_icml/REVIEWER_6RQA/` — QC audits and 4 spot-check batches already written.
- [ ] **15. Clarify what Table 1 tested** (exact contamination + modified eval, *not* rephrased
      contamination). 1wx9's W2 infers a symmetry never tested.
- [ ] **16. Lifecycle summary figure** — pretrain dose → overtrain → SFT → inference, in Math Verify
      space. Possible only after item 1.
- [ ] **17. Concede scope explicitly** and commit to ≥1B + second family + code benchmark for
      camera-ready. All three reviewers raised it; arguing costs credibility.

## Upside — not requested, most likely to move a score

- [ ] **18. Turn temperature response into a contamination detector.** The τ-response differential
      separates contaminated from uncontaminated models without corpus access. Report separability
      over data already collected.

## Stretch — decide end of day 2, only promise if it will land

- [ ] **19. Paraphrased contamination during pretraining.** Most-requested experiment (1wx9 Q1,
      aPBL Q1, AC bullet 1). Cheaper than it feels: 34M × R ∈ {32,100,316} ≈ 680M tokens each,
      **~10–25 GPU-h total**. Needs ~1 h of code first — wire `load_dataset_math_rephrased()` into the
      dispatch at `src/data.py:442`.
- [ ] **20. GSM8K contamination mini-sweep** at 34M/62M. Loaders exist; `create_dataset_for_pretraining`
      has never been run against GSM8K, so budget debugging.
- [ ] **21. Seeds at pivotal configs** — 34M/93M × R ∈ {0,1,10,100}. Item 6 covers the stated concern.

## Explicitly deferred to camera-ready

≥1B models · second architecture family · code benchmark · full leakage-mode grid · multi-seed across the
full grid · resolving Huang et al. / Hayes et al. empirically.

---

## Repo hygiene surfaced during the audit

- [ ] **Yegor Denisov-Blanch is missing from `manuscript_neurips_2026/00_main.tex`.** OpenReview lists 12
      authors; the LaTeX block has 11. Would ship a missing author in camera-ready. **Fix before anything
      else touches that file.**
- [ ] `memorization-scoring-vs-sampling-pt` does not exist on W&B but is referenced 16× (`notebooks/10_*`,
      `sweeps/pt/*.yaml`). Repoint at `-pt-v2` or document the rename.
- [ ] Replace the `sshpass` plaintext-password aliases in `~/.bashrc` with SSH keys.
- [x] Duplicate `load_dataset_math_rephrased` in `src/data.py` — removed.
- [x] README pointed at a nonexistent `manuscript/` directory (broken images) and a nonexistent
      `gen_contam_env` — fixed.
