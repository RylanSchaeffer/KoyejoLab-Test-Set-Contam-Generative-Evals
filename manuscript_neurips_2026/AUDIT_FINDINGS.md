# NeurIPS 2026 Submission Audit: Open Findings

Paper: "Quantifying the Effect of Test Set Contamination on Generative Evaluations"

Status date: 2026-05-06, America/Los_Angeles

This file intentionally lists only findings that still require author confirmation, source edits, or final-mode review.

## Submission Blocker

### B1. Checklist Has Three Remaining TODO Answers

Status: blocker.

Evidence:

- `checklist.tex:22`, `checklist.tex:42`, and `checklist.tex:102` still use `\answerTODO{}`.
- The unresolved questions are experimental reproducibility, open access to data/code, and new assets.
- These TODO answers render as `[TODO]` in the checklist pages.

Required action:

- Replace each remaining `\answerTODO{}` with the strongest defensible final macro after author confirmation.
- Confirm exact reproduction commands, environment or software versions, seeds or single-run convention, data-preparation scripts, figure-generation scripts, release links, checkpoint/model access, and new-asset documentation or model cards.
- Rebuild the PDF and confirm no `[TODO]` remains in the rendered checklist.

## Final And Camera-Ready Items

### F1. Competing-Interests Disclosure Is Missing

Status: final/camera-ready issue.

Evidence:

- `07_acknowledgements.tex:14-16` lists funding/support.
- `07_acknowledgements.tex:18-19` has a TODO for the NeurIPS competing-interests disclosure.
- The `ack` environment is hidden in the current anonymous submission build but will appear in final mode.

Required action:

- Add an explicit competing-interests disclosure inside `\begin{ack}...\end{ack}` before final/camera-ready.

### F2. Workshop Note And Author Block Need Final-Mode Review

Status: final-mode presentation issue.

Evidence:

- `07_acknowledgements.tex:5-12` contains a TODO about whether the prior-workshop note should be kept, rephrased, or removed until camera-ready.
- `neurips_2026.tex:77-79` and `neurips_2026.tex:109-111` duplicate the corresponding-author emails in both `\thanks` text and the address block.

Required action:

- In final/camera-ready mode, review the workshop-note wording and remove any unwanted duplicate author-block email presentation.
