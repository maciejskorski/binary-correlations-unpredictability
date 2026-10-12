# Bridging Binary Correlations and Unpredictability

Can small correlations certify high unpredictability?
This repo accompanies *Bridging Binary Correlations and Unpredictability: Sharp Bounds and Insights for TRNG Validation* (Skorski & Günlü).

**Supplementary material (all here):**
- **Extended version** (`main.pdf`): the paper with all appendices, i.e. full proofs of the sharp bounds, the sample bridge, and the finite-sample certificate for $h_\infty\ge0.99$.
- **Submission version** (`submission.pdf`): the 5+1-page conference paper.
- **Code** (`verification/`): scripts and JSON results for the fair-bit calibration (400 samples at $N=10^5$), the noisy-parity over-rejection runs, and the certificate sizes. Run, e.g., `python3 verification/calibration.py`.
- **Source** (`main.tex`, `references.bib`): the LaTeX sources, built as in `build-both.sh`.
