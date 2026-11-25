# Scam Detection AI Lab — model card

## Model snapshot

- Classifier: TF-IDF word and character n-grams + Logistic Regression.
- Additional detection: direct requests to disclose a one-time code, password, or card secret raise the threat score; a common ASR confusion of «код» with «кота» is handled when paired with a disclosure request.
- Target labels: `scam`, `legitimate`.
- Training rows: 120.
- Holdout rows: 40.
- Dataset: 160 synthetic examples after duplicate removal, balanced across classes and written in Russian and Kazakh.
- Data sources: 70 starter examples in `data/dialogues.csv` and 92 curated examples in `data/curated_training.csv` (some duplicate starter rows are removed before fitting).
- Languages represented: Russian and Kazakh.
- Artifact: `models/scam_model.joblib`.

## Current holdout results

| Metric | Result |
|---|---:|
| Accuracy | 85.0% |
| Precision (scam) | 85.0% |
| Recall (scam) | 85.0% |
| F1 (scam) | 85.0% |

Confusion matrix, with rows = actual and columns = predicted, order `legitimate`, `scam`:

| | Predicted legitimate | Predicted scam |
|---|---:|---:|
| Actual legitimate | 17 | 3 |
| Actual scam | 3 | 17 |

## Critical interpretation

These values are from a small, synthetic, hand-written demonstration set and a random stratified split. They are **not** an estimate of performance on real calls. The additions include a few transcription errors and safe counterexamples, but do not represent real speech, accents, background noise, all mixed-language conversations, new fraud scripts, or the variety of legitimate calls. Three scam examples were missed and three legitimate examples were flagged in this holdout.

The model is a software pipeline baseline, not a validated anti-fraud product. The score shown in the UI is not a calibrated probability. A missed scam and a false alarm are both possible.

## Required validation before a pilot

1. Collect data under an appropriate legal basis and with necessary participant notice/consent.
2. Remove or mask names, phone numbers, addresses, account identifiers, bank details, passwords and one-time codes.
3. Have at least two reviewers label ambiguous calls and resolve disagreements.
4. Keep all segments from one phone call in the same train or test split.
5. Build a separate test set from different people, dates, scam scripts and legitimate call types.
6. Report per-language, per-scenario false-positive and false-negative rates.
7. Test ASR errors and mixed Russian/Kazakh speech separately.

Do not use this model alone to block calls, accuse a person, or make financial or legal decisions.
