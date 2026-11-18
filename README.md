# Administrative complexity of government documents

Code to measure the administrative complexity of official documents along fifteen
dimensions of learning, compliance and uncertainty demands, and to test the measure
against estimates of compliance time. Twelve dimensions are computed with rule-based
text analytics; three are assessed by an open-weights language model against a fixed
rubric.

## Layout

| Path | Contents |
|---|---|
| `src/adminplex/collect/` | Collectors for each document source and for the burden estimates |
| `src/adminplex/clean/` | Normalisation, boilerplate removal and deduplication |
| `src/adminplex/features/` | Rule-based features, lexicons, readability indices, the model rubric (`llm.py`) and the editing instructions (`simplify.py`) |
| `src/adminplex/index/` | Dimension scores and the candidate aggregations |
| `src/adminplex/analysis/` | Linkage between texts and burden estimates |
| `scripts/` | Pipeline steps and analyses |
| `slurm/` | Cluster jobs for collection, features and model assessment |
| `config/default.yaml` | Every parameter that affects a result |
| `tests/` | Test suite for the measurement functions |

## Environments

Python 3.11. The scoring stack and the analysis stack need separate environments:

| Requirements | Used for |
|---|---|
| `requirements/analysis.txt` | Collection, cleaning, features, index and analysis |
| `requirements/scoring-vllm.txt` | Model assessment with vLLM (GPU) |
| `requirements/scoring-hf.txt` | Model assessment with transformers (GPU) |

## Data

The data files are distributed separately. Place them at these paths:

    data/processed/corpus.parquet                        documents (identifier, source, metadata, text)
    data/processed/validation_collections.parquet        burden estimates per information collection
    data/processed/validation_parts.parquet              burden estimates per regulatory part
    data/processed/validation_collections_scoring.parquet
    data/processed/robustness_sample.parquet
    data/processed/simplified_for_scoring.parquet
    data/interim/features.parquet                        rule-based features
    outputs/llm_scores/*.jsonl                           model assessments
    outputs/llm_scores_simplified/*.jsonl
    outputs/simplify/*.jsonl                             edited documents
    outputs/external/collections/*.parquet               collection descriptions on the corpus scale
    outputs/tables/                                      result tables
    coding/ratings_*.csv, coding/sample_key.csv          human coding

## Reproducing the results

`scripts/run_all.sh` runs the pipeline in order: the test suite, cleaning, features,
validation frames, model assessment, index construction, burden validation,
descriptive analyses, the editing experiment, the robustness runs and the figures.
The main steps are:

    python -m adminplex.clean.normalise
    sbatch slurm/10_features.sbatch
    python scripts/prepare_validation.py
    sbatch slurm/20_score_dimensions.sbatch
    python scripts/build_index.py --tag main
    python scripts/validate_burden.py --level parts
    python scripts/validate_burden.py --level collections
    python scripts/run_experiments.py

Further analyses:

    python scripts/validate_burden_extended.py      # functional form, body level, clarity
    python scripts/robustness_suite.py              # alternative analytical choices
    python scripts/program_size_checks.py           # program size and length
    python scripts/check_generalisation.py          # leave-one-jurisdiction-out, language
    python scripts/analyse_trends.py                # change over time
    python scripts/analyse_robustness.py            # stability across runs, rubrics and models
    python scripts/analyse_simplification.py --max-original-words 1000
    python scripts/analyse_human_coding.py --coding coding
    python scripts/extended_validation.py           # within-agency models, inference, trends by subperiod
    sbatch slurm/42_score_global.sbatch             # global model rating (benchmark)
    python scripts/global_rating_benchmark.py       # benchmark and human-validation intervals
    python scripts/sensitivity_checks.py            # context, counts, language, excerpting, profiles
    python scripts/criterion_stability.py           # stability of the estimates between approvals

Every result can then be checked against its source table:

    python scripts/verify_reported_results.py
    python scripts/recompute_untabulated_results.py

A single seed (20260726) governs every sample, split and bootstrap. Model
assessment runs at temperature zero with the model identifiers in `config/` and
`slurm/`.

## Scoring new documents

Percentiles are relative to a reference corpus, so new documents should be placed
in its distribution rather than ranked among themselves:

    python scripts/measure_external.py --input new_documents.parquet \
        --scores <assessment folder> --tag <assessment tag> --out <output folder>

## Licence

Apache License 2.0 (see `LICENSE`).
