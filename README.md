# GA-Bench Code

This repository contains the code associated with the GA-Bench project.

Dataset repository:  
https://huggingface.co/datasets/ga-bench/GA-Bench

## Repository structure

```text
GA-Bench-Code/
├── completeness/
│   ├── direct_prompt_baseline/
│   │   ├── codes/
│   │   ├── pbs/
│   │   └── prompts/
│   ├── main_pipeline/
│   │   ├── prompts/
│   │   ├── stage1/
│   │   └── stage2/
│   └── scoring_codes/
│       ├── task1_coverage_breakdown.py
│       ├── task1_human_relation_validation.py
│       ├── task1_inter_annotator_validation.py
│       ├── task1_naive_vs_human.py
│       ├── task1_relation_agreement_tables.py
│       ├── task1_scoring_copy.py
│       └── task1_scoring.py
├── extraction/
│   └── parser_grobid/
│       └── pbs/
└── README.md
```

## Folder guide

### `completeness/`

Contains the code, prompts, job files, pipeline stages, and scoring scripts used for the completeness-related workflow.

#### `completeness/direct_prompt_baseline/`

- `codes/` — Python code for the direct-prompt baseline.
- `pbs/` — PBS job scripts for running the baseline on an HPC system.
- `prompts/` — prompt files used by the baseline.

#### `completeness/main_pipeline/`

- `prompts/` — prompt files used by the main pipeline.
- `stage1/` — Stage 1 implementation files.
- `stage2/` — Stage 2 implementation files.

#### `completeness/scoring_codes/`

Contains scoring, validation, agreement, and comparison scripts:

- `task1_coverage_breakdown.py`
- `task1_human_relation_validation.py`
- `task1_inter_annotator_validation.py`
- `task1_naive_vs_human.py`
- `task1_relation_agreement_tables.py`
- `task1_scoring_copy.py`
- `task1_scoring.py`

### `extraction/`

Contains the document-extraction code.

#### `extraction/parser_grobid/`

Contains the GROBID-based parser implementation and its related files.

- `pbs/` — PBS job scripts for running the extraction pipeline on an HPC system.

## Usage notes

- Update local input, output, model, and environment paths before running the scripts.
- PBS scripts may need to be adapted to the resource requirements and scheduler configuration of the target cluster.
- Keep datasets, model weights, generated outputs, logs, access tokens, and environment-specific files outside the Git repository unless they are intentionally part of the code release.

## Dataset

The associated GA-Bench dataset is available on Hugging Face:

https://huggingface.co/datasets/ga-bench/GA-Bench

## Citation

```bibtex
@dataset{nur2026gabench,
  author       = {Nur, Shafayet and Hossain, Adiba Ibnat and Chowdhury, Maliha Zahan and Alhoori, Hamed},
  title        = {{GA-Bench: 10,000 Source-Linked Graphical Abstracts for Evaluating IMRaD Coverage}},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/datasets/ga-bench/GA-Bench},
  note         = {Dataset and accompanying research artifact}
}
```
