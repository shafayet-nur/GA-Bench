# GA-Bench: Code and Evaluation Pipeline

This repository contains the **code, prompts, job scripts, extraction utilities, and evaluation pipeline** associated with the GA-Bench project.

GA-Bench is a source-linked resource of **10,000 open-access paper–graphical-abstract pairs** designed for evaluating descriptive **IMRaD coverage** and **cross-section relation traceability** in scientific graphical abstracts.

## Related Resources

- **Dataset:** https://huggingface.co/datasets/shafayet217/GA-Bench
- **Dataset DOI:** https://doi.org/10.57967/hf/10514
- **Paper:** *GA-Bench: 10,000 Source-Linked Graphical Abstracts for Evaluating IMRaD Coverage* — accepted for presentation at the 2026 ACM/IEEE Joint Conference on Digital Libraries (JCDL 2026)

For dataset contents, metadata fields, statistics, licensing, annotations, and responsible-use information, please refer to the Hugging Face dataset repository.

For details on dataset construction, annotation, evaluation methodology, experimental setup, results, limitations, and reproducibility, please refer to the companion JCDL 2026 paper.

---

## Repository Structure

```text
GA-Bench/
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
│       └── task1_scoring.py
├── extraction/
│   └── parser_grobid/
│       └── pbs/
└── README.md
```

---

## Folder Guide

### `completeness/`

Contains the code, prompts, job scripts, pipeline stages, and scoring utilities used for the graphical-abstract completeness evaluation workflow.

### `completeness/direct_prompt_baseline/`

Contains the implementation of the direct-prompt baseline used for comparison with the Structured Reference Profile pipeline.

- `codes/` — Python scripts for the direct-prompt baseline
- `pbs/` — PBS job scripts for running the baseline on an HPC system
- `prompts/` — prompt templates used by the baseline

### `completeness/main_pipeline/`

Contains the main **Structured Reference Profile (SRP)** evaluation pipeline.

- `prompts/` — prompt templates used by the main pipeline
- `stage1/` — Stage 1 implementation for constructing Structured Reference Profiles
- `stage2/` — Stage 2 implementation for graphical-abstract grounding and completeness evaluation

The evaluation pipeline uses two source-paper input variants:

- **Variant A:** title, abstract, and IMRaD sections
- **Variant B:** Variant A inputs plus paper figures and tables

### `completeness/scoring_codes/`

Contains scripts used for scoring, validation, agreement analysis, and comparison of model-generated and human annotations.

Included scripts:

- `task1_coverage_breakdown.py`
- `task1_human_relation_validation.py`
- `task1_inter_annotator_validation.py`
- `task1_naive_vs_human.py`
- `task1_relation_agreement_tables.py`
- `task1_scoring.py`

### `extraction/`

Contains code related to scholarly-document extraction and preprocessing.

### `extraction/parser_grobid/`

Contains the GROBID-based document parsing workflow and related files.

- `pbs/` — PBS job scripts for running the extraction pipeline on an HPC system

---

## Usage Notes

Before running the scripts:

- Update local input and output paths as needed.
- Update model paths and environment-specific settings.
- Ensure that the required model and Python dependencies are installed.
- Adapt PBS scripts to the scheduler and computational resources available on the target HPC system.
- Review file paths carefully because some scripts may contain environment-specific directory configurations.

The provided PBS files reflect the computing environment used during development and may require modification for other systems.

---

## Dataset

The associated **GA-Bench dataset** is available on Hugging Face:

**Dataset repository:**  https://huggingface.co/datasets/shafayet217/GA-Bench

**Persistent DOI:**  https://doi.org/10.57967/hf/10514

The dataset contains the source-linked paper–graphical-abstract pairs, bibliographic metadata, extracted scholarly-document components, human annotations, and model-generated completeness outputs used with the code in this repository.

---

## Paper

**GA-Bench: 10,000 Source-Linked Graphical Abstracts for Evaluating IMRaD Coverage**

**Authors:** Shafayet Nur, Adiba Ibnat Hossain, Maliha Zahan Chowdhury, and Hamed Alhoori

**Venue:** 2026 ACM/IEEE Joint Conference on Digital Libraries (JCDL 2026)

The paper has been **accepted for presentation at JCDL 2026**.

For details on dataset construction, annotation, evaluation methodology, experimental setup, results, limitations, and reproducibility, please refer to the companion paper.

---

## Citation

### Code Repository

If you use the code or evaluation pipeline, please cite the software repository:

```bibtex
@software{nur2026gabenchcode,
  author  = {Nur, Shafayet and Hossain, Adiba Ibnat and Chowdhury, Maliha Zahan and Alhoori, Hamed},
  title   = {{GA-Bench}: Source Code and Evaluation Pipeline},
  year    = {2026},
  version = {1.0},
  url     = {https://github.com/shafayet-nur/GA-Bench}
}
```

### Dataset

Please also cite the GA-Bench dataset when using the released data:

```bibtex
@dataset{nur2026gabenchdataset,
  author    = {Nur, Shafayet and Hossain, Adiba Ibnat and Chowdhury, Maliha Zahan and Alhoori, Hamed},
  title     = {{GA-Bench Dataset: 10,000 Source-Linked Graphical Abstracts for Evaluating IMRaD Coverage}},
  year      = {2026},
  publisher = {Hugging Face},
  version   = {1.0},
  doi       = {10.57967/hf/10514},
  url       = {https://huggingface.co/datasets/shafayet217/GA-Bench}
}
```

### Paper

The companion paper can currently be cited as:

```bibtex
@unpublished{nur2026gabench,
  author = {Nur, Shafayet and Hossain, Adiba Ibnat and Chowdhury, Maliha Zahan and Alhoori, Hamed},
  title  = {{GA-Bench: 10,000 Source-Linked Graphical Abstracts for Evaluating IMRaD Coverage}},
  note   = {Accepted for presentation at the 2026 ACM/IEEE Joint Conference on Digital Libraries (JCDL 2026)},
  year   = {2026}
}
```

The paper citation will be updated with the official proceedings information and DOI after publication.
