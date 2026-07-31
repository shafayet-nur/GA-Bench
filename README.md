# GA-Bench

Code for **GA-Bench: 10,000 Source-Linked Graphical Abstracts for Evaluating IMRaD Coverage**.

This repository contains the extraction, dataset-construction, prompting, completeness-evaluation, scoring, and analysis pipeline used for GA-Bench.

The corresponding dataset is hosted on Hugging Face:

**https://huggingface.co/datasets/ga-bench/GA-Bench**

## Repository structure

- `extraction/` — PDF-to-structured-content pipeline using GROBID and pdffigures2. It produces full text, IMRaD-structured text, figures, tables, equations, extracted figure images, and extraction-quality reports.
- `dataset_build/` — scripts for dataset filtering, sampling, statistics, metadata preparation, path generation, validation, and release packaging.
- `prompts/` — prompt builders and schemas for Structured Reference Profile construction, GA grounding, and direct-prompt baselines.
- `task1_completeness/` — completeness-evaluation pipeline, including:
  - Structured Reference Profiles for Variants A and B
  - GA-grounding verdicts
  - direct-prompt baselines
  - section-level entity judgments
  - cross-section relation judgments
  - deterministic completeness scoring
  - failure handling and resumable inference
- `analysis/` — scripts for aggregating outputs, comparing model configurations, evaluating agreement with human annotations, and generating analysis-ready tables.
- `pbs/` — PBS job scripts for running the pipeline on an HPC cluster.

## Dataset

GA-Bench contains 10,000 open-access paper–graphical-abstract pairs. Each record links a publisher-hosted graphical abstract to its source article, bibliographic metadata, extracted full text, IMRaD sections, figures, tables, equations, extraction-quality information, human-annotation values, and model-generated completeness outputs.

Dataset repository:

**https://huggingface.co/datasets/ga-bench/GA-Bench**

The Hugging Face dataset includes:

- graphical abstract images
- source article PDFs
- extracted full text
- IMRaD-structured text
- figure, table, and equation extraction outputs
- extracted figure images
- extraction-quality reports
- per-paper completeness outputs
- metadata in Parquet and Excel formats
- selected Qwen-B completeness values
- two sets of human-annotation values embedded in the metadata files

## Completeness evaluation

The completeness task evaluates how well a graphical abstract represents the source paper's IMRaD narrative:

- Introduction
- Methods
- Results
- Discussion

The pipeline also evaluates three cross-section relations:

- Introduction → Methods
- Methods → Results
- Results → Discussion

### Stage 1: Structured Reference Profile

A Structured Reference Profile is generated from the source paper. It includes section summaries, paper-grounded entities, candidate visual proxies, and cross-section relations.

Two input variants are supported:

- **Variant A:** title, abstract, and IMRaD text
- **Variant B:** Variant A plus source-paper figures and tables

### Stage 2: GA grounding

The graphical abstract is evaluated against the corresponding Structured Reference Profile.

The model produces:

- entity judgments: `explicit`, `implied`, or `absent`
- relation judgments: `traceable`, `partially traceable`, or `not traceable`
- a discrete completeness level from 0 to 4

### Direct-prompt baseline

The direct-prompt baseline receives the same source-paper inputs but predicts the IMRaD component labels and completeness level without first constructing a Structured Reference Profile.

## Models

The evaluation pipeline supports three open vision–language models:

- Qwen3-VL-32B-Instruct
- Gemma-3-27B-IT
- Mistral-Small-3.1-24B-Instruct

Models are served locally through a vLLM OpenAI-compatible endpoint.

## Environment and execution

- The scripts were developed for an HPC environment using PBS.
- Update all dataset, model, output, and environment paths before running.
- PBS resource requests should be adapted to the target cluster.
- The inference scripts use a local vLLM OpenAI-compatible endpoint.
- Long-running jobs support resumable processing and failure manifests where applicable.
- Model weights, dataset files, generated outputs, access tokens, logs, and cluster-specific temporary files should not be committed to GitHub.

## Recommended configuration steps

1. Create or activate the required Python environment.
2. Install the dependencies required by the selected pipeline component.
3. Update local path settings in the relevant configuration or job file.
4. Start the required vLLM model server.
5. Submit the corresponding PBS job.
6. Validate the generated outputs and failure manifests.
7. Run the scoring and aggregation scripts.

## Reproducibility notes

- Keep model names, variants, decoding settings, image limits, and prompts unchanged when reproducing reported comparisons.
- Preserve paper identifiers and repository-relative paths when joining outputs with the Hugging Face metadata.
- The `id` field in the dataset metadata is the filesystem-safe DOI and matches the paper folder name.
- The Hugging Face `completeness/` folders contain the released model-generated SRPs, GA-grounding verdicts, direct-prompt outputs, and related completeness annotations.

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
