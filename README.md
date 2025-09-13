# EMERGE-REPLICATE

*Last updated: 14/09/2025*

Replicating the results and methodology of the paper **“EMERGE: Enhancing Multimodal Electronic Health Records Predictive Modeling with Retrieval-Augmented Generation.”**
This repository provides data preprocessing notebooks, an experimentation pipeline, and utility functions to reproduce the core ideas.

> **Datasets**: We use **MIMIC-III / MIMIC-IV** (credentialed access via PhysioNet) and **PrimeKG** (publicly available). You must obtain and manage access to these resources in accordance with their licenses and data use agreements.

---

## Repository Structure

```
/preprocess-zhu/        # Raw MIMIC preprocessing scripts (choose III or IV)
preprocess-full.ipynb   # End-to-end data preprocessing notebook
pipeline.ipynb          # Experiment pipeline (training/evaluation)
utils/                  # Helper functions used across notebooks
```

---

## Quick Start

### 1) Raw data preparation

Navigate to `/preprocess-zhu/` and select the MIMIC version you plan to use (III or IV).
Follow the guidance in that folder’s README to generate the intermediate artifacts. No code changes are typically required.

### 2) Data preprocessing

Run `preprocess-full.ipynb` to build the processed datasets and features required by the pipeline.

* Configuration cells in the notebook let you adjust paths and options.
* Additional helper functions live in `/utils/`.

### 3) Experiments

Run `pipeline.ipynb` to reproduce the baseline pipeline.
Some customization is still limited; you may need to add minor adjustments for your specific setup or experiments.