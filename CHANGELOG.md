# 📋 Changelog — AI Business Process Intelligence

All notable changes to this project are documented in this file.

---

## [2.1.0] — 2026-02-11

### 🧹 Project Cleanup

**Deleted 13 useless files** to clean the repository, freeing ~4.2 GB of disk space.

#### Deleted — Superseded Scripts
| File | Reason |
|------|--------|
| `salesforce-generator.py` | Old TAWOS pipeline; replaced by SAP-based generators |
| `teams-generator.py` | Replaced by `generate_teams_from_sap.py` |
| `prepare_training_data.py` | Replaced by `unify_datasets.py` (produces `training_data.csv` directly) |
| `extract_from_kaggle.py` | One-off Kaggle download script; project now uses BPI 2019 XES |

#### Deleted — Old Pipeline Data
| File | Size | Reason |
|------|------|--------|
| `TAWOS.sql` | 4.2 GB | Old TAWOS SQL database dump, unreferenced |
| `real_jira_data.csv` | 22 MB | Old TAWOS Jira dataset |
| `unified_jira_data.csv` | 4.2 MB | Old intermediate file from `salesforce-generator.py` |

#### Deleted — Stale Metadata & Artifacts
- `dataset-metadata.json` — Kaggle metadata for old TAWOS dataset
- `CHANGELOG.md.resolved` — Git merge conflict artifact
- `implementation_plan.md.resolved` — Git merge conflict artifact
- `task.md.resolved` — Git merge conflict artifact
- `__pycache__/` — Python bytecode cache
- `21308124/` — TAWOS dataset README leftover

#### Updated — Stale Default Parameters
Three scripts had default file paths pointing to deleted `unified_jira_data.csv`. Updated to `synthetic_jira_data.csv`:
- `worker_data.py` → `generate_worker_assignments(jira_csv='synthetic_jira_data.csv')`
- `simulation_engine.py` → `jira_file='synthetic_jira_data.csv'`
- `dependency.py` → `generate_dependencies(input_csv='synthetic_jira_data.csv')`

Also fixed stale error message in `worker_data.py` referencing deleted `salesforce-generator.py`.

---

## [2.0.0] — 2026-02 (SAP Pipeline Migration)

### 🔄 Complete Data Pipeline Overhaul

Migrated from the TAWOS/Kaggle Jira dataset to real SAP procurement data (BPI Challenge 2019).

#### New Pipeline Architecture
```
BPI_Challenge_2019.xes (728 MB, real SAP data)
    ├── parse_sap_xes.py        → sap_event_log.csv
    ├── generate_jira_from_sap.py   → synthetic_jira_data.csv
    ├── generate_teams_from_sap.py  → synthetic_teams_data.csv
    └── unify_datasets.py       → unified_master.csv + training_data.csv
```

#### New Files
| File | Purpose |
|------|---------|
| `parse_sap_xes.py` | Stream-parses 728 MB XES file into CSV with case IDs, activities, timestamps, values |
| `generate_jira_from_sap.py` | Maps SAP purchase orders → synthetic Jira tickets (type, priority, assignee, domain) |
| `generate_teams_from_sap.py` | Generates Teams chatter linked to SAP POs (sentiment-aware, noise-based) |
| `unify_datasets.py` | Merges SAP + Jira + Teams into `unified_master.csv`, also outputs `training_data.csv` |
| `worker_data.py` | Generates worker profiles (skills, speed, quality) and ticket assignments |

---

## [1.2.0] — 2026-01-29

### 🚀 AI Performance Optimization

Achieved **+520% improvement** over random ticket prioritization through 4 iterations of environment and training optimization.

#### Environment Improvements (`custom_env.py`)

| Version | Change | Result |
|---------|--------|--------|
| v1 (Original) | Raw observation values (0–250k), simple reward | AI performed **-9%** worse than random |
| v2 | Normalized observations to 0–1 range | **+65%** improvement |
| v3 | Reward shaping with opportunity cost penalties | **+100%** improvement |
| v4 (Final) | Sorted backlog + rank-based aggressive rewards | **+520%** improvement 🏆 |

**Key changes in v4:**
- Backlog sorted by value (highest first) → position encodes rank
- Rank 0 (best): reward = `10.0 + (value/max) * 5.0`
- Rank 1: reward = `2.0`
- Rank 2: reward = `0.0`
- Rank 3+: reward = `-5.0 * rank` (heavy punishment)

#### Training Improvements (`train_agent.py`)

| Parameter | v1 | Final |
|-----------|----|----|
| `total_timesteps` | 20,000 | **200,000** |
| Network | 64×64 | **256×256** |
| `n_steps` | 2048 | **4096** |
| `batch_size` | 64 | **256** |
| `n_epochs` | 10 | **20** |
| `ent_coef` | 0.01 | **0.005** |
| Activation | ReLU | **GELU** |

#### Performance Results

| Agent | Revenue | Improvement |
|-------|---------|-------------|
| Random Baseline | $1,773,251 | — |
| **PPO AI Agent** | **$10,996,205** | **+520%** |

---

## [1.1.0] — 2026-01-29

### 🔧 Critical Bug Fixes

#### Windows Encoding Errors
- Replaced all Unicode emoji with ASCII text (`📂` → `[INFO]`, `✅` → `[SUCCESS]`, `❌` → `[ERROR]`)
- Affected: `simulation_engine.py`, `teams-generator.py`, `dependency.py`

#### CSV Date Parsing Failures
- Added quote-stripping before `pd.to_datetime()` in `simulation_engine.py`
- Fixed embedded quotes from `quoting=QUOTE_NONE` CSV loading

#### Empty DataFrame Merge Crash
- Added defensive check before merge when `mapping_df` is empty

#### Missing Dependencies
Installed: `simpy`, `gymnasium`, `stable-baselines3`, `shimmy`, `streamlit`, `plotly`

---

## [1.0.0] — 2026-01 (Initial Release)

### 🎯 Initial Project Setup

- Streamlit dashboard (`app.py`) with live AI vs Random showdown
- PPO agent training pipeline (`train_agent.py`, `custom_env.py`)
- SimPy-based ticket lifecycle simulation (`simulation_engine.py`)
- Synthetic data generators for Jira, Teams, Salesforce, and dependencies
- AI performance benchmarking (`test_ai_performance.py`)

---

## Dependencies

```
pandas
numpy
faker
simpy
gymnasium
stable-baselines3
shimmy
torch
streamlit
plotly
```
