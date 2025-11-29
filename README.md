# AI-ENGINEER-ASSIGNMENT-SDGE-

# AI Engineer Assignment – SDGE

## Project Overview

Please see: ***docs/masternotebook.pdf*** for full rationale and breakdown of dataset/solution.

### TCGA-OV Cohort

The TCGA Ovarian Serous Custadenocarcinoma (TCGA-OV) cohort is one of the most comprehensive datasets available for molecular, genetic expression, clinical and phenotypes at patient level. The data set includes multi-omic assays curated through the NCI Genomic Data Commons (GDC). 

These datasets were downloaded using the UCSC Xena Browser (associated download scripts, as described in the project README.)

The TCGA-OV dataset is well suited to perform Synthetic Data Generation (SDGE), with rich multi-omic hetoerogentiy and good quality clinical meta data. 
Generating high-fidelity synthetic biomedical data serves many practical and scientific motivations:

- (1) Privacy Preservation of data sharing 
- (2) Augment smaller sample biomedical data 
- (3) Hypothesis exploration (gene/gene correlation structures etc.)

This notebook explores the following workflow of SDGE via a **diffusion model** on the TCGA-OV dataset:

- preprocessing of data
- contstucting a model-ready feature matrix 
- training diffusion-based generative model
- evaluating and comparision of synthetic data 

Generated data will be tested for statistical fidelity, correlation and similarity metrics to original data. 

## Why Diffusion Instead of GAN

Tabular cancer data mixes continuous genetic data with categorical clinical variables. Diffusion models handle mixed tupes naturally through score-based denoising compared to GAN.
Gan's may collapse and ignore minoraity classes, while diffusion models improve on training stability, coverage, and will synthesize better data. 


### Choosing DDPM instead of DDIM

DDPM denoising better captures biological variablilty due to its fully stochastic nature. DDIM's determinsitic sampling trades diversity for speed. for Heterogeneous genomics, DDPM preserves richer distributiuonal structure, synthesizing better data. 


## 1. Environment Setup
```bash
git clone https://github.com/scottpenco/AI-ENGINEER-ASSIGNMENT-SDGE-.git
cd AI-ENGINEER-ASSIGNMENT-SDGE

pip install -r requirements.txt
```

### 2. Download and Preprocess Data 
```bash
python data/download_data.py
python data/preprocess_data.py
```

### 3. Train the model
```bash
python models/train_ddm.py
```

### 4. Generate Synthetic Samples
```bash
python models/sample_ddpm.py
```

### or use API/generate.py 
```bash
python api/generate.py m-- NUM_SAMPLES CONFIG_PATH
```

### Evaluate generated synthetic samples
```bash
python utils/evauate.py
```

## Preprocessing explaination

#### 1. Data Alignment
- Load TCGA-OV FPKM gene expression and clinical metadata.
- Transpose expression so rows = samples.
- Left-join clinical features to expression to retain all molecular samples.

#### 2. Clinical Cleaning
- Drop administrative, redundant, and non-analytic fields (IDs, timestamps, non-primary diagnosis metadata, etc.).
- Impute missing values:
  - `days_to_death` → `0` for living patients.
  - `days_to_last_follow_up` and `age_at_earliest_diagnosis` → median.
  - Categorical `not reported` / missing → `"Unknown"`.

#### 3. Outcome Construction
- Convert `vital_status.demographic` into a binary `death_event` (Dead=1, Alive=0).
- Supports stratified dataset splitting and downstream survival-related modeling.

#### 4. Categorical Encoding
- One-hot encode all categorical variables (`drop_first=True`) to produce numeric model-ready features.

#### 5. Feature Typing
- Organize features into:
  - **Genetic:** all `ENSG*` expression genes.
  - **Binary clinical:** one-hot indicators.
  - **Continuous clinical:** age, survival times, etc.
- Enables type-appropriate transformations.

#### 6. Gene Expression Outlier Control
- Winsorize all gene expression features to the 1st–99th percentile.
- Stabilizes PCA and reduces sensitivity to extreme RNA-seq values.

#### 7. Train/Validation/Test Split
- 80/10/10 split using stratification on `death_event` to preserve outcome balance.

#### 8. Continuous Feature Scaling
- Standardize continuous clinical features with `StandardScaler` (fit on train only).
- Save `scaler.pkl` for reproducibility.

#### 9. PCA on Gene Expression
- Fit **32-component PCA** on training gene expression only.
- Transform validation/test sets using the same PCA.
- Save `pca_genetic.joblib` + metadata.

#### 10. Combo Feature Space
- Concatenate:
  - PCA latent genetic PCs  
  - Binary clinical features  
  - Scaled continuous features  
- Produces a compact, well-conditioned input space for DDPM training.

#### 11. Scaling the Combo Space
- Fit a separate `StandardScaler` on the combined PCA+clinical space.
- Save `combo_scaler.joblib` and export scaled train/val/test matrices.

## Known Limitations and Possible Future Improvements

Although the current diffusion based SDGE produces realistic and diverse data, there are several limitations:

- 1. **Trade off with latent-only modeling**: Model poperates on PCA- reduced latent space rather than full expression, which may omit finer grain signals. 
- 2. **Better clinical data: Better patient data**. A lot of clinical data was dropped or imputed.
- 3. **Small sample size**: TCGA-OV has usable sample size of 429 before preprocessing. This may limit the richness of the learned distribution. Some synthetic augmentation could help with this.


### What I would do with more time:

- 1. Atchitecture Enhancements: 
        - Experiment with learned encoder instead of PCA to capture nonlinear structure.
        - Compare and contrast different architecture solutions (tabDDPM vs CTGAN)

- 2. Deployment and Production
        - Build FASTAPI endpoint or develop lightweight UI Dsashboard (streamlit) where users can synthesize cohorts, visualize distributions and validate outputs. 



