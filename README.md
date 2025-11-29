# AI-ENGINEER-ASSIGNMENT-SDGE-

# AI Engineer Assignment – SDGE

## Project Overview

### TCGA-OV Cohort

The TCGA Ovarian Serous Custadenocarcinoma (TCGA-OV) cohort is one of the most comprehensive datasets available for molecular, genetic expression, clinical and phenotypes at patient level. The data set includes multi-omic assays curated through the NCI Genomic Data Commons (GDC). 

These datasets were downloaded using the UCSC Xena Browser (associated download scripts, as described in the project README.)

Generating high-fidelity synthetic biomedical data serves many practical and scientific motivations:

(1) Privacy Preservation of data sharing 
(2) Augment smaller sample biomedical data 
(3) Hypothesis exploration (gene/gene correlation structures etc.)

The TCGA-OV dataset is well suited to perform Synthetic Data Generation (SDGE), with rich multi-omic hetoerogentiy and good quality clinical meta data. 

This notebook explores the following workflow of SDGE on the TCGA-OV dataset:

- preprocessing of data
- contstucting a model-ready feature matrix 
- training diffusion-based generative model
- evaluating and comparision of synthetic data 

Generated data will be tested for statistical fidelity, correlation and similarity metrics to original data. 


## 1. Environment Setup
```bash
git clone https://github.com/scottpenco/AI-ENGINEER-ASSIGNMENT-SDGE-.git
cd AI-ENGINEER-ASSIGNMENT-SDGE

pip install -r requirements.txt
```

## 2. Download and Preprocess Data 
```bash
python data/download_data.py
python data/preprocess_data.py
```

## 3. Train the model
```bash
python models/train_ddm.py
```

## 4. Generate Synthetic Samples
```bash
python models/sample_ddpm.py
```

## or use API/generate.py 
```bash
python api/generate.py m-- NUM_SAMPLES CONFIG_PATH
```

## Evaluate generated synthetic samples
```bash
python utils/evauate.py
```

## Preprocessing explaination
