#!/usr/bin/env python3
"""
Preprocess TCGA-OV data for DDPM synthetic data generation.

- Align expression (ENSG*) and clinical samples.
- Keep key clinical variables:
    * race.demographic (categorical)
    * ethnicity.demographic (categorical)
    * vital_status.demographic (categorical)
    * figo_stage.diagnoses (categorical)
    * days_to_death.demographic (continuous)
    * days_to_last_follow_up.diagnoses (continuous)
    * age_at_earliest_diagnosis_in_years.diagnoses.xena_derived (continuous)
- Derive binary death_event from vital_status.demographic.
- One-hot encode categoricals. 
- PCA on genetic block with N_COMPONENTS components.
- Build PCA + clinical "combo" space and standardize.
- Save:
    * train/val/test_seed.csv       (full genes+clinical)
    * train/val/test_combo_scaled.csv
    * models/pca_genetic.joblib
    * models/pca_genetic_metadata.json
    * models/combo_scaler.pkl
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib




random_state = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Winsorize to clip outliers to 99, 1 percetile 
def winsorize_df(df, cols, lower=1, upper=99, verbose=True):
    df = df.copy()  # protect original unless assigned back
   
    for col in cols:
        col_before = df[col].copy()
        
        # Compute percentiles on original values
        lo = np.percentile(col_before, lower)
        hi = np.percentile(col_before, upper)
        
        # Clip
        df[col] = col_before.clip(lo, hi)
        
        # Count clipped values
        n_clipped = (df[col] != col_before).sum()
        #if verbose and n_clipped > 0:
            #print(f"{col}: clipped {n_clipped} values (lo={lo:.3f}, hi={hi:.3f})")
        
    return df

def main():

# TCGA-OV Gene expression DF 
    gene_expression = pd.read_csv("raw/TCGA-OV.star_fpkm.tsv.gz", sep="\t", index_col=0)

 # TCGA-OV clinical/phenotypic DF 
    phenotype = pd.read_csv('raw/TCGA-OV.clinical.tsv.gz', sep="\t", index_col=0)   

    gene_exp_T = gene_expression.T # Transpose the gene expression data to have rows as samples

    # Performing Left Join to create seed dataset 
    seed = gene_exp_T.join(phenotype, how='left')
   
    # Create processed
    os.makedirs("./processed", exist_ok=True)
    seed.to_csv("processed/seed_matrix.csv")
    
    # Dropping all columns tagged with sample, annotation, hospital, and project metadata 
    seed_cleaned = seed.drop(seed.filter(regex=r'\.annotations$|\.samples$|\.project$|\.tissue_source_site$').columns, axis=1)

    # Dropping other Non-Primary or not clincal columns
    cols_to_drop =[
        # Administrative
        'id', 'case_id', 'submitter_id','primary_site', 'state.treatments.diagnoses', 'submitter_id.treatments.diagnoses', 'treatment_id.treatments.diagnoses', 'icd_10_code.diagnoses',
        # Datetime admin fields
        'created_datetime.treatments.diagnoses',
        'updated_datetime.treatments.diagnoses',
        # Non-Primary or redundant Age fields
        'age_at_index.demographic',
        'age_at_earliest_diagnosis.diagnoses.xena_derived',
        'age_at_diagnosis.diagnoses',
        'days_to_birth.demographic',
        # Other irrelevant field
        'gender.demographic', 
        'year_of_death.demographic',
        'year_of_birth.demographic',
        'year_of_diagnosis.diagnoses',
        'days_to_diagnosis.diagnoses',
        'treatment_or_therapy.treatments.diagnoses',
        'disease_type',
        'tissue_or_organ_of_origin.diagnoses',
        'prior_treatment.diagnoses',
        'morphology.diagnoses',
        'classification_of_tumor.diagnoses',
        'site_of_resection_or_biopsy.diagnoses'
    ]

    seed_cleaned = seed_cleaned.drop(cols_to_drop, axis=1)

    seed_cleaned['days_to_death.demographic'] = seed_cleaned['days_to_death.demographic'].fillna(0)

    seed_cleaned['days_to_last_follow_up.diagnoses'] = seed_cleaned['days_to_last_follow_up.diagnoses'].fillna(seed_cleaned['days_to_last_follow_up.diagnoses'].median())

    seed_cleaned['age_at_earliest_diagnosis_in_years.diagnoses.xena_derived'] = seed_cleaned['age_at_earliest_diagnosis_in_years.diagnoses.xena_derived'].fillna(seed_cleaned['age_at_earliest_diagnosis_in_years.diagnoses.xena_derived'].median())
    
    seed_cleaned['figo_stage.diagnoses'] = seed_cleaned['figo_stage.diagnoses'].fillna('Unknown')

    seed_cleaned.isna().sum().sort_values(ascending=False).head(5)

    cols_to_drop = [
            'alcohol_history.exposures', 'synchronous_malignancy.diagnoses', 'last_known_disease_status.diagnoses', 
            'primary_diagnosis.diagnoses', 'prior_malignancy.diagnoses', 'tumor_grade.diagnoses',
            'progression_or_recurrence.diagnoses','treatment_type.treatments.diagnoses'
            ]
    
    seed_cleaned = seed_cleaned.drop(columns=cols_to_drop)
    
    seed_cleaned['death_event'] = seed_cleaned['vital_status.demographic'].map({
    'Dead': 1,
    'Alive': 0
    })

    seed_cleaned = seed_cleaned.drop(columns=['vital_status.demographic'])

    # Changing not reported to Unknown
    seed_cleaned['race.demographic'] = (seed_cleaned['race.demographic'].replace({'not reported':'Unknown'}))
    seed_cleaned['ethnicity.demographic'] = (seed_cleaned['ethnicity.demographic'].replace({'not reported':'Unknown'}))

    # creating dummy variables (One Hot encoding)
    categorical_cols = seed_cleaned.select_dtypes(include='object').columns

    seed_cleaned = pd.get_dummies(seed_cleaned, columns= categorical_cols, drop_first=True, dtype=int)

    # Getting list of Columns by kind and checking their length 

    numeric_cols = seed_cleaned.select_dtypes('number').columns.tolist()

    # 1. Genetic columns (genes)
    expr_cols = [c for c in seed_cleaned.columns if c.startswith("ENSG")]

    # 2. Binary non-genetic (0/1)
    bin_cols = [
        col for col in numeric_cols
        if col not in expr_cols and seed_cleaned[col].isin([0,1]).all()
    ]

    # 3. Continuous non-genetic
    continuous_cols = [
        c for c in numeric_cols
        if c not in expr_cols and c not in bin_cols
    ]

    # 4. Combined non-genetic
    non_genetic_cols = bin_cols + continuous_cols

    seed_cleaned = winsorize_df(seed_cleaned, expr_cols, lower=1, upper=99)
    bounds = {}  # store original bounds for verification

    for col in expr_cols:
        lo = np.percentile(seed_cleaned[col], 1)
        hi = np.percentile(seed_cleaned[col], 99)
        bounds[col] = (lo, hi)

    for col in expr_cols:
        lo, hi = bounds[col]
        seed_cleaned[col] = seed_cleaned[col].clip(lo, hi)

    # Performing train test and validation Split (80/10/10)
    train_df , testval_df = train_test_split(seed_cleaned, train_size=0.8, stratify=seed_cleaned['death_event'], random_state= random_state)

    test_df, val_df = train_test_split(testval_df, test_size=0.5, stratify=testval_df['death_event'], random_state= random_state)

    # Scaling Continuous Variables 
    scaler = StandardScaler()

    train_df[continuous_cols] = scaler.fit_transform(train_df[continuous_cols]) # Fit scalar to train_df 

    val_df[continuous_cols] = scaler.transform(val_df[continuous_cols])
    test_df[continuous_cols] = scaler.transform(test_df[continuous_cols])

    # Saving scaled DF
    train_df.to_csv("processed/train_seed.csv", index=False)
    val_df.to_csv("processed/val_seed.csv", index=False)
    test_df.to_csv("processed/test_seed.csv", index=False)

        # Saving scaler.pkl
    joblib.dump(scaler, "processed/scaler.pkl")

        # Saving Meta Data
    metadata = {
        "binary_cols": list(bin_cols),
        "continuous_cols": list(continuous_cols),
        "all_cols": list(seed_cleaned.columns)
    }

    with open("processed/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    n_components = 32 

    pca_gen = PCA(n_components=n_components, random_state=42)
    Z_train_gen = pca_gen.fit_transform(train_df[expr_cols].values)

    print("PCA fitted on:", pca_gen.n_features_in_) 
    print("Z_train_gen shape:", Z_train_gen.shape)   

    Z_val_gen  = pca_gen.transform(val_df[expr_cols].values)
    Z_test_gen = pca_gen.transform(test_df[expr_cols].values)

    latent_cols = [f"gene_pc_{i+1}" for i in range(n_components)]

    train_gen_latent_df = pd.DataFrame(Z_train_gen, columns=latent_cols)
    val_gen_latent_df   = pd.DataFrame(Z_val_gen,   columns=latent_cols)
    test_gen_latent_df  = pd.DataFrame(Z_test_gen,  columns=latent_cols)

    non_genetic_cols = bin_cols + continuous_cols

    train_non_gen_df = train_df[non_genetic_cols].reset_index(drop=True)
    val_non_gen_df   = val_df[non_genetic_cols].reset_index(drop=True)
    test_non_gen_df  = test_df[non_genetic_cols].reset_index(drop=True)

    non_genetic_cols = bin_cols + continuous_cols

    train_non_gen_df = train_df[non_genetic_cols].reset_index(drop=True)
    val_non_gen_df   = val_df[non_genetic_cols].reset_index(drop=True)
    test_non_gen_df  = test_df[non_genetic_cols].reset_index(drop=True)

        # Concatinating Genetic and Non Genetic fields back together 
    train_pca_combo = pd.concat([train_gen_latent_df, train_non_gen_df], axis=1)
    val_pca_combo   = pd.concat([val_gen_latent_df,   val_non_gen_df],   axis=1)
    test_pca_combo  = pd.concat([test_gen_latent_df,  test_non_gen_df],  axis=1)


        # Saving combo
    train_pca_combo.to_csv("processed/train_pca_genetic_combo.csv", index=False)
    val_pca_combo.to_csv("processed/val_pca_genetic_combo.csv", index=False)
    test_pca_combo.to_csv("processed/test_pca_genetic_combo.csv", index=False)

    joblib.dump(pca_gen, "../models/pca_genetic.joblib")

    metadata = {
        "expr_cols": expr_cols,
        "bin_cols": bin_cols,
        "continuous_cols": continuous_cols,
        "non_genetic_cols": non_genetic_cols,
        "latent_genetic_cols": latent_cols,
        "n_genetic_components": n_components
    }

    with open("../models/pca_genetic_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved PCA + datasets.")

    os.makedirs("../models", exist_ok=True)

    combo_cols = train_pca_combo.columns.tolist()
    print("Combo dim:", len(combo_cols))

    # Fit a StandardScaler on the *combo* space
    scaler_combo = StandardScaler()
    Z_train = scaler_combo.fit_transform(train_pca_combo.values)
    Z_val   = scaler_combo.transform(val_pca_combo.values)
    Z_test  = scaler_combo.transform(test_pca_combo.values)

    # Save scaler
    joblib.dump(scaler_combo, "../models/combo_scaler.joblib")

    # Save scaled combo datasets with same column order
    train_combo_scaled = pd.DataFrame(Z_train, columns=combo_cols)
    val_combo_scaled   = pd.DataFrame(Z_val,   columns=combo_cols)
    test_combo_scaled  = pd.DataFrame(Z_test,  columns=combo_cols)

    train_combo_scaled.to_csv("processed/train_combo_scaled.csv", index=False)
    val_combo_scaled.to_csv("processed/val_combo_scaled.csv", index=False)
    test_combo_scaled.to_csv("processed/test_combo_scaled.csv", index=False)

    print("Saved scaled combo datasets with shape:", train_combo_scaled.shape, val_combo_scaled.shape, test_combo_scaled.shape)

if __name__ == "__main__":
    main()

