# AI-ENGINEER-ASSIGNMENT-SDGE-

# AI Engineer Assignment – SDGE

## 1. Environment Setup
```bash
git clone https://github.com/scottpenco/AI-ENGINEER-ASSIGNMENT-SDGE.git
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
