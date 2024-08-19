# RS-Coreset



## Requrements
pytorch  
numpy  
sklearn  
matplotlib  

## Usage

### For Yield Prediction
1. Prepare an xlsx file of reaction data in SMILES format according to the given style.
2. Convert the reaction space into a binary npy file (we provide a basic molecular fingerprint conversion script in utils).
3. Modify the dataset name in main.py and run main.py

### For Test
We prepare BH.py, SM.py and BH_Plus.py for performance test on these three public HTE dataset
```
python ./dataset/prepare_dataset.py
python ./test/BH.py
python ./test/SM.py
python ./test/BH_Plus.py
```
