# RS-Coreset

## Requirement
python3+  
pytorch  
numpy  
sklearn  
matplotlib  

You can also use our `requirements.txt` to install
```
python==3.9+
pip install -r requirements.txt
```

## Usage

### For your own dataset
1. Prepare an xlsx file of reaction data in SMILES format according to the given style and convert the reaction space into a binary npy file with provided script in utils.
2. Or prepare your own molecular descriptor npz file with data store as `train_data` and yield(0-100) store as `train_label`
3. Modify the arguments in `class Arguments` of `main.py`
4. run `main.py` and the prediction result will be represent as `result.csv`

### For our real world dataset

Download and unzip `reaction93.zip` in current dir from https://drive.google.com/file/d/1O_Qcn5Z2gwr5e93Uh694d7d5HK3spkJo/view?usp=drive_link
```
python yield_predict_real.py
```

### For public HTE dataset
We prepare BH.py, SM.py and BH_Plus.py for performance test on these three public HTE dataset  
Download and unzip `datasets.zip` in current dir from [https://drive.google.com/drive/folders/1Dioh_fcPyMbhNNtEmNyUE4TrzxMnXgkV?usp=sharing](https://drive.google.com/file/d/1LGjips42xvGdOU_8pzS3quciIiXiYLfz/view?usp=drive_link)  
```
python BH.py
python SM.py
python BH_Plus.py
```
