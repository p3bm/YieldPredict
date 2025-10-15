# RS-Coreset

## Requirement
python3+  
pytorch  
numpy  
sklearn  
matplotlib  
rdkit  

You can also use our `requirements.txt` to install  

```
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


# 🧪 HTS Project — Reaction Suggestion Workflow

## Overview
This project helps automate reaction selection for high-throughput screening (HTS).  
It iteratively suggests new reactions based on existing data and experimental results.

---

## 🚀 How to Run

### 1. Run the main script
Execute the following command to start the process:
```bash
python catsci_main.py
```
- You need to provide the path of the csv file and it will create 2 npz files.
- It will give you next 15 reactions. If you wanna change the number of reactions it suggest go to main main.py and change the step size value.
- Use the suggested reactions and once you get the actual yield for those reactions combine it with the original data and run the catsci_main.py file again to get next batch of reactions.
