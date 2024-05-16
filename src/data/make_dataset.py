#here some of the code from the streamlit app is stored for downloading the datasets directly from kaggle.

import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.model_selection import train_test_split

def download_datasets(download_path, dataset_owner="shayanfazeli", dataset_name="heartbeat"):
    # Configure and authenticate with the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Check if the dataset folder already exists
    dataset_folder = os.path.join(download_path, dataset_name)
    if not os.path.exists(dataset_folder):
        # Dataset folder does not exist --> Download and save the datasets
        api.dataset_download_files(dataset_owner + "/" + dataset_name, path=dataset_folder, unzip=True)
        print("Datasets are downloaded and unzipped.")
    else:
        # Dataset folder exists, but datasets might be missing
        missing_files = [] 
        for file_name in ["mitbih_test.csv", "mitbih_train.csv", "ptbdb_abnormal.csv", "ptbdb_normal.csv"]:  
            file_path = os.path.join(dataset_folder, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)

        if missing_files:
            # If missing files are present, download ALL files and overwrite the old folder.
            api.dataset_download_files(dataset_owner + "/" + dataset_name, path=dataset_folder, unzip=True, force=True)
            print("Missing data was downloaded and unzipped. All Datasets are now available.")
        else:
            print("All Datasets are already available.")

def load_datasets_in_workingspace(path_to_datasets="./heartbeat"):
    #reading in the datasets from the local ../data folder --> this folder is not pushed on github and only locally available.
    mitbih_test = pd.read_csv(path_to_datasets + "/" + "mitbih_test.csv",header=None)
    mitbih_train = pd.read_csv(path_to_datasets + "/" + "mitbih_train.csv",header=None)
    ptbdb_abnormal = pd.read_csv(path_to_datasets + "/" + "ptbdb_abnormal.csv",header=None)
    ptbdb_normal = pd.read_csv(path_to_datasets + "/" + "ptbdb_normal.csv",header=None)
    return mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal

##### ORIGINAL DATAPATH / DOWNLOADPATH FROM STREAMLIT APP ######
data_path = "."
download_datasets(data_path)


#####ALL CODE ABOVE SHOULD BE USED WHEN FEEDING THE MODELS (SEPARATE FUNCTION) #####
#### Loading the datasets into workingspace ####
mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace()
# --> This are the complete datasets. If we want to use batch-sizes, we have to split them in order to simulate a continuous flow of new training data.



# We now make the test and train set for ptbdb directly
ptbdb_concated = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True).sample(frac=1, random_state=42)
print("Size of ptbdb_concated:", ptbdb_concated.size)
X_ptbdb = ptbdb_concated.iloc[:,:186]
y_ptbdb = ptbdb_concated.iloc[:,:-1]
X_train_ptbdb, X_test_ptbdb, y_train_ptbdb, y_test_ptbdb = train_test_split(X_ptbdb, y_ptbdb, test_size=0.25, random_state=42)
#Test and train set are mere variables and can only be passed as globals. This is not good practice and a specific function should be called for this.
#THis is just for test purposes and rapid prototyping.
print("All test and train sets successfully prepared, but not globally available")