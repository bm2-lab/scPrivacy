# scPrivacy: Privacy-preserving integration of multiple institutional data for single-cell type identification
## Introduction
scPrivacy presents a generalized automatically single-cell type identification prototype to facilitate single cell annotations in a data privacy-perserving collaboration manner, by integrating multiple references data distributed in different institutions using an federated learning based deep metric learning framework. The basic idea of scPrivacy is to make each institution train their models locally and aggregate encrypted models parameters for all institutions by avoiding putting raw data of all institutions together directly. We evaluated scPrivacy on 42 publicly available benchmark datasets for single cell type identification to stimulate the scenario that the reference datasets are rapidly generated and accumulated from multiple institutions, while they are prohibited to be integrated directly or exposed to each other due to the data privacy regulations, and demonstrated its effectiveness and time efficiency. 
## Workflow
![](https://github.com/bm2-lab/scPrivacy/blob/main/scPrivacy_workflow.png)
scPrivacy comprises two main steps: model learning and cell assignment.
* (1) In the model learning process, scPrivacy trains a federated deep metric learning model on multiple institutional datasets in a data privacy-preserving manner. The institutions train the models on the basis of the data of local datasets. The trained model parameters are then encrypted and transmitted to the server. The server updates the federated model via aggregating model parameters. Finally, the institutions download the updated federated model. The process will repeat many times.
* (2) In the cell assignment process of scPrivacy, the federated model is utilized to transform the query cells. Then, the transformed query cells are compared against cell type landmarks of transformed institutions datasets, and the predicted cell type with the highest similarity among all cell type landmarks is obtained.

## Install
Environment: Python>=3.6
* (1) Clone the repository
```
git clone https://github.com/bm2-lab/scPrivacy.git  
```
* (2) Install the dependencies
```
pip install scprivacy
```
* (3) Create folder 

   Linux or Mac:
   ```
   cd scPrivacy
   sh create_folder.sh
   ```
   Windows:
   ```
   cd scPrivacy
   create_folder.bat
   ```
## Tutorial
### Format of input data
A routine normalization and quality control should be performed. For example, there are three commonly used cell quality criteria, namely, the number of genes detected (default >500), the number of unique molecular identifiers induced (default >1500), and the percentage of mitochondrial genes detected (default <10% among all genes). Then, datasets should be normalized, i.e., scaling to 10,000 and then with log(counts+1).
The format of training data should be a csv or tab-delimited txt or h5ad(scanpy) format where the columns correspond to genes and the rows correspond to cells. The column of cell types should be the last column and named as "cell_label". In a word, the format of training data is a transposed normalized dataset with a cell type column in the right. A sample file looks something like:

|   | tspan6 | dpm1 | cell_label |
| ------------- | ------------- |------------- | ------------- |
| pbmc1_SM2_Cell_133  | 0.745639  |0.0  |CD4+_T_cell |
| pbmc1_SM2_Cell_142  | 0.0  |0.778851  |B_Cell  |

The format of test data is the format of training data without the column of "cell_label".
We also provide a script `preprocess.py` to handle the cell quality control and normalization of origin counts matrix dataset. The processed file will end up with "\_treated.h5" in its name and can read through `pandas` package's `read_hdf` function. After processing by the script, you just need to add the cell type column in the right of the processed matrix. The origin counts matrix dataset should be a csv or tab-delimited txt or h5ad(scanpy) format where the columns correspond to cells and the rows correspond to genes looks like:
|   | pbmc1_SM2_Cell_133 | pbmc1_SM2_Cell_142 |
| ------------- | ------------- |------------- |
| tspan6  | 4  |0  |
| dpm1  | 0  |9  |

The script can be run by calling the following command. The `filename` is file name of the origin counts matrix dataset.
  ```
  python preprocess.py -f filename
  ```
### How scPrivacy works
* **Example datasets:** The example data is in https://www.jianguoyun.com/p/DbqGHM0Q7oftCBiFjNQD. You can download whole `train_set` and `test_set` folder. Once downloaded, you need to unzip the `train_set.zip` and `test_set.zip` and then use the unzipped `train_set` folder and `test_set` folder to replace the origin `train_set` folder and `test_set` folder. The demo example can be run by calling the following command.
  ```
  python run.py
  ```
* **Cell type identification with model trained by your datasets:** You just need to put reference datasets files in the `train_set` folder, put the query datasets files in the `test_set` folder and run the following command.
  ```
  python run.py
  ```
* **Cell type identification with trained models:** You can also use your own model. You just need to put the query dataset file in the `test_set` folder and run the following command.
  ```
  python test_with_trained_model.py -m new_model
  ```
* **Check the results:** For all the above tests, the type identification results can be found in the `result.txt`  after running. The content of the `result.txt` consists of two columns. The first column represents cell barcodes and the second column represents predicted cell class information. It looks like:<br />
pbmc1_SM2_Cell_133&nbsp;&nbsp;&nbsp;&nbsp;CD4+_T_cell<br />
pbmc1_SM2_Cell_142&nbsp;&nbsp;&nbsp;&nbsp;B_Cell

## Contacts  
bm2-lab@tongji.edu.cn
