# Test Case Minimization with Quantum Annealers
This repository contains the code necessary to reproduce the results of BootQA approach.

# Installation
Currently the repository only supports Linux distributions and is tested on Ubuntu 20.

### Dependencies

Anaconda Python distribution is required [here](https://www.anaconda.com/products/distribution):

Steps:

    1. Clone repository
    2. cd ASE_submission
    3  conda env create -f ocean.yml
    4. conda activate ocean
 
 # How to run
 You can find the implementation of BootQA, EIDQ, VQ, and SA under the "code" folder.
 
 ## To run BootQA
 You can run the following command line in the terminal under the root of the reporistory.
 ```
 python code/bootqa.py [sub-problem size] [run] [dataset_name]
 ```
 You can implement experiments of various sub-problem sizes (i.e., 10, 20, ...). You can set the number of repeat by the run parameter. You can select which dataset to run (i.e., "paintcontrol", "gsdtsr", "iofrol").
 
## To run EIDQ
You can use the following command line in the terminal.
 ```
 python code/eidq.py [sub-problem size] [run] [dataset_name]
 ```
## To run VQ
You can use the following command line in the terminal. Note that VQ can be implemented on the "paintcontrol" dataset.
 ```
 python code/vq.py [run]
 ```
 
## To run SA
You can use the following command line in the terminal. Note that VQ can be implemented on the "paintcontrol" dataset.
 ```
 python code/sa.py [run] [dataset_name]
 ```
 
# Results:
> **_NOTE:_** Each approach (BootQA, EIDQ, SA, and VQ ) has it's own folder. For BootQA and EIDQ the results are seperated by dataset/subproble_size/runs_number and for SA and VQ the results are seperated by dataset/runs_number

The results for each Run is provided seperately.
## Files related to results of SA and VQ:

    1. approach/dataset/run/summary.csv  (Contains summarized data for RQ results)
    2. approach/dataset/run/result.csv  (constains data of selected test cases)
For example for paintcontrol dataset for VQ run 0 the file path would be VQ/paintcontrol/0/summary.csv

## Files related to results of BootQA and EIDQ:

    1. approach/dataset/size/run/summary.csv  (Contains summarized data for RQ results for each sub problem size)
    2. approach/dataset/size/run/log.csv  (constains raw data of the final soultion for each sub problem size)
    3. approach/dataset/size/run/sample_info/ (this folder contains the data for each test case in the final solution for each sub problem size)
    4. approach/dataset/size/run/result.csv/ (constains data of selected test cases)
For example for paintcontrol dataset for BootQA for subproblem size 20 and run 0 the file path would be BootQA/paintcontrol/size_20/0/summary.csv

> **_NOTE:_** Running the experiment from scratch requred D-wave api authentication which is not provided in this repository.
