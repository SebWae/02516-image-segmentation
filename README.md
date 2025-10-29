# Project 3 - Image Segmentation
Welcome to repository of the second project of the DTU course [Introduction to Deep Learning in Computer Vision](https://kurser.dtu.dk/course/2025-2026/02516?menulanguage=en) about image segmentation. 

All scripts are being executed using the [DTU HPC](https://www.hpc.dtu.dk/).  

📅 Hand-in: November 16th, 2025

## Project Structure
```
├── docs                                            <- Project description and instructions
│   │
│   ├── hpc_instructions.md                         <- Instructions on how to use the HPC   
│   │
│   └── IDLCV_Project_3_Segmentation_part1.pdf      <- Projection description part 1
│
├── lib                                             <- Imports for datasets and models
│   │
│   ├── dataset                                     <- Dataset classes for DataLoader
│   │   |
|   │   └── PhCDataset.py                           <- Dataset class for the PH2 dataset
│   │
│   ├── model                                       <- Segmentation model classes
│   │   |
|   │   ├── DilatedNetModel.py                      <- Class for the DilatedNet model
│   │   |
|   │   ├── EncDecModel.py                          <- Class for the EncDec model
│   │   |
|   │   └── UNetModel.py                            <- Class for the UNet model
│   │   
│   └── losses.py                                   <- Loss functions for ablation study
│
├── bash_script.sh                                  <- Bash script template used to run scripts on HPC
│
├── measure.py                                      <- Script to evaluate the segmentation models
│
├── predict.py                                      <- Script to generate segmentations using our trained models
│
├── requirements.txt                                <- Packages used to initialize the virtual environment
│
└── train.py                                        <- Script used to train the segmentation models
```

## Virtual Environment
To setup the virtual environment used when submitting a batch job, follow the instructions in [HPC Instructions](https://github.com/SebWae/02516-image-segmentation/blob/main/docs/hpc_instructions.md). Once the virtual environment (`venv_proj3`) has been created, install the required packages specified in `requirements.txt` by running: 
```
pip install -r requirements.txt
```
If more packages/libraries are needed, install these using `pip` and then update `requirements.txt` by running: 
```
pip freeze > requirements.txt
```
Remember to commit and push the changes 😊 
