# Final-Project-Yash-Trivedi
Repository containing files related to the final project in ECEN 5743: Deep Learning

This repository has four folders:
1. Code: Contains python files and data to be used
   - .xlsx files for each data set which contain raw temperature data, calculated temperature gradient, and CHF proximity
   - .csv files which were used when splitting videos into frames that line up with data points
   - "outputs" folder which is where train/test output files like .pt files and prediction .csv files are stored
   - case84_aligned_frames and case91_aligned_frames python files used to split videos into frames aligned with data points
   - train_resnet18.py holds training program usign the ResNet 18 pretrained model
   - test_resnet18.py is matching test file to test on ResNet 18 pretrained model
   - Two folders which contain 1200 frames from each video test set
2. Final-Project-Report: Contains final report document
3. Final Presentation: Contains final presentation
4. Proposal: Contains initial proposal document and relevant documents

How to run training and test:
No data processing needs to be done beforehand, data is preprocessed to be ready to train and test. 
Example calls for running training or testing are given below, some changes may have to be made to match program names.
The training and test files are set up with parser arguments which can be customized if so wished, but some are necessary:
1. "--annotations": Mandatory, include path to annotations file for dataset
2. "--image_dir": Mandatory, include path to folder of images
3. "--output_dir": Optional, include path to output files, default is "outputs_case84_resnet18"
4. "--epochs", optional, default=10
5. "--batch_size", optional, default = 32
6. "--lr", optional, default is 1e-4
7. "--weight_decay", optional, default is 1e-4
8. "--val_split", optional, set percentage of validation split, default is 0.2
9. "--num_workers", optional, default is 0.2
10. "--seed", optional, sets random seed, default is 42
11. "--freeze_backbone_epochs", optional, allows user to freeze backbone training for given epochs, default is 0


Example Code Calls:

python3 Code/train_alexnet.py \                                              
  --annotations Code/Boiling-91_Temperature_10Hz.xlsx \
  --image_dir Code/Boiling-91_frames \
  --output_dir Code/Outputs/outputs_alexnet_91_Aug2 \             
  --epochs 8 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 1e-4

python3 Code/test_alexnet.py \                                               
  --checkpoint Code/Outputs/outputs_alexnet_91_Aug2/best_model_weights.pt \
  --annotations Code/Boiling-84_Temperature_10Hz.xlsx \
  --image_dir Code/Boiling-84_frames \
  --output_csv Code/Outputs/outputs_alexnet_91_Aug2/test_on_84.csv

  
