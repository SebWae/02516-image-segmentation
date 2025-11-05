# this script should include a train function taking one of the models as input

# applies 8-fold cross-validation for the PH2 dataset and 4-fold cross-validation for the DRIVE dataset
# each fold should act the role as the validation set
# performance is reported as the average across all folds
# applies early-stopping when validation loss increases (after x initial epochs) or only decreases by something smaller than a tolerance (e.g. e-4)

 