# GB500

# How To Run

## Data

Get the data from `http://medicaldecathlon.com/` and put the files inside one folder called data_heart, the folder structure must be:
    \- data_heart
        \- imagesTr
        \- imagesVal
        \- labelsTr
        \- labelsVal

- images are the volumes and labels are the groud truths
- Tr means Training, you should put the nifti volumes for training inside this folder
- Val means Testing, you shoul put the nifti volumes for testin inside this folder

## Files

- The file directories follow:
    - `MainBackbone.ipynb`      -> Run the Neural Network
    - `Load Results.ipynb`      -> Load the test dataset with trained weights
    - `DivergentNets.ipynb`     -> Load weights from the base folders and make the average results (DivergentNets).
    - `Main.ipynb`              -> Test Script, doesn't have any valuable code.


