# EECS595_Team15
The repository is for EECS595's final project of team 15.

The topic is: [< Physical Action-Effect Prediction with GPT and CLIP >] 

## CLIP Requirements

```
    torch >= 1.13.1
    clip >= 1.0
    numpy
    tqdm
```

## CLIP Usage

After putting the dataset to the folder "action_effect_image_rs", run the command on terminal to run CLIP.
```
    python action_effect.py
```

## GPT4 Usage
Setting Up:
The script starts by loading the "sled-umich/Action-Effect" dataset.
It seeds the random number generator to ensure reproducibility.
A subset of 500 images is selected for processing.
Image Processing:
The images are converted to RGB format if they are not already.
The dataset is divided into 10 chunks, each containing 50 images.
Model Interaction:
run_model function is used to send batches of images to the GPT-4 model.
You need to replace api_key with your OpenAI API key.
The function processes the images in batches and retrieves the model's responses.
Data Processing and Analysis:
The responses from the model are processed to extract the classifications.
The script calculates the accuracy of the model by comparing its predictions against the labels provided in the dataset.
Calculating Accuracy:
The final part of the script calculates the accuracy of the model's classifications.
It compares the predicted action-effect pairs with the actual labels and computes the percentage accuracy.


