# Signboard Detection using Custom dataset by YOLOv5

This was an undergrad course project for `"Image Processing" (CSE428)`.
This project is aimed at detecting signboards using the YOLOv5 model. Below are the steps and details on how we developed, trained, and used the model for prediction.

## Project Overview

1. **Annotation of Images**
   - We used the CVAT tool to annotate the images and exported them in YOLO-compatible format.

2. **Dataset Preparation**
   - From the `global_images` and `global_annotations`, we separated our desired train, test, and validation datasets using the `data-split(Group-5).ipynb` notebook. 

3. **Model Training and Evaluation**
   - We trained the YOLOv5 model using the prepared datasets.
   - Check `Model_train.ipynb`
   - After training, we evaluated the model's performance using various metrics.

4. **Prediction on New Images**
   - We created a folder named `for_prediction` containing images that we want to predict.
   - Using our trained YOLOv5 model, we performed predictions on the images in the `for_prediction` folder.
   - Run `Predictions.ipynb` to get the predictions and more.

## Usage Instructions

### 1. Annotation

To annotate your images, follow these steps:

1. Open the CVAT tool.
2. Annotate your images and export the annotations in YOLO format.

### 2. Preparing the Dataset

1. Place your annotated images and labels in `global_images` and `global_annotations` directories, respectively.
2. Run the `data-split(Group-5).ipynb` notebook to split the data into training, testing, and validation sets.

### 3. Training the Model

1. Ensure your datasets are properly placed in the respective directories.
2. Train the YOLOv5 model using the training dataset.
3. Evaluate the model's performance on the validation and test datasets.

### 4. Making Predictions

1. Add the images you want to predict into the `for_prediction` folder.
2. Use the trained YOLOv5 model to predict the signboards in the images within the `for_prediction` folder.


## Additional Information

- **CVAT Tool**: [CVAT]([https://app.cvat.ai/]) is an open-source tool for annotating digital images and videos.
- **YOLOv5**: YOLOv5 is a family of object detection architectures and models pre-trained on the COCO dataset, optimized for real-time processing.

## Conclusion

This project demonstrates the process of annotating images, preparing datasets, training a YOLOv5 model, and using the trained model to predict signboards. By following the instructions provided, you can replicate the process and apply it to your own custom object detection tasks.
