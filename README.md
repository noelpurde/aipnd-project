# Flower Species Image Classifier

This project is an AI application for classifying flower species using deep learning. It trains a neural network to recognize 102 flower categories and serves as an example of incorporating pre-trained deep learning models into practical applications like mobile apps or web services.  

## Features  
- **Dataset**: Trained on a dataset of 102 flower categories.  
- **Deep Learning**: Utilizes a pre-trained VGG16 model with a custom feed-forward classifier.  
- **Image Processing**: Includes image resizing, normalization, and data augmentation for robust training.  
- **Performance**: Achieves over 70% accuracy on unseen test data.  
- **Checkpointing**: Saves trained models for future inference and fine-tuning.  

## Implementation Details  
1. **Data Preprocessing**:  
   - Applied transformations like random scaling, cropping, flipping, and normalization.  
   - Used PyTorch's `torchvision.datasets` and `DataLoader` for training, validation, and testing.  
2. **Model Architecture**:  
   - Pre-trained VGG16 as the feature extractor.  
   - Custom classifier with ReLU activations, dropout, and a softmax output layer for 102 flower classes.  
3. **Training**:  
   - Loss function: Negative Log Likelihood (`NLLLoss`).  
   - Optimizer: Adam with a learning rate of 0.001.  
   - Tracked validation accuracy and loss for hyperparameter tuning.  
4. **Testing**:  
   - Evaluated model accuracy on the test dataset to validate generalization.  
5. **Inference**:  
   - Includes a prediction function to classify images and return top probabilities and flower classes.  

## How to Use  
1. **Download Dataset**:  
   The dataset is automatically downloaded and extracted from the provided URL.  
2. **Train the Model**:  
   Use the included script to train the classifier with your chosen hyperparameters.  
3. **Save and Load Model**:  
   Save the trained model using a checkpoint and reload it for future predictions.  
4. **Make Predictions**:  
   Use the inference function to classify new flower images.  

## Prerequisites  
- Python 3.7 or above  
- PyTorch  
- torchvision  
- matplotlib  
- numpy  
- PIL (Pillow)  

## Setup  
Clone the repository:  
   ```bash  
   git clone https://github.com/noelpurde/aipnd-project.git
```
```
   cd https://github.com/noelpurde/aipnd-project.git
   ```
