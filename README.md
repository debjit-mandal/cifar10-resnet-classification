
# CIFAR-10 Image Classification using ResNet

This project aims to classify images from the CIFAR-10 dataset using a Residual Network (ResNet). The notebook implements a deep learning model that leverages the ResNet50 architecture, pretrained on ImageNet, to achieve high accuracy in image classification tasks.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Model Architecture
The model uses the ResNet50 architecture with pre-trained ImageNet weights. It includes additional custom layers to adapt the network for CIFAR-10 classification.

## Getting Started

### Prerequisites
Ensure you have the following libraries installed:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

### Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/debjit-mandal/cifar10-resnet-classification
    ```
2. Navigate to the project directory:
    ```sh
    cd cifar10-resnet-classification
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Notebook
1. Start Jupyter Notebook:
    ```sh
    jupyter notebook
    ```
2. Open the `CIFAR10_Image_Classification_ResNet.ipynb` notebook and run the cells to execute the project.

## Project Structure
```
CIFAR10_Image_Classification_ResNet/
├── CIFAR10_Image_Classification_ResNet.ipynb
├── README.md
├── requirements.txt
├── LICENSE
```

## Results
The model achieved high accuracy in classifying images into 10 categories. Below is the classification report and confusion matrix:

### Classification Report
```
              precision    recall  f1-score   support

           0       0.87      0.89      0.88      1000
           1       0.95      0.94      0.94      1000
           2       0.84      0.82      0.83      1000
           3       0.75      0.73      0.74      1000
           4       0.87      0.86      0.87      1000
           5       0.86      0.85      0.85      1000
           6       0.92      0.92      0.92      1000
           7       0.91      0.91      0.91      1000
           8       0.92      0.94      0.93      1000
           9       0.94      0.94      0.94      1000

    accuracy                           0.88     10000
   macro avg       0.88      0.88      0.88     10000
weighted avg       0.88      0.88      0.88     10000
```


## Conclusion
### Key Findings
- Achieved high accuracy in classifying images into 10 categories.
- Effective data augmentation improved model performance.
- Insights into model performance using the classification report and confusion matrix.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)

----------------------------------------------------------------

Feel free to suggest any kind of improvements.