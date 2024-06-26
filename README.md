# Card Detection

It detects different types of cards with their symbols and numbers. It is a classification model that classifies cards using resnet 18. 

![add image descrition here](direct image link here)

## The Algorithm

ResNet-18, short for Residual Network with 18 layers, is a deep learning model designed for image classification tasks. It is part of the ResNet family, which introduced the concept of residual learning to address the vanishing gradient problem in deep neural networks.

## How ResNet-18 Works

### Architecture
ResNet-18 consists of 18 layers, including convolutional layers, batch normalization layers, ReLU activation functions, and fully connected layers. The key innovation in ResNet is the introduction of residual blocks, which allow the network to learn residual functions with reference to the layer inputs, rather than learning unreferenced functions.

### Residual Blocks
A residual block typically contains two or three convolutional layers. The input to a residual block is added to the output of the block before applying the activation function. This shortcut connection helps in mitigating the vanishing gradient problem by allowing gradients to flow directly through the network.

### Training and Inference
1. **Input**: The model takes an input image of size $$224 \times 224$$ with three color channels (RGB).
2. **Convolutional Layers**: The initial layers perform convolution operations to extract low-level features such as edges and textures.
3. **Residual Blocks**: These blocks further process the features, capturing more complex patterns and structures in the image.
4. **Fully Connected Layer**: The final layer is a fully connected layer that outputs a probability distribution over the possible classes.

### Pre-training and Transfer Learning
ResNet-18 is often pre-trained on large datasets like ImageNet, which contains millions of images across a thousand classes. This pre-training allows the model to learn a wide variety of features that can be fine-tuned for specific tasks using transfer learning. Fine-tuning involves training the pre-trained model on a new dataset with a smaller learning rate to adapt the learned features to the new task[4].

### Implementation
ResNet-18 can be implemented using deep learning frameworks like PyTorch. The typical steps include:
1. **Loading the Pre-trained Model**: Load the ResNet-18 model pre-trained on ImageNet.
2. **Modifying the Final Layer**: Adjust the final fully connected layer to match the number of classes in the new dataset.
3. **Training**: Train the model on the new dataset, fine-tuning the weights to improve performance on the specific task.
4. **Evaluation**: Evaluate the model's performance on a validation set to ensure it generalizes well to unseen data[1][2][3].

### Applications
ResNet-18 has been successfully applied in various domains, including:
- **General Object Classification**: Classifying objects in images into predefined categories.
- **Remote Sensing**: Classifying land cover types in satellite images[5].

In summary, ResNet-18 is a powerful and efficient model for image classification, leveraging residual learning to achieve high performance while mitigating common issues in deep neural networks.

Citations:
[1] https://www.kaggle.com/code/ggsri123/implementing-resnet18-for-image-classification
[2] https://www.youtube.com/watch?v=mn5QDKQ54dQ
[3] https://www.kaggle.com/code/ggsri123/implementing-resnet18-for-image-classification/code
[4] https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/resnet-18-pytorch/README.md
[5] https://www.mdpi.com/2072-4292/14/19/4883
---
## Running this project
### 1. Connect to your Nano in VS Code.


 
### 2. If you are in the jetson-inference directory, click File > Open Folder, and open the nvidia folder. 
Home folder open 
![image](https://github.com/Kook1y/card_detection/assets/173178626/27079ff1-cbf1-4b62-bb3f-633e650cde07)

### 3. Use the New Folder icon to create a new directory called my-recognition.
![image](https://github.com/Kook1y/card_detection/assets/173178626/67471950-c4df-4fb4-ac05-590343eb46fe)


### 4. Use the New File icon to create a new file called my-recognition.py.
![image](https://github.com/Kook1y/card_detection/assets/173178626/b10eb94c-d642-4197-b88b-9d3be72d8314)


### 5. At the top add the code below to automatically use the python interpreter.
#!/usr/bin/python3

![image](https://github.com/Kook1y/card_detection/assets/173178626/9a1c782e-6122-472b-82aa-5568f2fbcdc1)


 
### 6. Import the jetson modules used for recognizing and loading images. Also, import argparse to parse the command line. 
```
import jetson_inference
import jetson_utils
import argparse
```
 
### 7. Add this code to parse the image file name and select a network.
```
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
opt = parser.parse_args()
``` 
### 8. Load the image from the disk using the loadImage function. This line will use the image name from the command line.
```
img = jetson_utils.loadImage(opt.filename)
```
### 9. Load the recognition network as specified in the command line. 
```
net = jetson_inference.imageNet(opt.network)
 ```
### 10. Now it's time to classify the image. In order to do this, you will run the imageNet.Classify() function. You will get the index of the class the image is, and the confidence.
```
class_idx, confidence = net.Classify(img)
 ```
### 11. Next, get the description for the class that the image belongs to.
```
class_desc = net.GetClassDesc(class_idx)
```
 
### 12. Finally, write a print statement to print out the information that you have gathered including the class index, class description, and confidence. The confidence will be a decimal, so it is recommended to multiply it by 100 to get a percentage.
```
print("image is recognized as "+ str(class_desc) +" (class #"+ str(class_idx) +") with " + str(confidence*100)+"% confidence")
codefinal
```
![image](https://github.com/Kook1y/card_detection/assets/173178626/6888abe3-792b-4ae0-95d0-a7f5f7d88ff6)


 
### 13. Save your work with Ctrl + S.

 
### 14. Select Terminal > New Terminal to open a new terminal. In this terminal, you can run all the same commands as in a terminal on your Nano.

![image](https://github.com/Kook1y/card_detection/assets/173178626/33e7ab28-7690-4d17-ae64-c42f07da03e4)

If your terminal is closed, you can open up the terminal option at the bottom by clicking this icon at the top.
![image](https://github.com/Kook1y/card_detection/assets/173178626/5be86047-cfb6-4be0-8b48-c18b4f8e2955)


 

 
### 15. Change directories (cd) into your my-recognition directory.

 
### 16. Download an image of a polar bear using this command:
```
wget https://github.com/dusty-nv/jetson-inference/raw/master/data/images/polar_bear.jpg
```

![image](https://github.com/Kook1y/card_detection/assets/173178626/bfc019c7-78e5-427a-852a-39e50f98baaa)


 
### 17. Run your image recognition program on it like this.
```
python3 my-recognition.py polar_bear.jpg
```
![image](https://github.com/Kook1y/card_detection/assets/173178626/152d51b8-50c0-41cf-8dc5-b267863f8fd6)


Output from my-recognition
The code runs and gives you the bottom line that says:  ```image is recognized as ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus (class #296) with 100.0% confidence.```
This tells you that the image was recognized successfully as a polar bear.



readme.md
Displaying readme.md.
