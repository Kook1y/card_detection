# Card Detection

## Important Files to Download

### Please download these files from Kaggle to follow the steps.
- [Kaggle_Card_Detection](https://www.kaggle.com/datasets/tanavyedlapalli/card-detection-dataset-for-image-classification) 


## Authors

- [@Kook1y](https://github.com/Kook1y)

## 🚀 About Me
I am a high school student in India who just entered 10th grade. I have an interest in AI models and machine as I always found it fascinating how the models worked. I have learnt a littlebit of python and machine learning. I have also created this model using resnet-18 pretrained on imagenet.

## What is the AI model
It detects different types of cards with their symbols and numbers. It is a classification model that classifies cards using resnet 18. 

![image](https://github.com/Kook1y/card_detection/assets/173178626/600f9292-8b50-4edc-9a6f-a5c0a70bfbea)



## What is a Jetson Nano
The NVIDIA Jetson Nano is a compact, powerful AI computer designed for developers, learners, and hobbyists to create and deploy AI applications at the edge. Here are the key features and functionalities of the Jetson Nano:

![image](https://github.com/Kook1y/card_detection/assets/173178626/adee1f84-f868-489c-8ccb-60d11f2b5e60)


### Specifications
- **CPU**: 64-bit quad-core ARM Cortex-A57 running at 1.43 GHz.
- **GPU**: NVIDIA Maxwell architecture with 128 CUDA cores, capable of 472 GFLOPs (FP16).
- **Memory**: 4GB of 64-bit LPDDR4 RAM.
- **Storage**: 16GB of eMMC storage.
- **Connectivity**: Includes support for Gigabit Ethernet, USB 3.0, HDMI, DisplayPort, and MIPI-CSI camera interfaces[2][4].

### Key Features
- **AI and Machine Learning**: The Jetson Nano is optimized for AI workloads, capable of running multiple neural networks in parallel for applications such as image classification, object detection, and segmentation[1][5].
- **NVIDIA JetPack SDK**: This comprehensive suite includes libraries, APIs, and tools for developing AI applications, ensuring compatibility with the entire NVIDIA ecosystem[5].
- **Low Power Consumption**: The board operates between 5 and 10 watts, making it suitable for power-constrained environments[4].
- **Peripheral Support**: The Jetson Nano Developer Kit includes a 40-pin GPIO header compatible with many peripherals and add-ons, similar to the Raspberry Pi[4].

### Applications
- **Educational**: Ideal for learning AI and robotics, with free courses and certifications from NVIDIA[2].
- **Robotics**: Supports frameworks like ROS and tools like PyBullet and Realsense depth cameras, making it suitable for robotics projects[2].
- **Edge AI**: Enables the development of AI applications at the edge, such as smart cameras and IoT devices, with efficient processing capabilities[1][5].

### Advantages
- **Affordable**: Provides a cost-effective entry point into AI development.
- **Community Support**: A large community and extensive tutorials are available for beginners[2].
- **Scalability**: Flexible and scalable platform, reducing development costs and time to market[2].

### Disadvantages
- **Limited Connectivity**: Lacks built-in WiFi and Bluetooth, which are available in some competing products like the Raspberry Pi[2].
- **PWM Pins**: Only two PWM pins are provided, which may limit certain applications[2].

In summary, the NVIDIA Jetson Nano is a versatile and powerful tool for AI and robotics development, offering a balance of performance, power efficiency, and affordability. It is particularly well-suited for those looking to explore AI applications without significant financial investment.



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
```#!/usr/bin/python3```

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

---
## How to Re-Train Image classification Model
```## Lesson Materials
Hardware Components

Jetson Nano with SD Card Inserted
Barrel Power Supply
SSH Connection
Reliable Internet Connection
```
---
### Now that you know how to create your own code for running an image network, you are going to learn how to train a model. The first model that we'll be re-training is a simple model that recognizes two classes: cat or dog.

![image](https://github.com/Kook1y/card_detection/assets/173178626/a96bb449-4704-4141-8f37-f2e8731fde06)
![image](https://github.com/Kook1y/card_detection/assets/173178626/bb6f2caa-1210-4cd3-aabb-90989cc68dfd)


### 1. cd back to nvidia/jetson-inference/

You may get an error later saying you're out of memory. You can ensure your system overcommits memory which allows it to have more memory for the task by running this now. Run this in your terminal when you're in your jetson-inference directory:
 ```echo 1 | sudo tee /proc/sys/vm/overcommit_memory```

 

 
### 2. Once that has run and you're still back in the jetson-inference folder, run 
```./docker/run.sh to run the docker container.``` (You may have to re-enter your nvidia password at this step)

![image](https://github.com/Kook1y/card_detection/assets/173178626/f8a3112a-c01e-483b-9aa4-adc50bedaf62)


 
### 3. From inside the Docker container, change directories so you are in _jetson-inference/python/training/classification__

![image](https://github.com/Kook1y/card_detection/assets/173178626/bc3ff9c3-ff2e-4256-a73c-8b85fff2b6ca)


 
### 4. Now you are ready to run your script.

Run the training script to re-train the network where the model-dir argument is where the model should be saved and where the data is. 
```python3 train.py --model-dir=models/cards data/cards```

 

You should immediately start to see output, but it will take a very long time to finish running. 
It could take hours depending on how many epochs you run for your model.

 

When running the model you can also specify the value of how many epochs and batch sizes you want to run. 
For example at the end of that code you can add:

```--batch-size=NumberOfBatchFiles --workers=NumberOfWorkers --epochs=NumberOfEpochs```

 ![image](https://github.com/Kook1y/card_detection/assets/173178626/b003cf99-63c4-434c-bf06-f13576e9cf63)




While it's running, you can stop it at any time using ```Ctl+C```. You can also restart the training again later using the ```--resume``` and ```--epoch-start``` flags, so you don't need to wait for training to complete before testing out the model.

Run ```python3 train.py --help``` for more information about each option that's available for you to use, 
including other networks that you can try with the ```--arch``` flag.


You should have followed the steps above to commit more memory. If you didn't do that step and you get an error saying you are out of memory you can ensure your system overcommits memory which allows it to have more memory for the task. If you get an error like that run this in your terminal:  ```echo 1 | sudo tee /proc/sys/vm/overcommit_memory```

---
## Exporting the Network

To run your re-trained ResNet-18 model for testing you will need to convert it into ONNX format. ONNX is an open model format that supports many popular machine learning frameworks, and it simplifies the process of sharing models between tools. 

Pytorch comes with built-in support to do this, so follow these steps to export your model 

### 1. Make sure you are in the docker container and in jetson-inference/python/training/classification

 
### 2. Run the onnx export script.
```'python3 onnx_export.py --model-dir=models/cards'```

 
### 3. Look in the jetson-inference/python/training/classification/models/cat_dog folder to see if there is a new model 
called resnet18.onnx there. That is your re-trained model!

![image](https://github.com/Kook1y/card_detection/assets/173178626/861ed92a-b389-41ca-aa14-bd558046480f)
---

## Processing Images 

In order to see how your network functions you can run images through them. You can use the imageNet command line arguments to test out your re-trained network.

### 1. Exit the docker container by pressing 'Ctl + D'.
 
### 2. On your nano, navigate to the 'jetson-inference/python/training/classification' directory.

 
### 3. Use ```ls models/cards/``` to make sure that the model is on the nano. You should see a file called resnet18.onnx.

![image](https://github.com/Kook1y/card_detection/assets/173178626/f7d544b9-29b4-49f2-8118-44218821e8fd)



 
### 4. Set the NET and DATASET variables
```NET=models/cards```
```DATASET=data/cards```

 
### 5. Run this command to see how it operates on an image from the cat folder.
```imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/aceofclubs/1.jpg aoc.jpg```

 
### 6. Open VS Code to view the image output. 

![image](https://github.com/Kook1y/card_detection/assets/173178626/249fecee-ce42-4208-85c5-5617ea500499)

![image](https://github.com/Kook1y/card_detection/assets/173178626/f3af43d7-e544-4542-98fb-11232673f498)



 
### 7. If you want to process all 200 test images, follow the instructions [Click Here ](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-cat-dog.md#processing-all-the-test-images)to an external site.. You can also run the network with live video. 


## Acknowledgements
- https://www.kaggle.com/code/ggsri123/implementing-resnet18-for-image-classification
- https://www.youtube.com/watch?v=mn5QDKQ54dQ
- https://www.kaggle.com/code/ggsri123/implementing-resnet18-for-image-classification/code
- https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/resnet-18-pytorch/README.md
- https://www.mdpi.com/2072-4292/14/19/4883
- https://www.nvidia.com/en-au/autonomous-machines/embedded-systems/jetson-nano/
- https://robu.in/what-is-jetson-nano/
- https://en.wikipedia.org/wiki/Nvidia_Jetson
- https://www.hackster.io/news/introducing-the-nvidia-jetson-nano-aaa9738ef3ff
- https://visionplatform.ai/jetson-nano/
## Badges

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)



