Introduction
The presented project is one asset of the computer vision framework that is part of the telepresence toolkit, which is been developed for the mobile robot platform PlatypOUs, from the laboratory of robotics of  Obuda University (IROB).
This project describes the workflow required to create a convolutional neural network (CNN) for object detection in images obtained from USB cameras.  Going from the image capture and labeling process, to the proper configuration and model exportation for further implementation. 
The complete set of steps has been condensed in a Jupyter Notebook (Anaconda) to ensure the correct reproduction of the model, or to simplify the iterative process of development of new models. 
Note: as the target object to be detected can variate regarding the requirements of the application, this work intends to create an algorithm to help in the creation of models to detect objects “on-the-go” reducing the time required to provide a mobile platform with this capabilities avoiding time looses in the development process of any application. 

Data recollection
The development of a CNN model to detect objects starts with the decision of what objects are we interested to recognize, here we determine how many classes we should create in our model, as well as how many samples we will use for training and testing our model. In this specific scenario, we are going to create a model to detect 4 different “objects” from the scene, we will detect sign language gestures (ThumbsUp, ThumbsDounw, LiveLong, ThankYou ) and will display its meaning.
The first Jupyter Notebook we are going to use is the “Image Collection.ipynb” where the first step is to install the required dependencies that are not included by default with the installation of anaconda and python: OpenCv, uuID. 
 ![image](https://github.com/user-attachments/assets/3bc175bc-236c-4726-9cf0-8f1f5eb34d9c)

To posteriorly define the amount and labels of the classes, along with the number of samples we are going to take for each class. 
 ![image](https://github.com/user-attachments/assets/89601d1a-0891-4237-9f52-e9522c0b9ec8)


Image labeling 
In order to label the images that we have collected, we need to use the LabeIImag algorithm from Tzutalin github (can be found here https://github.com/tzutalin/labelImg)
Once we have successfully downloaded the repository we need to compile and run the python script with the name LabelImg.py
 ![image](https://github.com/user-attachments/assets/ecb326b1-3c85-442d-b930-8b95c44f10a9)

This will open a windows similar as the following, where we need to open the directories where the images were saved, mark the Region Of Interest (ROI) and we insert the label we have previously chosen for the selected class.
 ![image](https://github.com/user-attachments/assets/c90a7b88-1436-4e51-a164-b97449e0a211)

After performing this action along all the image sets we created, we are ready to move the images and labels into two different folders, the first one will contain a portion of the samples to “Train” the model, and a second section to contain images to use in the “testing” or validation of the model. 
![image](https://github.com/user-attachments/assets/4882b383-0446-4825-8d72-df3cb3d15b2c)

Note: the more samples we provide showing different angles or points of view of the desired object, the better the resultant certainty of the model while recognizing objects.
TensorFlow 2 Detection Model Zoo
TensorFlow offers a wide range of pretrained models in their repository “Models Zoo”, where the huge advantage is that this is a collection of State of the Art Models that have been previously implemented and analyzed, giving us an idea of the general performance of the selected model prior its implementation or retraining process. 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
In the previous link we can find a variety of available models that has been already tested and its performance is documented in a table as follows: 
![image](https://github.com/user-attachments/assets/110747a3-ce7a-4bd6-bcb0-ee9f4068de94)


Download TensorFlow Models API
The TensorFlow repository contains its API to manage and modify the model of our selection, the set of codes can be found in the folder: https://github.com/tensorflow/models
Install TensorFlow-Slim image classification model library 
In order to compile the Model library, the Protobuf package can be downloaded from the following link:
https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip
After downloading the Zip folder, extract it in your preferred place and include the “bin” PATH in the environmental variables of your system (Windows) and compile using the following command: 
![image](https://github.com/user-attachments/assets/deaff0d2-dda3-4266-b5a0-7ed40590d6a3)

In order to verify the successful installation of the libraries, we can run a python script that perform and test all the required assets. The mentioned script can be found inside the “models” folder we extracted, in this scenario the path of the file is: 
'''\research\object_detection\builders\model_builder_tf2_test.py '''
If all the dependencies and required libraries are successfully installed in the environment, we should get a prompt output showing “OK” as in the following image. 
![image](https://github.com/user-attachments/assets/1a0e59c8-dda6-4810-9a14-225a383a866f)
 
Once we have got the OK sign we are ready to select and modify a model from the TensorFlow MD Zoo.
For the purpose of this explanation, we are going to use an SSD_mobilenet_v2 model that can be found in the same github:
http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz 
Which structure and configuration we have downloaded from the TensorFlow repository and extracted into a “pretrained model” folder as the following code states: 
![image](https://github.com/user-attachments/assets/781091c1-1a9c-42fc-9d38-6bfa8856503a)
 
After extracting the files contained, we can modify the configuration of the model we downloaded to fit the new configuration of the model we want to train. For this purpose, we can copy the “pipeline.config” file inside the folder we will use for our model 
 
If we check the content of this configuration file we will see relevant information about the SSD model we are about to retrain. As we can notice in the following image, this model contains 90 classes and the input image size is 320x320 pixels.
![image](https://github.com/user-attachments/assets/86e95775-b1c0-46d2-873f-9436345c6604)
 

In order to train the model with our selected classes we need to update the configuration file included in our folder. This is performed by tensorflow using the following code: 
-	First read the pipeline configuration:
![image](https://github.com/user-attachments/assets/00c3ecca-68a9-436a-b3b3-29d4ba8ca831)
 
-	Update the values to match the model we want to train and the path of the folders where we have selected the checkpoint and records (input and output), in this scenario we have 4 classes
![image](https://github.com/user-attachments/assets/a13adfc5-83d1-432a-ac57-ff9f34222a58)
 
-	Rewrite the new configuration parameters
![image](https://github.com/user-attachments/assets/816a9ce7-78bd-42c1-a265-970b00cbeff5)

 

Now we can check the resultant configuration file we have updated. 
![image](https://github.com/user-attachments/assets/e28d23a4-148e-408b-a6a0-ecbf2e985243)
 
In this point we are ready to train our SSD model, for which we need to run the following command pointing the original model and our new configuration file. 
Command: 
''' python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --checkpoint_dir=Tensorflow\workspace\models\my_ssd_mobnet '''

the time it will take to get the job done will depend on the number of classes and samples we have provided, as well as the computer hardware.
![image](https://github.com/user-attachments/assets/9efe31ff-9212-4b92-ac2d-9d1a1748fb60)
 

Once the process is done, we can check the performance and evaluate the model using TensorBoard, were we can analyze the time series result of the training.
![image](https://github.com/user-attachments/assets/7662b24a-4594-47b2-abc1-a754a4091d37)
![image](https://github.com/user-attachments/assets/bbabc341-40e9-4348-a09e-cf210f03145c)
 ![image](https://github.com/user-attachments/assets/e552bf9d-28ff-4e7a-8022-7bf09fedb5a1)



 

As well as the resultant images applied to train our model: 
![image](https://github.com/user-attachments/assets/dc751876-0f5d-437b-a6b7-86e1e9af8bde)
 

On the other side, in the evaluation folder of our model we have information related to the performance of our model during the test process. 
![image](https://github.com/user-attachments/assets/4f255863-f2dc-49f2-9ddd-bd8fe87f630a)

![image](https://github.com/user-attachments/assets/5f4a9a55-ea25-4c17-b6d1-8305dc4c1b31)

 
 ![image](https://github.com/user-attachments/assets/dfe1536b-aa4b-474e-8df4-ae322cf46057)

 
As well as the results applying the test image set we provided. 
 ![image](https://github.com/user-attachments/assets/5b1b0c43-cfe8-4cb4-86f8-bdef21537cb6)
 ![image](https://github.com/user-attachments/assets/ecc20c0b-c13c-4b4e-9143-40a1e11aec0b)
![image](https://github.com/user-attachments/assets/706b853c-f73e-4d59-afe5-026087f4c45d)
 
As it can be notices in the first group of images we have gotten a false “ThumbsUp” recognition with a 58% of certainty, one method to avoid this error is to increase the certainty threshold to avoid false positives. 
