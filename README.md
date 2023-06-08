# Face-Mask-Detection Project
CNN are a kind of deep neural network which is typically used in deep learning to examine visual imagery. A CNN is a Deep Learning  algorithm that  would take an  image as  input, assign meaning to different parts of the image, and differentiate between them. Because of their high precision, CNNs are used for image  detection and identification. The CNN  uses a hierarchical model that builds a network in the shape of a funnel and then outputs a fully-connected layer where all the neurons are connected to  each other and the data is  stored. Artificial Intelligence  has  made  important  strides  in  bridging  the difference  between  human  and  computer  capabilities. Researchers and enthusiasts alike operate in a number of facets of  the  area  to produce  impressive performance. The  field of computer vision is one of several such fields. The goal of this area is to allow machines to see and understand the environment in the same way that humans do, and to use that information for picture  and  video  identification,  image  interpretation  and labeling,  media recreation,  recommendation  systems,  natural language  processing,  and  other  functions  are  only  a  few examples.
## Data
The data used to train the model is availbale here: 
The dataset contains, around 2000 images for both the categories i.e. with mask and without mask. The  images  in  the dataset  are  not  all  the  same  size,  so preprocessing was required for this study. The training of deep learning models necessarily requires a large amount of data, therefore data was normalized all images where converted to 100 * 100 after changing all imaged to gray scale using CV2. Further for faster calculation, images were converted to NumPy arrays, which were later passed on to the model for tarining. 
## Architecture 
* CNN layer: 200 (3*3) with relu activation
* Max pooling (2*2)
* CNN layer: 100 (3*3) with relu activation
* Max pooling (2*2)
* Flatten
* Dense layer: 50 with relu activation
* Dense layer: 2 with softmax activation
* Add the end categorical cross entropy was used to calculate the loss and ADAM optimizer was used to optimize the model.
