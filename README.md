# NeuralNetAttack - 2023
An updated version of the regularized FGSM attack, this time aimed at ResNet50. This code was used in a session on adversarial attacks. No external-weights dependencies needed, and the code is much simpler now (using tensorflow v2).


# NeuralNetAttack - 2018
Attacking AlexNet (a convolutional neural network for object identification) to make it identify an image of a truck as a toaster with very high confidence and minimal observable-change of the image.


This code was used in the following (hebrew) article at Digital-Whisper magazine of July 2018: https://www.digitalwhisper.co.il/files/Zines/0x60/DW96-1-ToasterTruck.pdf


### Notes:       
* Make sure to download the network's weights ("bvlc_alexnet.npy" file).
* AlexNet code is based on: http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/  (the weights are also there).
