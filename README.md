# Welcome to Artifical Incoherence! 

Artificial Incoherence is a podcast primarly generated by Artificial Intelligence. The goal of this podcast is to simeltaneously demistify Machine Learning and natural langauge processing while also creating funny and interesting content.

## Methods and Design

The current goal of Artificial Incoherence is to document and epxlain the process of building a Generative Adversarial Network that can create coherent stories. To achieve this we will go through quite a few iterations (this section is subject to change).
### 1. Recurrent Neural Network (RNN)
For our first iteration we will start witha  slightly sipmler RNN model. The main factor that differentiates a RNN from a standard NN is it's reliance on the previous state. A RNN ulitmately creates a generated model that has a sort of context to it's output. The rnn takes it's hidden layers previous state as an input alongside new inputs so creating a sequentially weighted ntowrk. This is particularly beneficial for Langauge processing because without context, the outputs cannot be resonably expected to conform to the complicated linear structure of natural langauge. 
 ### 2. Long Short Term Memory (LSTM)
 A LSTM Network is a slightly more advanced version of the RNN. This network is influenced and dictated by essentially the same ideology as the RNN we will start with but by bringing a technique called Backpropogation into the equation. Backpropogation in short will enable access to the hidden layer retroactievely, which can simplifiy the process of calculating the gradient descent (essentially our ideal 'direction' from a given point in our conceptual model). This addition will provide a platform for increased supervison over the model itself. With back propogation implimented we can feed information into our model as well as allow it to reflect on itself. This also enables a sort of perservation of the overall state that is referred to as Long Short Term Memory. After this iteration we will be able to feed in higher volume,more diverse data without worrying about the model falling apart.
 ### 3. Generative Adversarial Network (GAN)
 A GAN is produced from two isolated nerual networks. One of theese networks is referred to as the discriminator. The discriminator is a network that does not generate any content, the network is designed and trained to judge an input's likelyhood in comparison to a given dataset. For example discriminative network for photos of house cats might be fed a photo lion and say theres a high percentage chance that it's observing a cat, but the same network given a photo of a sports car would give a low percentage. This output from this network can be particularly useful in traning our generative network. By pointing the network that is creating output into this network we then can get information on how accurate the generated content is. This accuracy evaluation is not simply for informing designers about the network. It can be fed back into the generative network as a way to steer the algorithim to better results faster.
