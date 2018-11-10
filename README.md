# scenic-recursion
Predicting scenes for Miniplaces challenge

### Please see report.pdf for more details.

Goal is to classify "scenes" from images. A "scene" is a single label that describes semantically the location of the image. For instance, a scence could be a traffic intersection, church, or classroom, or party.

The idea was to implement "visual attention". Humans seem to understand scenes by scanning several small parts of the image ("glimpses"), understanding at some level what those parts are, and then combining the semantic understanding of several glimpses together to generate an idea of what the image as a whole is describing.

In natural language processing, state-of-the-art models use a similar form of "attention", for instance, to add weights to each word in a sentence (for machine translation).
Here, we draw inspiration from those networks but add convolutional networks to semantically identify subscenes from glimpses. These glimpses are analyzed in sequence, and then weights are learned for the semantic classification of each glimpse so the model can learn which parts of a scene are important for the final classification.

However, the model is incomplete because the gradient update step is not defined between the "emission" layer and the "glimpse" layer. This results in the model not knowing how to determine the location of each glimpse in the image.
Because it was the final project for a graduate course, I did not have time to figure out the math and implementation of that step.

Relevant code is in miniplaces_code.

Much of the code is probably out of date too. It was built with the first released version of Tensor Flow in Fall 2015. Much of it could probably be implemented simpler in Keras now as well.
