Assignment 2 - A potential logical error — 5/3/26, 4:50 PM
alex:
In the train function, the hidden state is reinitialized per epoch. However, since the batches of the training set are shuffled during loading, they do not constitute a consecutive sequence. Thus, it makes sense to me to reinitialize the hidden state h after every batch instead of per epoch. 
The difference is probably not huge, but I am curious if my understanding is correct. @Yael Hamrani
alexbenari
Yael Hamrani:
Great catch Alex, since we are using the shuffle the hidden layer should be initialised every batch.