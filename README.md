# kaggle_learning -- Digit Recognizer
------
## tips
there are two files that can be run,Al_cnn.py and Digit*.py
## process
---
Code that created by theano is based LeNet,I am not choose the model that 
corresponds to the min-cost.Model is trained on the cpu, so I cancel the 
validation during the each train epoch, and this make the code runs about 
less time than train-validation.
first, I use the raw dataset, accuracy=0.97886
then, I think data should be 1-0,
so I make the value who is larger than zero be 1,others is zero,
accuracy=0.97700
and I scala the data to [0, 1], 
accuracy=0.98043;
when I use the all train data to train the model,
accuracy=0.98371
## Improve
Change the structure of the LeNet、learning_rate、 add normalizer
  


