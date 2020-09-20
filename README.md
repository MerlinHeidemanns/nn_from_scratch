# Teaching myself to code a neural net from the ground up

This project is entirely personal amusement. It resulted from an assignment to code a neural net and use SGD to recover a simple image which at the time didn't
work out too well due to a lack of time. I am intending to implement most of the things in Goodfellow et al.'s Deep Learning and then maybe continue with something
else.

## Log

### September 17th, 2020

* Started BatchNorm implementation, realized that current implementation
of layers wouldn't work well with it. Have to separate linear and non-linear 
part and put the gradient descent algorithms maybe in the base module class.

### September 19th, 2020

* Added separate activation layer. Batch norm next.