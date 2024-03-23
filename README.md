# WakeWatch : An Alertness Assurance System
### Description
This project alerts the drivers in real-time and awakens them, if they ever fall asleep while driving. It plays an alarm whenever they appear to be drowsy.

The project makes use of a CNN model, that has been trained on a dataset that contains images of both eyes in open and closed state.
This model is then used to predict if the eyes of user in the frame is in Open or Closed state.

### Requirements 
Python ( [version 3.8](https://www.python.org/downloads/release/python-380/) to [version 3.10](https://www.python.org/download/releases/python-3100/) ).


### Dependencies

1) import cv2
2) import tensorflow.keras
3) import pygame
4) import numpy


### Execution
Download the .zip file of the code. Then in the terminal, type `python detection.py`
```
python detection.py
```
