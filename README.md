# PNW_Fungi
Training a model to classify fungi species from the Pacific Northwest

I thought this would be a fun project as a way of introducing myself to classification problems, as the outdoors and fungi are interests of mine.



Gathering Data: 

  I landed on choosing GBIF for gathering my data, as it had a plethora of available labeled images. I selected a region using a GEOJSON file that I considered to be the Pacific Northwest (basically Cascadia) and then GBIF created a       
  database of download links to images of Fungi labeled with their species.

Downlaoding images:

  With the help of copilot, I wrote a script to download the images labeled with their species names locally. This file is getfungi.py

Training Model: 

  My next step was to use these images to train a classifier of the fungi species. After I had downloaded 60,000 images of fungi, I thought it was time to get started with training. I wrote train_fungi.py. I decided to use the ResNet18 pre-trained model for this,   as it was recommended as a great pre-trained model for image classification. Training took a while, and I foolishly attempted to classify all images into all of their classes at first. However there were over 1000 species of fungi, and with images of similar looking mushrooms and lichen surrounded by dirt, moss, and ferns, the model had a difficult time distinguishing them, and the validation accuracy was quite poor. 
  I then tried to input some photos of non-fungi plants (also gathered from GBIF, and downloaded using getplants.py) into the model. This helped with the validation accuracy of the model.
  Training is still a work in progress, and I look forward to continue fine tuning this model to see what results I can get. 
Testing Model: 

  I then wrote a script to test the accuracy of the model by our images back into the model and seeing whether or not it correctly classified them. This is the testfungimodel.py.
Future Work:
  Once I have trained the model to accurately predict the species of fungi, I will try to develop it into an app where you can take a photo and receive a response of the models confidence of which species it is. 
  I am very much looking forward to harnessing the powers of deep learning and being humbled in the process.
