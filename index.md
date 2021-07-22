# Materials detection model
For my project at BSE, I wanted to create an AI model which can identify various materials from an input image. While there are already models that can detect objects, I have not seen any that can detect the specific materials used on the object. The model can currently predict the following materials: paper, cardboard, glass, metal, wood, plastic, and skin. This has applications from better self-driving cars software, to helping the blind "see," in more vivid detail.

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Tejus Singh | Evergreen Valley High School | "    " | Incoming Senior

### First Milestone
Reaching a working classifier was a task that appeared easier than it actually happened to be. To have a trained model, I need a good dataset, and a model that would train on the data. 

I happened to stumble upon a garbage classification dataset, which although small, contained the images I needed to start testing. 
With my dataset found, I turned my attention to the model. I first attempted to use an untrained tensorflow model with 6 convolutional 2D layers, but after training and testing, it became very clear that I would be wasting time creating my own model. So I turned to pretrained classification models, and the best one happened to be InceptionV3. Yet, while my model would train, and reach a validation accuracy of 77%, my validation loss would not decrease, but rather stayed the same throughout training. When testing, the predictions were strangely off, and would put random probability on my class labels.



<img src = "https://cdn.discordapp.com/attachments/861646157956382750/863124447980290058/unknown.png" width = "300">
<img src = "https://cdn.discordapp.com/attachments/861646157956382750/863124684966199336/unknown.png" width = "300">

Clearly, the model hasn't trained properly despite the numbers looking good


I then tried a clusttering approach, in which the model uses K Nearest-neighbors (KNN) to group different attributes of an image. This is different from the other models which use image recognition. However, this seemed to create more errors and problems, so I switched back to using a CNN model, but this time decided to use VGG16. In addition, I realized my dataset was extremely small for the task I was trying to achieve. I didn't have the time to manually take photos of objects and classify them. [Thanks to Jason Inirio's google images to dataset tool](https://github.com/jasoninirio/BSE_Toolbox/tree/main/dataset_maker), I was able to quickly add hundreds of images to each label in my dataset. 

<img src = "https://cdn.discordapp.com/attachments/856058763894063114/865272247030775839/unknown.png" width = "600">


<img src = "https://cdn.discordapp.com/attachments/856058763894063114/867809699230384159/unknown.png" width = "600">

```
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)
```

After experimenting with my own model, and many pretrained image classification models, I realized regular computer vision is not the ideal method for material classification. Instead, I decided to combine traditional machine algorithm with deep learning. My final model is a sort of pipeline. The first part of this conveyer line style model is a VGG16 model that will not be trained. Instead, it will detect various features from the image, and as It takes in an input image of 224px x 224px, and finds various patterns in the texture, opacity, color, etc, of the object in the image. 
After training for 70 epochs, the model had a validation accuracy of 80.1% and a validation loss of 0.5.
The validation loss is not ideal, but it is the best I could do with a limited dataset.

```python
from PIL import Image
width = 224
height = 224
img = Image.open('/content/IMG_7192.jpg')
img = img.resize((width, height), Image.ANTIALIAS)
img.save('resized_image.jpg')
img = mpimg.imread(path)

x = keras.preprocessing.image.img_to_array(img)
x = normalize(x)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images)
pred = labels[np.argmax(classes)]
```
After training, this code can be used to use our model on images. It takes in the user defined image, a jpg file, and resizes it to the input size of the model. Then, the pixels of the resized image are converted into an array which can be read by the model. The image is then normalized so that all the features are predicted in a range from 0-1 This is fed into the model which outputs a predicted label.

<img src = "https://cdn.discordapp.com/attachments/856058763894063114/865269216885473301/unknown.png" width = "600">

It finally works (well, some of the time)!

<iframe width="560" height="315" src="https://www.youtube.com/embed/UEGhcrTHicw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<img src = "https://cdn.discordapp.com/attachments/856058763894063114/863103776180142100/IMG_7216.JPG">
This is the raspberry pi that the final AI model will run on.

# Final Milestone
My final milestone is to implement my model to an Android app, so that a user can detect materials with the click of one button.   

<--!final milesstone video here-->

# Second Milestone

<img src = "https://cdn.discordapp.com/attachments/856058763894063114/863103776180142100/IMG_7216.JPG">
This is the raspberry pi that the final AI model will run on.

My second milestone was to run my model on a raspberry pi 3. While previous rpi's have been very underpowered, the newer models such as the 3 and 4 have gained significant performance gains. So while training still cannot happen on the pi itself (because of the lack of an egpu), the final model CAN run on it, and predictions can be made entirely on device rather than on google colab or a separate device. 

In addition, I once again rebuilt my model. I realized that while it worked well on the test images from my dataset, those images might have been very similar to the training dataset, and had only been classified accurately due to overfitting. I found this out when trying to classify photos I took of hairbrushes and plastic cups and the model making weird predictions.

<img src = "https://cdn.discordapp.com/attachments/856058763894063114/867809699230384159/unknown.png" width = "600">

```
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)
```

I realized regular computer vision is not the ideal method for material classification. Instead, I decided to combine traditional machine algorithm with deep learning. My final model is a sort of pipeline. The first part of this conveyer line style model is a VGG16 model that will not be trained. Instead, it will detect various features from the image, and pass them on to a RandomForest Classifier which determines whether the features detected are ones that match with cardboard, glass, or plastic, etc. 

<--!2nd milesstone video here-->
# Final Milestone
  

My first milestone was setting up my raspberry pi and preparing it so that it would be able to run the model after it had been trained. Of greater accomplishment was the creation of my dataset, which was done by combining several datasets together. I replaced their labels and prepared them to be used for training. Training could not be done on the actual pi because of its limited processing power. Therefore, I used Google Colab which allowed me to quickly train my model on a powerful gpu.The accuracy of my first prototype model was around 94%. 

<--!first milesstone video here-->
