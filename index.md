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

```python
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(384, 512, 3)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Dropout(0.2),
    
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Dropout(0.2),
    
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Dropout(0.2),
    
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=keras.optimizers.Adam(lr = 0.0001), 
              metrics=['accuracy'])
model.summary()
```

This was my original model

<img src = "https://cdn.discordapp.com/attachments/856058763894063114/868169390317781022/unknown.png" width = "600">

While there's nothing inherently wrong with my own model or the various pretrained models I used, they don't work very well with small datasets because there simply isn't enough data to train 74 million parameters. Yet, these models didn't work accurately if I only trained the final layers and weights and left the core parameters untrained.

After experimenting with my own model, and many pretrained image classification models, I decided to use VGG16 as an image classification model. It is trained on ImageNet, and I would utilize the massive dataset and the months put into its training, by only training the weights. I would not train the actual convolutional layers because my dataset is not large enough to train the model accurately. This model takes in an input image of 224px x 224px, and compares parts of the image to images of materials it had already trained on. After training for 70 epochs, the model had a validation accuracy of 80.1% and a validation loss of 0.5.
The validation loss is not ideal, but it is the best I could do with a limited dataset.

```python
from PIL import Image
width = 224
height = 224
img = Image.open('/content/image.jpg')
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

# Second Milestone

<img src = "https://cdn.discordapp.com/attachments/856058763894063114/863103776180142100/IMG_7216.JPG" width = "800">
This is the raspberry pi that the final AI model will run on.

My second milestone was to run my model on a raspberry pi 3. While previous rpi's have been very underpowered, the newer models such as the 3 and 4 have gained significant performance gains. So while training still cannot happen on the pi itself (because of the lack of an egpu), the final model CAN run on it, and predictions can be made entirely on device rather than on google colab or a separate device. 

In addition, I once again rebuilt my model. I realized that while it worked well on the test images from my dataset, those images might have been very similar to the training dataset, and had only been classified accurately due to overfitting. I found this out when trying to classify photos I took of hairbrushes and plastic cups and the model making weird predictions.

<img src = "https://cdn.discordapp.com/attachments/856058763894063114/867809699230384159/unknown.png" width = "600">

```python
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)
```

I realized regular computer vision is not the ideal method for material classification. Instead, I decided to combine traditional machine algorithm with deep learning. My final model is a sort of pipeline. The first part of this conveyer line style model is a VGG16 model that will not be trained. Instead, it will detect various features from the image, and pass them on to a RandomForest Classifier which determines whether the features detected are ones that match with cardboard, glass, or plastic, etc. 

<iframe width="560" height="315" src="https://www.youtube.com/embed/zaNmfd-J6KU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Final Milestone
  
 My final milestone was to get my code running on a web app. The code still runs on the Raspberry Pi, but now has a simple interface in the form of a website, which allows anyone on the web to input an image and get an output prediction.
 
<img src = "https://cdn.discordapp.com/attachments/856058763894063114/867834822596886538/unknown.png">

There are 2 blocks to this running website: the server and the client. 

```javascript
var express = require("express");
var app = express();

app.listen(3000, function () {
  console.log("server running on port 3000");
});
```
My server runs locally on port 3000, and utilizes Node JS and Express to create my server.

```javascript
function callD_runpyscript(req, res) {

  var spawn = require("child_process").spawn; 
  var ls = spawn("python", ["./pyscript.py"]);

  ls.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
    response = data.toString();
  //res.send(response);
  });
  return (res.redirect("/"));
};
```
Using child a process, I was able to run my python script from my Node JS server. This script contains my recognition program. it takes an image as input and outputs the predicted label. 

The client side of the code is pretty basic html code. This is just to build the visual interface such as the image selector, and the window that displays the image with its  label. 

# Reflections

Being my second AI project, I learned a lot about creating and using AI models to solve custom problems. Whenever I solved on problem, a host of new problems presented themselves. Through this process of tedious debugging and problem solving I learned that while figuring problems out can be fun, it is also quite mentally draining. But as long as I didn't burn myself out, the process of engineering my way out of a problem was very rewarding. 

# Demo Night Presentation
