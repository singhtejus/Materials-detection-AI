# Materials detection model
For my project at BSE, I wanted to create an AI model which can identify various materials from an input image. While there are already models that can detect objects, I have not seen any that can detect the specific materials used on the object. The model can currently predict the following materials: paper, cardboard, glass, metal, wood, plastic, and skin. This has applications from better self-driving cars software, to helping the blind "see," in more vivid detail.

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Tejus Singh | Evergreen Valley High School | "    " | Incoming Senior

![Headstone Image](https://cdn.discordapp.com/attachments/856058763894063114/863103776180142100/IMG_7216.JPG)
This is the raspberry pi that the final AI model will run on.

```
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
```
After experimenting with my own model, and many pretrained image classification models, this is the one that produced the best accuracy after training on the dataset. It takes in an input image of 224px x 224px, and finds various patterns in the texture, opacity, color, etc, of the object in the image. 
After training for 70 epochs, the model had a validation accuracy of 80.1% and a validation loss of 0.5.
The validation loss is not ideal, but it is the best I could do with a limited dataset.

```
from PIL import Image

width = 384
height = 512
img = Image.open('/content/IMG_7192.jpg')
img = img.resize((width, height), Image.ANTIALIAS)
img.save('resized_image.jpg')
img = mpimg.imread(path)

x = keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images)
pred = labels[np.argmax(classes)]
```
After training, this code can be used to use our model on images. It takes in the user defined image, a jpg file, and resizes it to the input size of the model. Then, the pixels of the resized image are converted into an array which can be read by the model. This is fed into the model which outputs a predicted label.

# Final Milestone
My final milestone is to implement my model to an Android app, so that a user can detect materials with the click of one button.   

<--!final milesstone video here-->

# Second Milestone
My final milestone is the increased reliability and accuracy of my robot. I ameliorated the sagging and fixed the reliability of the finger. As discussed in my second milestone, the arm sags because of weight. I put in a block of wood at the base to hold up the upper arm; this has reverberating positive effects throughout the arm. I also realized that the forearm was getting disconnected from the elbow servo’s horn because of the weight stress on the joint. Now, I make sure to constantly tighten the screws at that joint.

<--!2nd milesstone video here-->
# Final Milestone
  

My first milestone was setting up my raspberry pi and preparing it so that it would be able to run the model after it had been trained. Of greater accomplishment was the creation of my dataset, which was done by combining several datasets together. I replaced their labels and prepared them to be used for training. Training could not be done on the actual pi because of its limited processing power. Therefore, I used Google Colab which allowed me to quickly train my model on a powerful gpu.The accuracy of my first prototype model was around 94%. 

<--!first milesstone video here-->
