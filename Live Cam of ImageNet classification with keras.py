'''This Program uses VGG16 model to predict the image captured by camera.
Press ESC to capture next image and "q" to quit. '''

# import the necessary packages
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
import numpy as np
import argparse
import cv2
from gtts import gTTS 
from os import system
 
imageName = 'cam.jpg'

# Language in which you want to convert 
language = 'en'

# load the VGG16 network pre-trained on the ImageNet dataset
print("[INFO] loading network...")
#Load VGG model 
model = VGG16(weights="imagenet")

cap = cv2.VideoCapture(0)

#wait to stablize camera 
cv2.waitKey(100)

loop = True

while  loop:
	ret,frame = cap.read()

	if ret is False:
		break

	frame = cv2.flip(frame, 1)
	cv2.imwrite(imageName,frame)

	print("[INFO] loading and preprocessing image...")
	image = image_utils.load_img(imageName, target_size=(224, 224))
	image = image_utils.img_to_array(image)

	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)


	print("[INFO] classifying image...")
	preds = model.predict(image)
	P = decode_predictions(preds)
	# loop over the predictions and display the rank-5 predictions +
	# probabilities to our terminal
	for (i, (imagenetID, label, prob)) in enumerate(P[0]):
		print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
	# load the image via OpenCV, draw the top prediction on the image,
	# and display the image to our screen
	orig = cv2.imread(imageName)
	(imagenetID, label, prob) = P[0][0]
	cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

	system('say '+ label)
 
	cv2.imshow("Classification", orig)

	# Press ESC for next image 
	ch = cv2.waitKey(0)

	#Press "q" to exit 
	if ch  == ord('q'):
		loop = False


os.remove(imageName)
cap.release()

