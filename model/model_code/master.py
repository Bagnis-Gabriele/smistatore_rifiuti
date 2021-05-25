import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2 as cv
import pyttsx3


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
#image = Image.open('test_photo.jpg')


def SpeechText(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    
capture = cv.VideoCapture(0)
soglia = 0.9

materiale = "niente"
pre_materiale = "niente"

while True:
    #resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
    isTrue,image = capture.read()
    
    im_pil = Image.fromarray(image)
    size = (224, 224)
    image = ImageOps.fit(im_pil, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    cv.imshow('Video',np.asarray(im_pil))
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    
    if prediction[0,3] >soglia:
        materiale="plastica"
        
    elif prediction[0,1] >soglia:
        materiale = "ferro"
        
    elif prediction[0,4] >soglia:
        materiale = "bio"
        
    elif prediction[0,2] >soglia:
        materiale = "vetro"
        
    elif prediction[0,0] >soglia:
        materiale = "carta"
        
    

    if materiale != pre_materiale:
        print(materiale)
        SpeechText(materiale) 

    pre_materiale = materiale


    if cv.waitKey(27) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

