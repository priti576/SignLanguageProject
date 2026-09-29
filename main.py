import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
import speech_recognition as sr

# Load model
model = load_model("model/sign_model.h5")

import os
labels = sorted(os.listdir("dataset"))
# Text to Speech
engine = pyttsx3.init()

# Speech to Text
recognizer = sr.Recognizer()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # ROI (hand area)
    roi = frame[100:300, 100:300]
    img = cv2.resize(roi, (64,64))
    img = img / 255.0
    img = np.reshape(img, (1,64,64,3))

    prediction = model.predict(img)
    index = np.argmax(prediction)
    result = labels[index]

    cv2.putText(frame, result, (100,90),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

    cv2.rectangle(frame, (100,100), (300,300), (255,0,0), 2)

    cv2.imshow("Sign Detection", frame)

    key = cv2.waitKey(1)

    # Press 's' for speech to text
    if key == ord('s'):
        with sr.Microphone() as source:
            print("Speak...")
            audio = recognizer.listen(source)

            try:
                text = recognizer.recognize_google(audio)
                print("You said:", text)

                engine.say(text)
                engine.runAndWait()

            except:
                print("Could not understand")

    # Press 't' for text to speech
    if key == ord('t'):
        engine.say(result)
        engine.runAndWait()

    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()