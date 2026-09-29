
import tkinter as tk
from tkinter import scrolledtext
import cv2
import numpy as np
from tensorflow import python
from tensorflow.keras.models import load_model
import pyttsx3
import speech_recognition as sr
import os
import threading
import time

# ---------------- LOAD MODEL ----------------
model = load_model("model/sign_model.h5")
labels = sorted(os.listdir("dataset"))

# ---------------- TTS ----------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ---------------- STT ----------------
recognizer = sr.Recognizer()

# ---------------- GUI ----------------
root = tk.Tk()
root.title("EduSign - AI Communication System")
root.geometry("700x500")

text_box = scrolledtext.ScrolledText(root, width=70, height=12)
text_box.pack(pady=10)

# ---------------- VARIABLES ----------------
camera_running = False
speak_enabled = False



# ---------------- TEXT TO SPEECH ----------------

engine = pyttsx3.init(driverName='sapi5')

engine.setProperty('rate', 150)

voices = engine.getProperty('voices')

engine.setProperty('voice', voices[0].id)

def text_to_speech():

    text = text_box.get("1.0", tk.END).strip()

    if text == "":
        return

    try:

        engine.stop()

        root.update()

        engine.say(text)

        engine.runAndWait()

    except Exception as e:

        text_box.insert(
            tk.END,
            f"TTS Error: {e}\n"
        )



# ---------------- SPEECH TO TEXT ----------------

def speech_to_text():

    def listen():

        try:

            with sr.Microphone() as source:

                text_box.insert(
                    tk.END,
                    "\nSpeak Now...\n"
                )

                recognizer.adjust_for_ambient_noise(
                    source,
                    duration=1
                )

                audio = recognizer.listen(
                    source,
                    timeout=10,
                    phrase_time_limit=10
                )

                text_box.insert(
                    tk.END,
                    "Recognizing...\n"
                )

                text = recognizer.recognize_google(audio)

                text_box.insert(
                    tk.END,
                    "You Said: " + text + "\n"
                )

        except sr.WaitTimeoutError:

            text_box.insert(
                tk.END,
                "No speech detected\n"
            )

        except sr.UnknownValueError:

            text_box.insert(
                tk.END,
                "Could not understand audio\n"
            )

        except sr.RequestError:

            text_box.insert(
                tk.END,
                "Internet connection required\n"
            )

        except Exception as e:

            text_box.insert(
                tk.END,
                f"STT Error: {e}\n"
            )

    threading.Thread(
        target=listen,
        daemon=True
    ).start()


# ---------------- CAMERA FUNCTION ----------------

def start_camera():

    global camera_running

    if camera_running:
        return

    camera_running = True

    cap = cv2.VideoCapture(0)

    last_prediction = ""

    sentence = ""

    last_time = time.time()

    space_added = False

    while camera_running:

        ret, frame = cap.read()

        if not ret:
            break

        # ROI
        roi = frame[100:300, 100:300]

        img = cv2.resize(roi, (64, 64))

        img = img / 255.0

        img = np.reshape(img, (1, 64, 64, 3))

        # Prediction
        prediction = model.predict(img, verbose=0)

        index = np.argmax(prediction)

        confidence = np.max(prediction)

        # Detect only if confidence is high
        if confidence > 0.95:

            result = labels[index]

        else:

            result = ""

        current_time = time.time()

        # Store detected letters
        if result != "" and result != last_prediction and current_time - last_time > 2:

            sentence += result

            last_prediction = result

            last_time = current_time

            space_added = False

        # Add automatic space after delay
        if current_time - last_time > 4 and not space_added:

            sentence += " "

            space_added = True

        # Display sentence on camera
        cv2.putText(
            frame,
            sentence,
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        # Draw rectangle
        cv2.rectangle(
            frame,
            (100, 100),
            (300, 300),
            (255, 0, 0),
            2
        )

        cv2.imshow("Sign Detection", frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()

    camera_running = False


# ---------------- CAMERA THREAD ----------------
def run_camera_thread():
    threading.Thread(target=start_camera, daemon=True).start()

# ---------------- CLEAR TEXT ----------------
def clear_text():
    text_box.delete("1.0", tk.END)

# ---------------- BUTTONS ----------------
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame,
          text="Start Camera",
          width=15,
          command=run_camera_thread).grid(row=0, column=0, padx=10)

tk.Button(btn_frame,
          text="Text to Speech",
          width=15,
          command=text_to_speech).grid(row=0, column=1, padx=10)

tk.Button(btn_frame,
          text="Speech to Text",
          width=15,
          command=speech_to_text).grid(row=0, column=2, padx=10)

tk.Button(btn_frame,
          text="Clear Text",
          width=15,
          command=clear_text).grid(row=0, column=3, padx=10)

# ---------------- RUN GUI ----------------
root.mainloop()

