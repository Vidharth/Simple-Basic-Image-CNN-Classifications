import pickle
from keras.models import load_model
from imutils import paths
import cv2
from helpers import resize_to_fit
import numpy as np

model_pics = "simple_captcha_model.hdf5"
model_labels = "model_labels.dat"
captcha_image_folder = "testing"

with open(model_labels, "rb") as f:
    lb = pickle.load(f)

model = load_model(model_pics)

captcha_image_files = list(paths.list_images(captcha_image_folder))
captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)

for image_file in captcha_image_files:

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    letters = []

    for contour in contours:

        (x, y, w, h) = cv2.boundingRect(contour)

        if w/h > 1.25:

            half_w = int(w/2)
            letters.append((x, y, half_w, h))
            letters.append((x + half_w, y, half_w, h))

        else:

            letters.append((x,y,w,h))

    if len(letters) != 4:
        continue

    letters = sorted(letters, key=lambda x:x[0])

    output = cv2.merge([image]*3)
    predictions = []

    for letter in letters:

        x, y, w, h = letter

        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

        letter_image = resize_to_fit(letter_image, 20, 20)

        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        prediction = model.predict(letter_image)

        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))

    cv2.imshow("Output", output)
    cv2.waitKey()
