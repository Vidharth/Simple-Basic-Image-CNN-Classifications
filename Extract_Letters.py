import cv2
import os
import glob

captcha_image_folder = "generated_captcha_images"
output_folder = "extracted_letters"

captcha_image_files = glob.glob(os.path.join(captcha_image_folder, "*"))
counts = {}

for (i, captcha_image_file) in enumerate(captcha_image_files):

    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    org = cv2.imread(captcha_image_file)
    gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 9, 9, 9, 9, cv2.BORDER_REPLICATE)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    letters = []

    for contour in contours:

        (x, y, w, h) = cv2.boundingRect(contour)

        if w/h > 1.25:

            half_w = int(w/2)

            letters.append((x, y, half_w, h))
            letters.append((x + half_w, y, half_w, h))

        else:

            letters.append((x, y, w, h))

    if len(letters) != 4:
        continue

    letters = sorted(letters, key=lambda x: x[0])

    for letter, letter_name in zip(letters, captcha_correct_text):

        x, y, w, h = letter

        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        save_folder = os.path.join(output_folder, letter_name)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        count = counts.get(letter_name, 1)
        image_name = os.path.join(save_folder, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(image_name, letter_image)

        counts[letter_name] = count + 1

print(counts)
