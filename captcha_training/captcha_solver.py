import os

import pytesseract
from PIL import Image
import requests
from io import BytesIO
import string

chars_list = string.ascii_letters + string.digits + string.printable
chars_list = chars_list.replace('"', '').replace("'", '')


# Function to fetch and solve captcha
def solve_captcha(img_path):
    with open(img_path, "rb") as f:
        img = f.read()
    image = Image.open(BytesIO(img))
    conf = '--oem 3 --psm 6 '
    # conf += f'-c tessedit_char_whitelist={chars_list} '
    text = pytesseract.image_to_string(image, config=conf)
    text = "".join(text.strip().split())
    image.show(title=f"Solved text: {text}")
    return text


img_paths = [
    "large_negative_transformed_10.png"
]


base_dir = os.path.dirname(__file__)

# files = next(os.walk(os.path.dirname(__file__)))
# files = filter(lambda x: ".png" in x, files[2])

files = [os.path.join(base_dir, filename) for filename in img_paths]

# Solving the captcha
for path in files:
    solved_text = solve_captcha(path)
    print(f"From img: {path}: {solved_text}")
