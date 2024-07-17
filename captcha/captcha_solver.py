import os.path

import pytesseract
from PIL import Image
import requests
from io import BytesIO
import string

chars_list = string.ascii_letters + string.digits + string.printable
chars_list = chars_list.replace('"', '').replace("'", '')
print(chars_list)

base_dir = os.path.dirname(__file__)


# Function to fetch and solve captcha
def solve_captcha(img_path):
    with open(img_path, "rb") as f:
        img = f.read()
    image = Image.open(BytesIO(img))
    conf = '--oem 3 --psm 8 '
    conf += f'-c tessedit_char_whitelist={chars_list} '
    # conf += f' -c tessedit_char_whitelist=#'
    # conf += f' -c edges_min_nonhole=15'
    # conf += " tessedit"
    # conf += " thresholding_method=2"
    conf += "-c rej_1Il_trust_permuter_type=0"
    # conf += "load_system_dawg=0 load_freq_dawg=0"
    text = pytesseract.image_to_string(image, lang="eng", config=conf)
    return text.strip()


# img_paths = [
#     "4.png", "c.png", "capital_c.png", "u.png", "3.png", "$.png", "#.png", "dollar.png", "dollar_bold.png",
#     "dollar_bold_ench.png", "percent.png", "dollar_custom.png", "dollar_tilted.png", "dollar_sheared.png",
#     "#_bold.png", "#_sheared.png", "custom_#.png"
# ]
img_paths = [
    "D:\Python_tests\image_manip_cv\\1259.png", "at.png"
]

# img_paths = []

# for factor in range(0, 50, 5):
#     factor_final = factor*0.01
#     img_path = f"#_{factor_final}_sheared.png"
#     img_paths.append(img_path)

# Solving the captcha
for path in img_paths:
    solved_text = solve_captcha(path)
    print(f"Path: {path}; Solved Captcha: {solved_text}")
