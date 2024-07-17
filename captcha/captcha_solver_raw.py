from PIL import Image
from pytesseract import TesseractError
from scipy.ndimage import gaussian_filter
import numpy
import pytesseract
from PIL import ImageFilter


def solve_captcha(filename):
    # thresold1 on the first stage
    th1 = 140
    th2 = 140  # threshold after blurring
    sig = 1.5  # the blurring sigma
    from scipy import ndimage

    original = Image.open(filename)
    final = original.filter(ImageFilter.EDGE_ENHANCE_MORE)
    final = final.filter(ImageFilter.SHARPEN)
    final.save("final.png")
    conf = '--psm {psm} --oem {oem} '
    # conf += '-c tessedit_char_whitelist=abcdefuABC34$#'
    try:
        value = pytesseract.image_to_string(Image.open('final.png'),
                                            lang='eng',
                                            config=conf.format(psm=8, oem=3)).strip()

        print(f"RESULT OF CAPTCHA:")
        print(value)
        print("===================")
    except TesseractError as e:
        print(f"Tesseract error: {e}")
    return


solve_captcha("captcha.png")
