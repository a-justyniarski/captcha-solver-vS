from io import BytesIO

import keras_ocr
print("Before pipeline")
pipeline = keras_ocr.pipeline.Pipeline()

print("After pipeline")

images = [
    "4.png", "c.png", "capital_c.png", "u.png", "3.png", "$.png", "#.png", "dollar.png", "dollar_bold.png",
    "dollar_bold_ench.png", "percent.png"
]

predictions = pipeline.recognize(images)

for image, prediction in zip(images, predictions):

    print(f"For image: {image} : {prediction}")

