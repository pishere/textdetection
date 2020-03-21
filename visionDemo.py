import os, io
from google.cloud import vision
from google.cloud.vision import types
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"serviceaccounttoken.json"

client = vision.ImageAnnotatorClient()


def detectText(img):
    with io.open(img, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    df = pd.DataFrame(columns=["locale", "description"])
    for text in texts:
        df = df.append(
            dict(
                locale=text.locale,
                description=text.description
            ),
            ignore_index=True
        )
    return df


FILE_NAME = "1.jfif"
FOLDER_PATH = r"C:\Projects\visionAPI"
print(detectText(os.path.join(FOLDER_PATH, FILE_NAME)))