import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.oauth2 import service_account


# Instantiates a client
credentials = service_account.Credentials.from_service_account_file(
    'doraoauth-1e8e4205584a.json')
client = vision.ImageAnnotatorClient(credentials=credentials)

def getXY(xy):
    string = str(xy)
    if "x" in string:
        x_index = string.index("x:") + 2
        x_value = int(string[x_index:string.index("\n", x_index)])
    else:
        x_value = 0

    if "y" in string:
        y_index = string.index("y:") + 2
        y_value = int(string[y_index:])
    else:
        y_value = 0

    return [x_value, y_value]


def run_quickstart() -> vision.EntityAnnotation:
    """Provides a quick start example for Cloud Vision."""

    # The name of the image file to annotate
    file_name = os.path.abspath(
        '/home/ubuntu/dora/dora-sam/imgs/88_0.jpg')

    # Loads the image into memory
    with open(file_name, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    print('Labels:')
    for label in labels:
        print(label.description)

    return labels

def detect_text(path):
    """Detects text in the file."""
    with open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')
    print(response.full_text_annotation.pages[0].blocks[0].paragraphs[0].words[0].symbols)

    for text in texts:
        print(f'\n"{text.description}"')

        vertices = ([f'({vertex.x},{vertex.y})'
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

def get_words_boxes(path):    
    """返回word的左上和右下顶点[x,y,x,y]以及识别出的字符"""
    with open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    # 获取最顶层page
    response_page = response.full_text_annotation.pages[0]
    # 进行blocks->paragraphs->words遍历得到所有的words
    words = []
    for b in response_page.blocks:
        for p in b.paragraphs:
            for w in p.words:
                words.extend(w.symbols)
    output = []
    for w in words:
        xyxy = []
        xyxy.extend(getXY(w.bounding_box.vertices[0]))
        xyxy.extend(getXY(w.bounding_box.vertices[2]))
        output.append({
            "xyxy": xyxy,
            "text": w.text
        })
    return output

if __name__ == "__main__":
    # run_quickstart()
    # detect_text('imgs/118_1.jpg')
    get_words_boxes('imgs/dora_1.jpg')
