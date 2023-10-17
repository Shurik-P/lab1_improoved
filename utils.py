import base64
import io
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def pil_to_b64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format='JPEG')
    image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return image_b64


def build_hist(img: np.ndarray):
    colors = 'bgr'
    labels = ('Blue', 'Green', 'Red')
    for i, col in enumerate(colors):
        hist, _ = np.histogram(img[:, :, i], 255)
        plt.plot(hist, color=col, label=labels[i])
    plt.legend()
    plt.grid()
    plt.xlabel('Color')
    plt.ylabel('Pixels')
    plt.tight_layout()
    _bytes = io.BytesIO()
    plt.savefig(_bytes, format='jpg')
    plt.close()
    _bytes.seek(0)
    base64_str = base64.b64encode(_bytes.read()).decode('utf-8')
    return base64_str


def scale_image(image: Image.Image, scale) -> Image.Image:
    new_w = image.size[0] * scale
    new_h = image.size[1] * scale
    return image.resize((new_w, new_h))
