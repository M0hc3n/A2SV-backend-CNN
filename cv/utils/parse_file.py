from PIL import Image
import io

def get_path_from_file(file):
    imagePil = Image.open(io.BytesIO(file.read()))
    imageBytesIO = io.BytesIO()
    imagePil.save(imageBytesIO, format="JPEG")
    imageBytesIO.seek(0)

    return imageBytesIO