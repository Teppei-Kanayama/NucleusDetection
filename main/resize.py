import cv2
import numpy as np

def resize(image):
    height = image.shape[0]
    width = image.shape[1]

    if height > width:
        resized_image = cv2.resize(image, None, fx = 640 / height, fy = 640 / height, interpolation = cv2.INTER_NEAREST)
        mirroring_image = cv2.flip(resized_image, 1)
        combined_image = cv2.hconcat([mirroring_image, resized_image, mirroring_image])
        output = np.zeros((640, 640), dtype = "uint8")
        combined_width = combined_image.shape[1]
        if combined_width < 640:
            n = int((640 - combined_width) / 2)
            output[::, n:n+combined_width] = combined_image
        else:
            n = int((combined_width - 640) / 2)
            output = combined_image[::, n:n+640]
    elif height < width:
        resized_image = cv2.resize(image, None, fx = 640 / width, fy = 640 / width, interpolation = cv2.INTER_NEAREST)
        mirroring_image = cv2.flip(resized_image, 0)
        combined_image = cv2.vconcat([mirroring_image, resized_image, mirroring_image])
        output = np.zeros((640, 640), dtype = "uint8")
        combined_height = combined_image.shape[0]
        if combined_height < 640:
            n = int((640 - combined_height) / 2)
            output[n:n+combined_height, ::] = combined_image
        else:
            n = int((combined_height - 640) / 2)
            output = combined_image[ n:n+640, ::]
    else:
        output = cv2.resize(image, (640, 640), interpolation = cv2.INTER_NEAREST)

    return output
