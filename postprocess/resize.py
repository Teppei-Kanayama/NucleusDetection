import cv2
import numpy as np


def to_square(image, size=640):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    if height > width:
        resized_image = cv2.resize(image, None, fx = size / height, fy = size / height, interpolation = cv2.INTER_NEAREST)
        mirroring_image = cv2.flip(resized_image, 1)
        combined_image = cv2.hconcat([mirroring_image, resized_image, mirroring_image])
        output = np.zeros((size, size, channels), dtype = "uint8")
        combined_width = combined_image.shape[1]
        if combined_width < size:
            n = int((size - combined_width) / 2)
            output[::, n:n+combined_width] = combined_image
        else:
            n = int((combined_width - size) / 2)
            output = combined_image[::, n:n+size]
    elif height < width:
        resized_image = cv2.resize(image, None, fx = size / width, fy = size / width, interpolation = cv2.INTER_NEAREST)
        mirroring_image = cv2.flip(resized_image, 0)
        combined_image = cv2.vconcat([mirroring_image, resized_image, mirroring_image])
        output = np.zeros((size, size, channels), dtype = "uint8")
        combined_height = combined_image.shape[0]
        if combined_height < size:
            n = int((size - combined_height) / 2)
            output[n:n+combined_height, ::] = combined_image
        else:
            n = int((combined_height - size) / 2)
            output = combined_image[ n:n+size, ::]
    else:
        output = cv2.resize(image, (size, size), interpolation = cv2.INTER_NEAREST)

    return output


def from_square(image, shape):
    height = shape[0]
    width = shape[1]
    size = image.shape[1]

    if height >= width:
        resized_width = int(width * size / height + 0.5)
        n = int((size - resized_width) / 2)
        resized_image = image[::, n:n + resized_width]
    elif height < width:
        resized_height = int(height * size / width + 0.5)
        n = int((size - resized_height) / 2)
        resized_image = image[n:n+resized_height, ::]

    resized_image = cv2.resize(resized_image, (width, height))

    return resized_image
