import cv2
from scipy.spatial import distance

from palette import get_color


class ReductionDetail():
    """ Resizing an Image to Reduce Detail """
    def __init__(self):
        pass

    def reduction(self, image):
        orig_height = image.shape[0]
        orig_width = image.shape[1]
        small_height, small_width = orig_height // 16, orig_width // 16
        image_resized = cv2.resize(image, (small_width, small_width), interpolation=cv2.INTER_LINEAR)
        image_resized = cv2.resize(image_resized, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
        return image_resized

class Quant():
    """ Reducing the number of colors """
    def __init__(self):
        pass

    def choose_closest(self, pixel, colors_list):
        dists = []
        for color in colors_list:
            dists.append(distance.euclidean(color, pixel)) # Calculate distance between colors
        id_of_closed = dists.index(min(dists)) # Choose the closest color id
        return id_of_closed

    def quantize_image(self, image):
        colors_list = get_color(image)
        colors_list = [(x[2], x[1], x[0]) for x in colors_list]  # Convert RGB to BGR for Opencv
        image_quantized = image.copy()
        for i in range(image.shape[0]): # on height
            for j in range(image.shape[1]): # on width
                closest_id_in_palette = self.choose_closest(image_quantized[i, j], colors_list)
                image_quantized[i, j] = colors_list[closest_id_in_palette]
        return image_quantized

    def reduction(self, image):
        orig_height = image.shape[0]
        orig_width = image.shape[1]
        small_height, small_width = orig_height // 16, orig_width // 16
        image_resized = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        # image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_result = self.quantize_image(image_resized)
        image_result = cv2.resize(image_result, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
        # image_result = cv2.cvtColor(image_result, cv2.COLOR_RGB2BGR)
        return image_result


if __name__ == '__main__':
    image = cv2.imread('data/eye.jpg')
    cv2.imshow('pixel_orig', image)
    filter = ReductionDetail()
    image = filter.reduction(image)
    cv2.imshow('pixel2', image)
    filter = Quant()
    image = filter.reduction(image)
    cv2.imshow('pixel3', image)
    cv2.waitKey(0)





