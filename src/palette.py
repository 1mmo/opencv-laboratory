from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from PIL import ImageColor

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_color(image, n_clusters=3):
    # image = cv2.imread('data/eye.jpg')
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    clf = KMeans(n_clusters=n_clusters)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    # rgb_colors = [ordered_colors[i] for i in counts.keys()]


    # plt.figure(figsize = (8, 6))
    # plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    # print(hex_colors)
    rgb_colors = []
    for i in hex_colors:
        rgb_colors.append(ImageColor.getcolor(i, "RGB"))
    # plt.show()
    return rgb_colors