import cv2
import numpy as np


def binarization(value, size):
    # The values will be held in this array..
    array = []
    # the corresponding index bit is set to one...
    num = 2 ** value

    # By shifting the numeric value we are storing its corresponding bits from left to right
    for index in range(0, size):
        array.append(float((num >> index) & 0x01))

    return array


X, Y = [], []
renkler = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255),
           3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255), 6: (255, 255, 255)}

numbers = {"sifir": 0, "bir": 1, "iki": 2, "uc": 3, "dort": 4, "bes": 5, "alti": 6, "yedi": 7,
           "sekiz": 8, "dokuz": 9, "A": 'A', "B": 'B', "C": "C", "D": 'D', "E": 'E', "F": 'F',
           "G": 'G', 'H': 'H', "I": 'I', "K": 'K', "M": "M", "N": "N", "P": "P", "S": "S", "T": "T",
           "V": "V", "Y": "Y", "Z": "Z", "R": "R", "CH": "CH", "SH": "SH", "U": "U", "J": "J", "L": "L"}

resim_directory = "CharacterCreater/"
rakamlar = ["sifir", "bir", "iki", "uc", "dort", "bes", "alti", "yedi", "sekiz", "dokuz", "A",
            "B", "C", "D", "E", "F", "G", "H", "I", "K", "M", "N", "P", "S", "T", "V", "Y", "Z",
            "R", "CH", "SH", "U", "J", "L"]
category_hot_vector = {element: binarization(ind, len(rakamlar)) for ind, element in enumerate(rakamlar)}

for kategory in rakamlar:

    for number in range(0, 50):
        yol = resim_directory + kategory + str(number) + ".jpg"

        resim = cv2.imread(yol, cv2.IMREAD_GRAYSCALE)

        arr = []

        for a in range(0, 28):
            for b in range(0, 28):

                if resim[a, b].all() > 0:
                    arr.append(1)
                else:
                    arr.append(0)

        X.append(np.array(arr, dtype=np.float32).reshape((28,28,1)))

        Y.append(category_hot_vector[kategory])

print("Images have been loaded")
