from keras.engine.saving import load_model
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import cv2


def click(event, x, y, flags, param):
    global mx, my, number_representation, mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
        latent_coordinates = np.array([mx(x), my(y)]).reshape((1, 2))
        number_representation = (decoder.predict(latent_coordinates)[0].reshape((28, 28)) * 255.0).astype('uint8')


decoder = load_model('run/weights/decoder_VAE.h5')
latent_space_image = cv2.imread('run/images/latent_space_VAE.png')

mouse_x = mouse_y = 0
mx = interp1d([170, 1222], [-3, 3])
my = interp1d([80, 630], [3, -3])

cv2.namedWindow("Latent Space")
cv2.setMouseCallback("Latent Space", click)

number_representation = None

while True:
    image_copy = latent_space_image.copy()

    cv2.putText(image_copy, "+", (mouse_x - 20, mouse_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

    cv2.imshow("Latent Space", image_copy)

    if number_representation is not None:
        number_representation = cv2.resize(number_representation, (300, 300), cv2.INTER_CUBIC)
        cv2.imshow("Latent Representation", number_representation)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
