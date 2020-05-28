from keras.engine.saving import load_model
from keras.datasets.mnist import load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
import logging
logging.getLogger("matplotlib").disabled = True

max_num = 5_000

(x_train, y_train), (_, _) = load_data()
x_train, y_train = shuffle(x_train, y_train)
X = x_train[:max_num]

color_dict = {0: (0, 0, 0), 1: (1, 0, 0),
              2: (0, 1, 0), 3: (0, 0, 1),
              4: (1, 1, 0), 5: (0, 1, 1),
              6: (1, 0, 1), 7: (0.1, 0.5, 0.25),
              8: (0.75, 0.1, 0.2), 9: (0.6, 0.2, 0.7)}

is_done = [False] * 10
encoder = load_model('run/weights/encoder-2D.h5')

with tqdm(total=max_num) as pbar:
    for x, y in zip(X, y_train[:max_num]):
        x = np.reshape(x, (1, 28, 28, 1)).astype('float32') / 255.0
        result = encoder.predict(x)[0]
        color = np.array(color_dict[y]).reshape(1, -1)

        if not is_done[y]:
            plt.scatter(result[0], result[1], c=color, label=y, alpha=0.8)
        else:
            plt.scatter(result[0], result[1], c=color, alpha=0.6)

        is_done[y] = True
        pbar.update()
plt.legend()
plt.show()
