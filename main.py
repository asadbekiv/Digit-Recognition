import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


img = plt.imread('./9.jpg')
print(img.shape)
img = np.uint8(np.dot(img, [0.33, 0.33, 0.34]))
print(img.shape)


l1 = 28/img.shape[0]
l2 = 28/img.shape[1]
newimg = np.zeros((28, 28))
for x in range(28):
    for y in range(28):
        newimg[x][y] = img[int(x/l1)][int(y/l2)]
newimg = 255-newimg
# plt.imshow(newimg, cmap='gray')
# plt.show()
z = np.array(newimg)
z = z.reshape((1, 28, 28))


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if os.path.exists('mnistdigits.keras'):
    mymodel = tf.keras.models.load_model('mnistdigits.keras')
    print('loaded from cache')
else:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    a = model.evaluate(x_test, y_test)
    print(a)

    model.save('mnistdigits.keras')
    print('trained and then save')

# n = np.random.randint(1, len(x_train))
predictions = mymodel.predict([z])
# print(f'true value: {y_train[n]}')
print(predictions[0])
print(f'predicted value: {np.argmax(predictions[0])}')
plt.imshow(z[0], cmap='gray')
plt.show()