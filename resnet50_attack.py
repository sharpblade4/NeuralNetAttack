import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

model = ResNet50(weights='imagenet')

img = tf.keras.preprocessing.image.load_img(
        '/Users/ruurbach/Downloads/hog3.png', 
        target_size=(224, 224)
)

x = preprocess_input(np.expand_dims(
    tf.keras.preprocessing.image.img_to_array(img), axis=0)
)
decode_predictions(model.predict(x), top=20)

model.trainable = False
base_image = tf.constant(x, dtype=tf.float32)
delta = tf.Variable(tf.zeros_like(base_image), trainable=True)


def show(pimg, flip=False):
    pimg = pimg[0].numpy().copy()
    channels = []
    for channel in range(pimg.shape[-1]):
        pimg[:,:,channel] += np.mean(pimg[:,:,channel])
        pimg[:,:,channel] -= pimg[:,:,channel].min()
        pimg[:,:,channel] /= pimg[:,:,channel].max()
    plt.imshow(pimg[:,:,::-1] if flip else pimg)
    plt.show()


epsilon = 0.008
steps = 50
target = 340 # zerba
optimizer = tf.keras.optimizers.Adam(learning_rate=0.95)


def fgsm(model, base_image, delta, target, epsilon, steps):
    cce_loss_func = tf.keras.losses.CategoricalCrossentropy()
    target_one_hot = np.expand_dims(tf.keras.utils.to_categorical(target, 1000), axis=0)
    for step in range(steps):
        if step % 10 == 0: print(f"step:{step}")
        with tf.GradientTape() as tape:
            tape.watch(delta)
            adversary = preprocess_input(base_image + delta)
            prediction = model(adversary, training=False)
            step_loss = cce_loss_func(target_one_hot, prediction)

        gradients = tape.gradient(step_loss, delta)
        optimizer.apply_gradients([(epsilon*tf.sign(gradients), delta)])
    return delta

perbutation = fgsm(model, base_image, delta, target, epsilon, steps)


def reg_fgsm(model, base_image, delta, target, epsilon, steps):
    mse_loss_func = tf.keras.losses.MeanSquaredError()  ##
    cce_loss_func = tf.keras.losses.CategoricalCrossentropy()
    target_one_hot = np.expand_dims(tf.keras.utils.to_categorical(target, 1000), axis=0)
    for step in range(steps):
        if step % 10 == 0: print(f"step:{step}")
        with tf.GradientTape() as tape:
            tape.watch(delta)
            adversary = preprocess_input(base_image + delta)
            prediction = model(adversary, training=False)
            step_loss = 0.3*cce_loss_func(target_one_hot, prediction) + 0.7*mse_loss_func(base_image, adversary)  ##

        gradients = tape.gradient(step_loss, delta)
        optimizer.apply_gradients([(epsilon*tf.sign(gradients), delta)])
    return delta
perbutation = reg_fgsm(model, base_image, delta, target, epsilon, steps)

example = preprocess_input(base_image + perbutation)
decode_predictions(model.predict(example), top=20)[0]

# plt.imsave('/Users/ruurbach/Downloads/zebra_pig_regu_diff.png', zaz)
