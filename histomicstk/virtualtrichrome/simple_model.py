import tensorflow as tf
filters = 32
size = 4
initializer = tf.random_normal_initializer(0., 0.02)

mymodel = tf.keras.Sequential()
mymodel.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

mymodel.summary()