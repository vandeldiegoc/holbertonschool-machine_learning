import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the inception network"""
    X = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(filters=64, kernel_size=7, strides=2,
                            padding="same", activation="relu")(X)
    max1 = K.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(conv1)
    conv2 = K.layers.Conv2D(filters=64, kernel_size=1,
                            padding="same", activation="relu")(max1)
    conv3 = K.layers.Conv2D(filters=192, kernel_size=3,
                            padding="same", activation="relu")(conv2)
    max2 = K.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(conv3)
    icp3a = inception_block(max2, [64, 96, 128, 16, 32, 32])
    icp3b = inception_block(icp3a, [128, 128, 192, 32, 96, 64])
    max3 = K.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(icp3b)
    icp4a = inception_block(max3, [192, 96, 208, 16, 48, 64])
    icp4b = inception_block(icp4a, [160, 112, 224, 24, 64, 64])
    icp4c = inception_block(icp4b, [128, 128, 256, 24, 64, 64])
    icp4d = inception_block(icp4c, [112, 144, 288, 32, 64, 64])
    icp4e = inception_block(icp4d, [256, 160, 320, 32, 128, 128])
    max4 = K.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(icp4e)
    icp5a = inception_block(max4, [256, 160, 320, 32, 128, 128])
    icp5b = inception_block(icp5a, [384, 192, 384, 48, 128, 128])
    avg = K.layers.AveragePooling2D(pool_size=7, padding="same")(icp5b)
    drop = K.layers.Dropout(0.4)(avg)
    output = K.layers.Dense(1000)(drop)
    model = K.models.Model(inputs=X, outputs=output)
    return(model)
