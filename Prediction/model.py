import tensorflow as tf
from tensorflow import keras


def full_input_loader(sample, img_size=(128, 128)):

    ####################################
    # load image
    ####################################
    img = tf.io.read_file(sample['IMG'])
    if tf.strings.split(sample['IMG'])[-1] == b"jpg":
        img = tf.image.decode_jpeg(img, 3)
    else:
        img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, img_size)

    ####################################
    # wash tag
    ####################################
    wash = sample["WASH"]
    wash = tf.subtract(wash, 1)

    return {"IMG": img, "WASH": wash}


def img_only_input_loader(sample, img_size=(128, 128)):
    ####################################
    # load image
    ####################################
    img = tf.io.read_file(sample['IMG'])
    if tf.strings.split(sample['IMG'])[-1] == b"jpg":
        img = tf.image.decode_jpeg(img, 3)
    else:
        img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, img_size)

    return {"IMG": img}


def create_full_model(is_train=True):

    # image input
    img = keras.Input(shape=(128, 128, 3), name="IMG")
    preprocessed_img = keras.applications.resnet_v2.preprocess_input(img)
    resnet = keras.applications.ResNet50V2(include_top=False, weights="imagenet", pooling="avg")
    resnet.trainable = True
    img_embedding = resnet(preprocessed_img)

    # wash tag
    wash = keras.Input(shape=(1), name="WASH")

    # input construction
    concatenated = keras.layers.concatenate([img_embedding, wash])

    # output head layer
    out = keras.layers.Dense(32, kernel_initializer="normal", activation="relu")(concatenated)
    out = keras.layers.Dense(256, kernel_initializer="normal", activation="relu")(out)
    out = keras.layers.Dense(512, kernel_initializer="normal", activation="relu")(out)
    out = keras.layers.Dense(512, kernel_initializer="normal", activation="relu")(out)
    out = keras.layers.Dense(512, kernel_initializer="normal", activation="relu")(out)
    out = keras.layers.Dense(64, kernel_initializer="normal", activation="relu")(out)
    out = keras.layers.Dense(1, kernel_initializer="normal", activation="relu")(out)

    return keras.Model([img, wash], out)


def create_img_only_model(is_train=True):

    # image input
    img = keras.Input(shape=(128, 128, 3), name="IMG")
    preprocessed_img = keras.applications.resnet_v2.preprocess_input(img)
    resnet = keras.applications.ResNet50V2(include_top=False, weights="imagenet", pooling="avg")
    resnet.trainable = True
    img_embedding = resnet(preprocessed_img)

    # wash tag
    wash = keras.Input(shape=(1), name="WASH")

    # input construction
    concatenated = keras.layers.concatenate([img_embedding, wash])

    # output head layer
    out = keras.layers.Dense(32, kernel_initializer="normal", activation="relu")(concatenated)
    out = keras.layers.Dense(256, kernel_initializer="normal", activation="relu")(out)
    out = keras.layers.Dense(512, kernel_initializer="normal", activation="relu")(out)
    out = keras.layers.Dense(512, kernel_initializer="normal", activation="relu")(out)
    out = keras.layers.Dense(512, kernel_initializer="normal", activation="relu")(out)
    out = keras.layers.Dense(64, kernel_initializer="normal", activation="relu")(out)
    out = keras.layers.Dense(1, kernel_initializer="normal", activation="relu")(out)

    return keras.Model([img, wash], out)


def create_img_only_dataset(dataframe, input_loader, batch_size=256, is_train=True):

    columns = ["IMG", "DAY"]
    df = dataframe[columns].copy()
    label = df.pop("DAY")
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), label))
    dataset = dataset.shuffle(buffer_size=len(dataframe))

    if is_train:
        dataset = dataset.shuffle(len(dataframe))

    dataset = dataset.map(lambda x, y: (input_loader(x), y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

def create_full_dataset(dataframe, input_loader, batch_size=256, is_train=True):

    columns = ["IMG", "DAY", "WASH"]
    df = dataframe[columns].copy()
    label = df.pop("DAY")
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), label))
    dataset = dataset.shuffle(buffer_size=len(dataframe))

    if is_train:
        dataset = dataset.shuffle(len(dataframe))

    dataset = dataset.map(lambda x, y: (input_loader(x), y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset