import numpy as np
from absl import app
from absl import flags
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import torchvision.transforms as transforms
from torch.utils.data import Subset
from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data

DATA_DIR = '/home/la_belva/Downloads/apt_dataset/samples'
TEST_DIR = '/home/la_belva/Downloads/apt_dataset/test'
NUM_CLASSES = 12
WIDTH = 64
HEIGHT = 64
CHANNELS = 3
SHADOW_DATASET_SIZE = 4000
ATTACK_TEST_DATASET_SIZE = 4000
BATCH_SIZE = 32
target_epochs = 1
num_shadows = 1

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "target_epochs", 12, "Number of epochs to train target and shadow models."
)
flags.DEFINE_integer("attack_epochs", 12, "Number of epochs to train attack models.")
flags.DEFINE_integer("num_shadows", 3, "Number of epochs to train attack models.")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


def get_data():
    datagen_kwargs = dict(rescale=1. / 255)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
     train_datagen.horizontal_flip = True
     train_datagen.vertical_flip = True
     train_datagen.width_shift_range = 0.2
     train_datagen.height_shift_range = 0.2
     train_datagen.zoom_range = [0.8, 1.2]
     train_datagen.shear_range = 10
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        shuffle=True,
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        shuffle=False,
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
    )

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    batch_index_train = 0
    batch_index_test = 0

    while batch_index_train <= train_generator.batch_index:
        image, labels = train_generator.next()
        if image.shape[0] != BATCH_SIZE:
            pass
        else:
            X_train.append(image)
            y_train.append(labels)
        batch_index_train = batch_index_train + 1

    while batch_index_test <= test_generator.batch_index:
        image, labels = test_generator.next()
        if image.shape[0] != BATCH_SIZE:
            pass
        else:
            X_test.append(image)
            y_test.append(labels)
        batch_index_test = batch_index_test + 1

    # now, data_array is the numeric data of whole images
    X_train = tf.stack(X_train)
    X_train = tf.reshape(X_train, (-1, HEIGHT, WIDTH, CHANNELS))
    y_train = tf.stack(y_train)
    y_train = tf.reshape(y_train, (-1, NUM_CLASSES))
    train_data_array = np.asarray(X_train)
    train_label_array = np.asarray(y_train)

    X_test = tf.stack(X_test)
    X_test = tf.reshape(X_test, (-1, HEIGHT, WIDTH, CHANNELS))
    y_test = tf.stack(y_test)
    y_test = tf.reshape(y_test, (-1, NUM_CLASSES))
    test_data_array = np.asarray(X_test)
    test_label_array = np.asarray(y_test)

    return (train_data_array, train_label_array), (test_data_array, test_label_array)


def target_model_fn():
    """The architecture of the target (victim) model.

    The attack is white-box, hence the attacker is assumed to know this architecture too."""

    model = tf.keras.models.Sequential()

    model.add(
        # layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(WIDTH, HEIGHT, CHANNELS),
        )
    )
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    model.add(layers.Dense(1024, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def attack_model_fn():
    """Attack model that takes target model predictions and predicts membership.

    Following the original paper, this attack model is specific to the class of the input.
    AttachModelBundle creates multiple instances of this model for each class.
    """
    model = tf.keras.models.Sequential()

    model.add(layers.Dense(1024, activation="relu", input_shape=(NUM_CLASSES,)))

    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def demo(argv):
    del argv  # Unused.

    (X_train, y_train), (X_test, y_test) = get_data()

    # Train the target model.
    print("Training the target model...")
    target_model = target_model_fn()
    target_model.fit(
        X_train, y_train, epochs=FLAGS.target_epochs, validation_split=0.1, verbose=True
    )

    # Train the shadow models.
    smb = ShadowModelBundle(
        target_model_fn,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=FLAGS.num_shadows,
    )

    # We assume that attacker's data were not seen in target's training.
    attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
        X_test, y_test, test_size=0.1
    )
    print(attacker_X_train.shape, attacker_X_test.shape)

    print("Training the shadow models...")
    X_shadow, y_shadow = smb.fit_transform(
        attacker_X_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=FLAGS.target_epochs,
            verbose=True,
            validation_data=(attacker_X_test, attacker_y_test),
        ),
    )

    # ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
    amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

    # Fit the attack models.
    print("Training the attack models...")
    amb.fit(
        X_shadow, y_shadow, fit_kwargs=dict(epochs=FLAGS.attack_epochs, verbose=True)
    )

    # Test the success of the attack.

    # Prepare examples that were in the training, and out of the training.
    data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]

    # Compile them into the expected format for the AttackModelBundle.
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, data_in, data_out
    )

    # Compute the attack accuracy.
    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

    print(attack_accuracy)


if __name__ == "__main__":
    app.run(demo)
