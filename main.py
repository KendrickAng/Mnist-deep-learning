import tensorflow as tf
import tensorflow.compat.v1.logging as log
import tensorflow_datasets as tfds
import numpy as np
import argparse
import sys

# initial config
log.set_verbosity(log.INFO)

# config - parse arguments
def load_config():
    parser = argparse.ArgumentParser("Hyperparameters for MNIST image recognition")
    parser.add_argument('--config_file', default='mnist.config', help='path to config file')
    parser.add_argument('--buffer_size', type=int, default=10000, help='how many images to include when shuffling the dataset')
    parser.add_argument('--batch_size', type=int, default=100, help='training images used per epoch')
    parser.add_argument('--input_size', type=int, default=784, help='dimension of flattened input tensor')
    parser.add_argument('--output_size', type=int, default=10, help='zero to nine')
    parser.add_argument('--hidden_layer_size', type=int, default=50, help='number of nodes per hidden layer')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of training steps')
    sys.stdout.flush() # prints any stuff in buffer to terminal (stdout)
    flags, unparsed = parser.parse_known_args() # FLAGS = namespace obj, access args like norm obj
    # read from config file and transfer to FLAGS
    import json
    with open(flags.config_file, 'r') as configs:
        config = json.load(configs)
        for key in config:
            setattr(flags, key, config[key]) 
    return flags

def main():
    FLAGS = load_config()
    log.info("Configuration: \n{}".format(FLAGS))

    # downloads the dataset to our local machine. ret tf.data.Dataset
    mnist_dataset, mnist_info = tfds.load(name='mnist', as_supervised=True, with_info=True)
    mnist_train, mnist_test = mnist_dataset["train"], mnist_dataset["test"]
    log.info("MNIST DATASET: \n{}".format(mnist_dataset))
    log.info("MNIST INFO: \n{}".format(mnist_info))

    # split out validation dataset
    num_validation_samples = 0.1 * mnist_info.splits["train"].num_examples
    num_validation_samples = tf.cast(num_validation_samples, tf.int64)

    num_test_samples = mnist_info.splits['test'].num_examples
    num_test_samples = tf.cast(num_test_samples, tf.int64)

    # standardization for reusability
    # input: mnist image and label
    # output: value between 0 and 1
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    # datasets can be enumerated
    scaled_train_and_validation_data = mnist_train.map(scale)
    scaled_test_data = mnist_test.map(scale)

    BUFFER_SIZE = FLAGS.buffer_size # dataset too big to shuffle at once
    shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE) # training sucks if data's ordered

    # final dataset
    validation_dataset = shuffled_train_and_validation_data.take(num_validation_samples)
    train_dataset = shuffled_train_and_validation_data.skip(num_validation_samples)
    test_dataset = scaled_test_data

    log.info("TRAIN DATASET: \n{}".format(train_dataset))
    log.info("VALIDATION_DATASET: \n{}".format(validation_dataset))

    # batching
    BATCH_SIZE = FLAGS.batch_size
    train_dataset = train_dataset.batch(BATCH_SIZE) # batch to find avg loss and acc for efficiency
    validation_dataset = validation_dataset.batch(num_validation_samples) # tensorflow expects dataset in batch size, don't act need batch
    test_dataset = test_dataset.batch(num_test_samples) # no need batch too, only cuz tf says so

    # load first batch
    log.info("BATCHED TRAIN DATASET: \n{}".format(train_dataset))
    log.info("BATCHED VALIDATION DATASET: \n{}".format(train_dataset))

    validation_inputs, validation_targets = next(iter(validation_dataset)) # create iterable, then take next e (batch)

    # outline the model
    input_size = FLAGS.input_size # 28 * 28
    output_size = FLAGS.output_size # 0 to 9
    hidden_layer_size = FLAGS.hidden_layer_size # length of cells

    model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # flatten to vector. input_shape required.
                tf.keras.layers.Dense(hidden_layer_size, activation="relu"),  # applies dot pdt for input and weights and adds bias
                tf.keras.layers.Dense(hidden_layer_size, activation="relu"),
                tf.keras.layers.Dense(output_size, activation="softmax")
            ])

    # Learning optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001);
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.summary()

    # train model
    NUM_EPOCHS = FLAGS.num_epochs
    model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose=2)

    # Test the model
    test_loss, test_accuracy = model.evaluate(test_dataset)
    log.info("Test loss: {.2f}, Test accuracy: {.2f}".format(test_loss, test_accuracy * 100));

if __name__ == "__main__":
    main()
