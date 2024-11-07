import tensorflow as tf
import numpy as np
import os
import time

from fourier_model import FourierModel
from prepare_data import prepare_all_data

# Hyperparameters
learning_rate = 0.001
batch_size = 32

# Learning Rate, Optimizer, Loss Function, Training Metrics
optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

train_loss_metric = tf.keras.metrics.SparseCategoricalCrossentropy()
val_loss_metric = tf.keras.metrics.SparseCategoricalCrossentropy()


# Model
model = FourierModel(batch_size)


@tf.function
def compute_loss(labels, predictions):
      per_example_loss = loss_fn(labels, predictions)
      return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        # loss_value = loss_fn(y, logits)
        loss_value = compute_loss(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    train_loss_metric.update_state(y, logits)
    return loss_value

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)
    val_loss_metric.update_state(y, val_logits)


def fft_images(x_images):
    # Shapes
    [n_images, height, width, channels] = x_images.shape

    # Create placeholder arrays
    x_image_fft_real = np.zeros_like(x_images, dtype=np.float32)
    x_image_fft_imag = np.zeros_like(x_images, dtype=np.float32)

    for i in range(n_images):
        image = x_images[i, :, :, :]
        real_channel_temp = np.zeros_like(image, dtype=np.float32)
        imag_channel_temp = np.zeros_like(image, dtype=np.float32)
        for j in range(channels):
            fft2d = np.fft.fft2(image[:, :, j])
            real_channel_temp[:, :, j] = fft2d.real
            imag_channel_temp[:, :, j] = fft2d.imag
        x_image_fft_real[i, :, :, :] = real_channel_temp
        x_image_fft_imag[i, :, :, :] = imag_channel_temp
        if ((i+1) % 100 == 0):
            print(i+1, " FFTs of ", n_images, " done.")

    return x_image_fft_real, x_image_fft_imag

def benchmark_fcnn(random_seed):

    # Dataset
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_all_data(random_seed)

    # Shapes
    n_train = x_train.shape[0]
    n_valid = x_valid.shape[0]
    n_test = x_test.shape[0]

    x_train_real, x_train_imag = fft_images(x_train)
    x_valid_real, x_valid_imag = fft_images(x_valid)
    x_test_real, x_test_imag = fft_images(x_test)

    # Prepare the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_real, x_train_imag, y_train))
    train_dataset = train_dataset.shuffle(n_train).batch(batch_size=batch_size)

    # Prepare the testing dataset
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid_real, x_valid_imag, y_valid))
    valid_dataset = valid_dataset.shuffle(n_valid).batch(batch_size=batch_size)

    # Prepare the testing dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_real, x_test_imag, y_test))
    test_dataset = test_dataset.shuffle(n_test).batch(batch_size=batch_size)

    # Lists to store training results
    train_acc_logger = []
    val_acc_logger = []
    train_loss_logger = []
    val_loss_logger = []

    # To modify learning rate
    val_loss_patience = 3
    val_loss_wait = 0
    best_val_loss = float('inf')


    epochs = 20
    for epoch in range(epochs):
        # if epoch > 5:
        #   optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, momentum=0.9)
        #   # optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)

        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset
        for step, (x_batch_train_real, x_batch_train_imag, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step((x_batch_train_real, x_batch_train_imag), y_batch_train)

            # Log every 500 batches
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch
        train_acc = train_acc_metric.result()
        train_acc_logger.append(train_acc)
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        train_loss = train_loss_metric.result()
        train_loss_logger.append(train_loss)
        print("Training loss over epoch: %.4f" % (float(train_loss),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()
        train_loss_metric.reset_states()

        # Run validation loop at the end of each epoch
        for x_batch_val_real, x_batch_val_imag, y_batch_val in valid_dataset:
            test_step((x_batch_val_real, x_batch_val_imag), y_batch_val)

        val_acc = val_acc_metric.result()
        val_acc_logger.append(val_acc)

        val_loss = val_loss_metric.result()
        val_loss_logger.append(val_loss)

        val_acc_metric.reset_states()
        val_loss_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Validation loss: %.4f" % (float(val_loss),))
        print("Time taken: %.2f" % (time.time() - start_time))

        val_loss_wait += 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_loss_wait = 0
        if ((val_loss_wait >= val_loss_patience) and (epoch<epochs-2)):
            learning_rate = learning_rate * 0.1
            print("Learning rate decreased to: ", learning_rate)
            optimizer.learning_rate = learning_rate
            best_val_loss = val_loss
            val_loss_wait = 0

    # Test Model
    test_logits_list = []
    y_actual_list = []
    for x_batch_test_real, x_batch_test_imag, y_batch_test in test_dataset:
        test_logits = model((x_batch_test_real, x_batch_test_imag), training=False)
        y_actual_list.append(y_batch_test)
        test_logits_list.append(test_logits)

    y_pred = []
    for index in range(len(test_logits_list)):
        y_pred.append(tf.argmax(test_logits_list[index], axis=1))

    y_pred = tf.concat(y_pred, axis=0)
    y_actual = tf.concat(y_actual_list, axis=0)

    save_folder = f"./model_results/fcnn/{random_seed}"
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)


    # Save results from training
    np.save(save_folder + '/' + str(random_seed) + '_val_loss_logger.npy' , np.array(val_loss_logger))
    np.save(save_folder + '/' + str(random_seed) + '_train_loss_logger.npy' , np.array(train_loss_logger))
    np.save(save_folder + '/' + str(random_seed) + '_val_acc_logger.npy' , np.array(val_acc_logger))
    np.save(save_folder + '/' + str(random_seed) + '_train_acc_logger.npy' , np.array(train_acc_logger))


    # Save results from testing
    y_pred_numpy = y_pred.numpy()
    y_actual_numpy = y_actual.numpy()
    file_name_pred = save_folder + '/' + str(random_seed) + "_y_pred.npy"
    file_name_actual = save_folder + '/ '+ str(random_seed) + "_y_actual.npy"
    np.save(file_name_pred, y_pred_numpy)
    np.save(file_name_actual, y_actual_numpy)


if __name__ == "__main__":

    if not os.path.isdir("./model_results"):
        os.mkdir("./model_results")
    if not os.path.isdir("./model_resutls/fcnn"):
        os.mkdir("./model_results/fcnn")

    list_of_seeds = [
        1327, 5208, 8208, 3515, 2710,
        4212, 1803, 3226, 2712, 7103,
        4310, 2808, 5916, 7600, 1522
    ]

    # Change the value of index to benchmark for a different seed
    index = 0
    benchmark_fcnn(list_of_seeds[index])
        

