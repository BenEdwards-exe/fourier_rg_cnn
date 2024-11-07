import tensorflow as tf
import numpy as np
import os

from rg_cnn_models import MCRGNet_regularised, FR_Deep_regularised, Toothless_regularised, RG_ZOO_regularised
from prepare_data import prepare_all_data


def benchmark_model(model_name, model, random_seed):
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    # Dataset
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_all_data(random_seed)
    
    # Compile Model
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    model.summary()

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0)
    history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=20, validation_data=(x_valid, y_valid), callbacks=[reduce_lr])

    # --------- Test model
    eval_result = model.evaluate(x_test, y_test)
    y_pred = model.predict(x=x_test)
    y_pred = np.argmax(y_pred, axis=1)


    # -------------------- Save results locally
    save_folder = f"./model_results/{model_name}/{random_seed}"
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    # Save results from training
    for key in history.history:
        np.save(f"{save_folder}/{random_seed}_{key}_training_log.npy", np.array(history.history[key]))

    # Save results from testing
    np.save(f"{save_folder}/{random_seed}_y_pred.npy", y_pred)
    np.save(f"{save_folder}/{random_seed}_y_actual.npy", y_test)
    np.save(f"{save_folder}/{random_seed}_eval.npy", np.array(eval_result))




if __name__ == "__main__":

    if not os.path.isdir("./model_results"):
        os.mkdir("./model_results")

    model_names = ['mcrgnet', 'toothless', 'fr_deep', 'rg_zoo']
    for model_name in model_names:
        if not os.path.isdir(f"./model_results/{model_name}"):
            os.mkdir(f"./model_results/{model_name}")


    list_of_seeds = [
        1327, 5208, 8208, 3515, 2710,
        4212, 1803, 3226, 2712, 7103,
        4310, 2808, 5916, 7600, 1522
    ]

    # MCRGNet Benchmark
    for random_seed in list_of_seeds:
        model = MCRGNet_regularised(input_shape=(150,150,1),n_classes=3, random_seed=random_seed)
        model_name = model_names[0]
        benchmark_model(model_name, model, random_seed)

    # Toothless Benchmark
    for random_seed in list_of_seeds:
        model = Toothless_regularised(input_shape=(150,150,1),n_classes=3, random_seed=random_seed)
        model_name = model_names[1]
        benchmark_model(model_name, model, random_seed)

    # FR-Deep Benchmark
    for random_seed in list_of_seeds:
        model = FR_Deep_regularised(input_shape=(150,150,1),n_classes=3, random_seed=random_seed)
        model_name = model_names[2]
        benchmark_model(model_name, model, random_seed)
    
    # RG-Zoo Benchmark
    for random_seed in list_of_seeds:
        model = RG_ZOO_regularised(input_shape=(150,150,1),n_classes=3, random_seed=random_seed)
        model_name = model_names[3]
        benchmark_model(model_name, model, random_seed)