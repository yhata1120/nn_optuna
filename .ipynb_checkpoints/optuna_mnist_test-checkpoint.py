#!/usr/bin/env python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# load MNIST data
(X_trainval, y_trainval), (X_test, y_test) = mnist.load_data()
X_trainval = X_trainval.reshape(X_trainval.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_trainval = X_trainval.astype('float32')/255.
X_test = X_test.astype('float32')/255.
y_trainval = to_categorical(y_trainval)
y_test = to_categorical(y_test)

import psycopg2
import optuna
import nn

def objective(trial):
    nn_dict = {
        "n_input": X_trainval.shape[-1],
        "n_output": y_trainval.shape[-1],
        "n_layer": trial.suggest_int("n_layer", 1, 10),
        "n_node": 2**trial.suggest_int("log2_n_node", 3, 8),
        "activation_hidden": 'relu',
        "activation_output": 'softmax',
    }
    model_config = nn.make_simple_nn(
        **nn_dict,
        get_config=True
    )
    score = nn.kfold_cv(
        model_config,
        X_trainval,
        y_trainval,
        optimizer="adam",
        loss='categorical_crossentropy',
        n_splits=5,
        batch_size=2**trial.suggest_int("log2_batch_size", 5, 10),
        epochs=10000,
        es_patience=10,
        save_dir="mnist_ffnn_optuna",
        prefix=f"trial_{trial.number}",
        eval_func=None,
        random_state=0
    )
    return score

#storage = optuna.storages.RDBStorage(
#    url='postgresql://postgres@192.168.100.251',
#    engine_kwargs={
#    'pool_size': 20,
#    'max_overflow': 0
#    }
#)
study = optuna.create_study(
    storage='postgresql://postgres@192.168.100.251',
    study_name="mnist_ffnn_optuna",
    direction="minimize",
    load_if_exists=True
)
study.optimize(objective, n_trials=10)