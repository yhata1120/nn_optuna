# x ^2などを回帰してみる
# k-fold　cross　validation


import os
import numpy as np

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Reshape, Dropout, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# load data of gb (今回はx^2を使って実験を行う) 
def dataset():
    sample_size = 10000
    noise_size = 100

    X_all = np.linspace(-100,100,sample_size)
    y_all = X_all**2 + noise_size*np.random.randn((len(X_all)))

    X_trainval, X_test, y_trainval, y_test = train_test_split(X_all,y_all,test_size=0.2,random_state=0)

    return X_trainval, X_test, y_trainval, y_test

X_trainval, X_test, y_trainval, y_test= dataset()

X_trainval = X_trainval.reshape(X_trainval.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
y_trainval = y_trainval.reshape(y_trainval.shape[0], -1)
y_test = y_test.reshape(y_test.shape[0], -1)

# 検証用のデータと訓練用のデータを準備
n_feature = len(X_trainval.T)
n_outputs = len(y_trainval.T)



min_vmae = np.inf

def main():
    
    # 探索試行回数を設定
    n_trials = 10
    # 最適化探索（optunaのstudyオブジェクト定義）
    study = optuna.create_study(sampler=optuna.samplers.TPESampler())
    # optimizeに最適化すべき目的関数（objective）を渡す。これをn_trials回試行する。目的関数の値が最小のものを探索する。
    study.optimize(outer_objective(), n_trials)
    # ここではouter_objective()を実行してobjective()を実行している
    
# objective関数を内包する高階関数。objective関数呼び出し前に、種々の事前設定等を行う。
def outer_objective():
    
    savefile = 'best_model'
    
    # ハイパーパラメータの調整設定読み込み
    # バッチサイズ設定
    n_bs = 2**trial.suggest_int("log2_batch_size", 5, 10)
    # エポック数
    nb_epochs = 10000 
    # 収束判定ループ（エポック）回数
    nb_patience = 500
    # 収束判定用差分パラメータ
    val_min_delta = 1e-5
    
    # 試行する中間層数の範囲設定
    n_layer_range = ('n_layer',1,5)

    # 試行する中間層のユニット（ニューロン）数の範囲設定
    mid_units_range = 2**trial.suggest_int("log2_n_node", 3, 8)

    # 試行するドロップアウト率の範囲設定
    dropout_rate_range = ('dropout_rate',0.0,0.5)

    # 試行する活性化関数のリスト設定
    activation_list = ('activation',['relu','tanh','sigmoid'])

    # 試行する最適化アルゴリズムのリスト設定
    optimizer_list = ('optimizer',['adam','sgd'])

    # 試行する学習率の範囲設定(adam)
    lr_adam_range = ("lr_adam", 1e-4, 1e-1)
    
    # 試行する学習率の範囲設定(sgd)
    lr_sgd_range = 0.01

    # 収束判定設定。以下の条件を満たすエポックがpatience回続いたら打切り。
    # val_loss(観測上最小値) - min_delta  < val_loss
    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=val_min_delta, patience=nb_patience, mode='min', restore_best_weights=True)

    print('obj_loop_start')

    # 目的関数
    def objective(trial):
        # グローバル変数の目的関数最小値呼び出し
        global min_vmae

        # 中間層数の探索範囲設定
        n_layer = trial.suggest_int(*n_layer_range)
        # ユニット数の探索範囲設定
        mid_units = int(trial.suggest_discrete_uniform(*mid_units_range))
        # ドロップアウト率の探索範囲設定
        dropout_rate = trial.suggest_uniform(*dropout_rate_range)
        # 活性化関数の探索候補設定
        activation = trial.suggest_categorical(*activation_list)
        # 最適化アルゴリズムの探索候補設定
        optimizer = trial.suggest_categorical(*optimizer_list)
        # 最適化アルゴリズムの学習率の探索候補決定(adam)
        lr_adam = trial.suggest_loguniform(*lr_range)
        # 最適化アルゴリズムの学習率の探索候補決定(sgd)
        lr_sgd = lr_sgd_range
        
        # 最適化アルゴリズム＋学習率の探索候補決定
        lr_dict = {"adam":lr_adam, "sgd":lr_sgd}
        lr=lr_dict.get(optimizer)
        optimizer = optimizer(lr = lr)
        
        
        # 学習モデルの構築と学習の開始
        # create_modelに入れるものはインプットとアウトプットの"次元"
        model_config = create_model(n_features, n_outputs, n_layer, activation, mid_units, dropout_rate, optimizer,get_config = True)
        
        score = kfold_cv(
        model_config,
        X_trainval,
        y_trainval,
        optimizer=optimizer,
        loss='mse',
        n_splits=5,
        batch_size=n_bs,
        epochs = nb_epochs,
        save_dir="mnist_ffnn_optuna",
        prefix=f"trial_{trial.number}",
        es_cb = es_cb,
        eval_func=None,
        random_state=0
        )
    # k分割交差検証でスコアを出す
        return score
    
    return objective



def create_model(n_features, n_outputs, n_layer, activation, mid_units, dropout_rate, weights=None, get_config=False):
    # ニューラルネットワーク定義
    inputs = Input(shape=(n_features,))
    x = BatchNormalization()(inputs)
    # 中間層数、各ニューロン数、ドロップアウト率の定義
    kernel_initializer_dict = {"relu": "he_normal"}
    hidden_layers_list = [Dense(n_node, activation=activation_hidden, kernel_initializer=kernel_initializer_dict.get(activation_hidden, "glorot_normal"), name=f"hidden_{i + 1}") for i in range(n_layer)] #"activation = relu以外の場合はglorot_normalを使う"
    x = inputs
    for layer in hidden_layers_list:
        x = layer(x)
        x = Dropout(rate=dropout_rate)(x)
    # 出力層を定義（ニューロン数は1個）
    outputs = Dense(n_outputs, activation='linear')(x)
    # 回帰学習モデル作成
    model = Model(inputs, outputs) # nn.model
    # モデルを返す　
    # weightか、configの形か、そのままモデルを帰すのか、今回はconfigを通してfileを返す
    if weights is not None:
        model.load_weights(weights)
    if get_config:
        return model.get_config()
    else:
        return model

def plot_loss(history):
    # 損失関数のグラフの軸ラベルを設定
    plt.xlabel('time step')
    plt.ylabel('loss')

    # グラフ縦軸の範囲を0以上と定める
    plt.ylim(0, max(np.r_[history.history['val_loss'], history.history['loss']]))

    # 損失関数の時間変化を描画
    val_loss, = plt.plot(history.history['val_loss'], c='#56B4E9')
    loss, = plt.plot(history.history['loss'], c='#E69F00')

    # グラフの凡例（はんれい）を追加
    plt.legend([loss, val_loss], ['loss', 'val_loss'])

    # 描画したグラフを表示
    #plt.show()

    # グラフを保存
    plt.savefig('dnn_reg_train_figure.png')
    
def kfold_cv(model_config, X_trainval, y_trainval, optimizer=optimizer, loss="mse", n_splits=5, batch_size=n_bs, epochs=nb_epochs, , save_dir=".", prefix="kfold_cv", es_cb = es_cb, eval_func=None, random_state=0):
    val_scores = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # 自分の場合はdata_trainvalはndarrayではなくdataframeのまま入れる
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    for fold, (train_indices, val_indices) in enumerate(kf.split(X_trainval)):
        # Prepare dataset
        X_train, X_val = X_trainval[train_indices], X_trainval[val_indices]
        y_train, y_val = y_trainval[train_indices], y_trainval[val_indices]
        # Create model from model_config
        if isinstance(model_config, dict):
            model = Model.from_config(model_config)
            model.compile(optimizer=optimizer, loss=loss)
        elif isinstance(model_config, str):
            if os.path.isfile(model_config):
                with open(model_config, "rt") as f:
                    json_string = f.read()
            else:
                json_string = model_config
            model = model_from_json(json_string)
            model.compile(optimizer=optimizer, loss=loss)
        elif callable(model_config):
            model = model_config()
        else:
            raise RuntimeError(f"unknown type of model_config: {type(model_config)}")
        model._name = f"{prefix}_{model.name}"
        # Save model architecture
        if fold == 0:
            with open(os.path.join(save_dir, f"{prefix}_architecture.json"), 'wt') as f:
                f.write(model.to_json())

        # Train model
        # validation_dataというoptionを使っている
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es_cb],
            verbose=0
        )
        # Save learning history
        with open(os.path.join(save_dir, f'{prefix}_cv{fold}_history.pickle'), 'wb') as f:
            pickle.dump(history.history, f)
        # Evaluate model by validation data
        # もし、何かしらの評価関数(eval_func)がなければmodel.evaluateで追加、
        # 何かしらの評価関数(eval_func)が存在すればそちらで誤差を評価
        if eval_func is None:
            score = model.evaluate(X_val, y_val)
        else:
            y_val_pred = model.predict(X_val)
            score = eval_func(y_val, y_val_pred)
        print(f'fold {fold} score: {score}')
        # append each fold scores
        val_scores.append(score)
        # Delete model and clear session # メモリ問題
        del model
        K.clear_session()
    # Get average of validation score
    # optunaによる実験一回分のスコア
    cv_score = np.mean(val_scores)
    print(f'CV score: {cv_score}')
    return cv_score
