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

    # X 入力変数
    # y 目的変数
    # n_features　説明変数の次元
    # n_outputs　目的変数の次元
    # それぞれdata_setで返す
    X, y, n_features, n_outputs = data_set()

    # ハイパーパラメータの調整設定読み込み
    # バッチサイズ設定
    n_bs = 8
    # エポック数
    nb_epochs = 3000 
    # 収束判定ループ（エポック）回数
    nb_patience = 300
    # 収束判定用差分パラメータ
    val_min_delta = 1e-5
    
    # 試行する中間層数の範囲設定
    n_layer_range = ('n_layer',1,5)

    # 試行する中間層のユニット（ニューロン）数の範囲設定
    mid_units_range = ('mid_units',10,30,5)

    # 試行するドロップアウト率の範囲設定
    dropout_rate_range = ('dropout_rate',0.0,0.1)

    # 試行する活性化関数のリスト設定
    activation_list = ('activation',['relu','tanh','sigmoid'])

    # 試行する最適化アルゴリズムのリスト設定
    optimizer_list = ('optimizer',['adam'])
    # 収束判定設定。以下の条件を満たすエポックがpatience回続いたら打切り。
    # val_loss(観測上最小値) - min_delta  < val_loss
    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=val_min_delta, patience=nb_patience, verbose=1, mode='min')

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

        # 学習モデルの構築と学習の開始
        model = create_model(n_features, n_outputs, n_layer, activation, mid_units, dropout_rate, optimizer)
        history = model.fit(X, y, verbose=0, epochs=nb_epochs, validation_split=0.1, batch_size=n_bs, callbacks=[es_cb])

        # 最小値探索(各エポックで得られた目的関数のうち最小値を返す)
        vmae = np.amin(history.history['val_mean_absolute_error'])

        # これまでの最小目的関数より小さい場合更新して、最適モデルとして保存
        if vmae < min_vmae:
            min_vmae = vmae
            model.save(savefile)

            # 損失関数の時系列変化をグラフ表示
            plot_loss(history)

        return vmae
    
    return objective



def create_model(n_features, n_outputs, n_layer, activation, mid_units, dropout_rate, optimizer):
    # ニューラルネットワーク定義
    inputs = Input(shape=(n_features,))
    x = BatchNormalization()(inputs)
    # 中間層数、各ニューロン数、ドロップアウト率の定義
    for i in range(n_layer):
        x = Dense(mid_units, activation=activation, kernel_initializer="he_normal")(x)
        x = Dropout(rate=dropout_rate)(x)

    # 出力層を定義（ニューロン数は1個）
    outputs = Dense(n_outputs, activation='linear')(x)
    # 回帰学習モデル作成
    model = Model(inputs, outputs) # nn.model
    model.compile("adam", "mse") # nn.model.compile
    # モデルを返す
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
