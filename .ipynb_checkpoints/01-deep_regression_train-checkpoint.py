# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats
import optuna
import configparser
import pandas as pd
import logging.config
import sys
import os
import time

# 設定ファイル読み込み
conf_file = 'dnn.conf'
config = configparser.ConfigParser()
config.read(conf_file, 'UTF-8')

# ログ設定ファイル読み込み
logging.config.fileConfig('logging.conf')
logger = logging.getLogger()

# 目的関数最小値の初期値定義
min_vmae = np.inf


def main():
    # 環境設定(ディスプレイの出力先をlocalhostにする)
    os.environ['DISPLAY'] = ':0'

    # コマンド引数確認
    if len(sys.argv) < 2:
        print('使用法: python deep_regression_train.py 保存ファイル名.h5')
        sys.exit()
    
    # 探索試行回数を設定
    n_trials = config.getint('Trials','trials')

    # 最適化探索（optunaのstudyオブジェクト定義）
    study = optuna.create_study(sampler=optuna.samplers.TPESampler())
    # optimizeに最適化すべき目的関数（objective）を渡す。これをn_trials回試行する。目的関数の値が最小のものを探索する。
    study.optimize(outer_objective(), n_trials)
    # ここではouter_objective()を実行してobjective()を実行している
    
    # 最適だった試行回を表示
    logger.info('best_trial.number: ' + 'trial#' + str(study.best_trial.number))
    # 目的関数の最適（最小）値を表示
    logger.info('best_vmae: ' + str(study.best_value))

    # ハイパーパラメータをソートして表示
    logger.info('--- best hyperparameter ---')
    sorted_best_params = sorted(study.best_params.items(), key=lambda x : x[0])
    for i, k in sorted_best_params:
        logger.info(i + ' : ' + str(k))
    logger.info('------------')


# objective関数を内包する高階関数。objective関数呼び出し前に、種々の事前設定等を行う。
def outer_objective():
    # 学習モデルファイル保存先パス取得
    savefile = sys.argv[1]

    # データセットファイル取得
    c_file_path = config['File Path']
    dataset_file = c_file_path['dataset_file']
    # データをロード
    X, y, n_features, n_outputs = data_set(dataset_file)
    #print(X)
    #print(y)

    # ハイパーパラメータの調整設定読み込み
    n_bs, nb_epochs, nb_patience, val_min_delta, n_layer_range, mid_units_range, dropout_rate_range, activation_list, optimizer_list = set_hyperparameter()
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

        # 各パラメータを画面出力
        logger.info("trial#" + str(trial.number) + ': ' +
        "n_layer= " + str(n_layer) + ', ' +
        "mid_units= " + str(mid_units) + ', ' +
        "dropout_rate= " + str(dropout_rate) + ', ' +
        "activation= " + str(activation) + ', ' +
        "optimizer= " + str(optimizer))

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


def data_set(dataset_file):
    # csvをロードし、変数に格納
    df = pd.read_csv(dataset_file)
    dfv = df.values.astype(np.float64)
    n_dfv = dfv.shape[1]

    # 学習データをシャッフル
    np.random.shuffle(dfv)

    # 特徴量のセットを変数Xに、ターゲットを変数yに格納
    X = dfv[:, np.array(range(0, (n_dfv-1)))]
    y = dfv[:, np.array([(n_dfv-1)])]

    # データの標準化
    X = scipy.stats.zscore(X)
    y = scipy.stats.zscore(y)

    # サンプル数、特徴量の次元、出力数の取り出し
    (n_samples, n_features) = X.shape
    n_outputs = y.shape[1]
    
    return X, y, n_features, n_outputs


def set_hyperparameter():
    # トレーニングパラメータセクション
    c_trpa = config['Train Parameter']
    # バッチサイズ設定
    n_bs = c_trpa.getint('batch_size')
    # エポック数
    nb_epochs = c_trpa.getint('epochs')
    # 収束判定ループ（エポック）回数
    nb_patience = c_trpa.getint('patience')
    # 収束判定用差分パラメータ
    val_min_delta = c_trpa.getfloat('min_delta')

    # 中間層数セクション
    c_layer = config['Layer']
    # 試行する中間層数の範囲設定
    n_layer_range = ('n_layer', c_layer.getint('layer_min'), c_layer.getint('layer_max'))

    # 中間層ユニット（ニューロン）数セクション
    c_mid_units = config['Mid Units']
    # 試行する中間層のユニット（ニューロン）数の範囲設定
    mid_units_range = ('mid_units', c_mid_units.getint('mid_units_min'),c_mid_units.getint('mid_units_max') , c_mid_units.getint('mid_units_step'))

    # ドロップアウトセクション
    c_dropout = config['Dropout']
    # 試行するドロップアウト率の範囲設定
    dropout_rate_range = ('dropout_rate', c_dropout.getfloat('dropout_rate_min'), c_dropout.getfloat('dropout_rate_max'))

    # 試行する活性化関数のリスト設定
    activation_list = ('activation', config['Activation']['activation_list'].split())

    # 試行する最適化アルゴリズムのリスト設定
    optimizer_list = ('optimizer', config['Optimizer']['optimizer_list'].split())

    #print(mid_units_range)
    #print(optimizer_list)

    return n_bs, nb_epochs, nb_patience, val_min_delta, n_layer_range, mid_units_range, dropout_rate_range, activation_list, optimizer_list


def create_model(n_features, n_outputs, n_layer, activation, mid_units, dropout_rate, optimizer):
    # ニューラルネットワーク定義
    model = Sequential()

    # 中間層数、各ニューロン数、ドロップアウト率の定義
    for i in range(n_layer):
        model.add(Dense(mid_units, activation=activation, input_shape=(n_features,)))
        model.add(Dropout(dropout_rate))

    # 出力層を定義（ニューロン数は1個）
    model.add(Dense(units=n_outputs, activation='linear'))
    # 回帰学習モデル作成
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
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


if __name__ == '__main__':
    main()
