import sklearn.datasets
import matplotlib.pyplot as plt
from keras.models import Sequential #モデルの型
from keras.layers import Dense, Activation #線形変換、活性化関数

if __name__ == "__main__":
    #データ生成 x:2次元のデータ(入力) label:1次元のデータ(出力)
    x, label = sklearn.datasets.make_classification(n_features=2, n_samples=300, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=3)
    x_test, label_test = sklearn.datasets.make_classification(n_features=2, n_samples=10, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=3)

    NN=Sequential()
    NN.add(Dense(units = 3, activation = 'tanh', input_dim=2))
    NN.add(Dense(units = 1, activation = 'softmax'))
    NN.summary()
    NN.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"])

    #訓練
    NN.fit(x, label, epochs=1000)
    
    #テスト
    print(NN.evaluate(x_test, label_test))
