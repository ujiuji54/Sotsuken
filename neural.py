import numpy as np

class Function(object):
    def tanh(self, x):#隠れ層の活性化関数
        return np.tanh(x)

    def dtanh(self, x):
        return 1. - x * x

    def softmax(self, x):#出力層の活性化関数
        e = np.exp(x - np.max(x)) #オーバーフローを防ぐ
        if e.ndim == 1:
            return e / np.sum(e, axis = 0)
        else:
            return e / np.array([np.sum(e, axis = 1)]).T #サンプル数が1の時

class Newral_Network(object):
    def __init__(self, unit):
        print ("Number of layer = ",len(unit))
        print (unit)
        print ("_________________________")
        self.F = Function()
        self.unit = unit
        self.W = []
        self.B = []
        self.dW = []
        for i in range(len(self.unit) - 1):
            w = np.random.rand(self.unit[i], self.unit[i + 1])
            self.W.append(w)
            dw = np.random.rand(self.unit[i], self.unit[i + 1])
            self.dW.append(dw)
            b = np.random.rand(self.unit[i + 1])
            self.B.append(b)

    def forward(self, _inputs):#順伝搬
        self.Z = []
        self.Z.append(_inputs)
        for i in range(len(self.unit) - 1):
            u = self.U(self.Z[i], self.W[i], self.B[i])
            if(i != len(self.unit) - 2):
                z = np.tanh(u)
            else:
                z = self.F.softmax(u)
            self.Z.append(z)
        return np.argmax(z, axis = 1)

    def U(self, x, w, b):#ユニットへの総入力を返す関数
        return np.dot(x, w) + b

    def calc_loss(self, label):#誤差の計算
        error = np.sum(label * np.log(self.Z[-1]), axis=1)
        return -np.mean(error)

    def calc_grad(self, w, b, z, delta):#勾配の計算
        w_grad = np.zeros_like(w)
        b_grad = np.zeros_like(b)
        N = float(z.shape[0])
        w_grad = np.dot(z.T, delta) / N
        b_grad = np.mean(delta, axis = 0)
        return w_grad, b_grad

    # 誤差逆伝搬
    def backPropagate(self, _label, eta, M):
        # calculate output_delta and error terms
        W_grad = []
        B_grad = []
        for i in range(len(self.W)):
            w_grad = np.zeros_like(self.W[i])
            W_grad.append(w_grad)
            b_grad = np.zeros_like(self.W[i])
            B_grad.append(b_grad)

        output = True

        delta = np.zeros_like(self.Z[-1])
        for i in range(len(self.W)):
            delta = self.calc_delta(delta, self.W[-(i)], self.Z[-(i + 1)], _label, output)
            W_grad[-(i + 1)], B_grad[-(i + 1)] = self.calc_grad(self.W[-(i + 1)], self.B[-(i + 1)], self.Z[-(i + 2)], delta)

        output = False

        #パラメータのチューニング
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - eta * W_grad[i] + M * self.dW[i]
            self.B[i] = self.B[i] - eta * B_grad[i]
            # モメンタムの計算
            self.dW[i] = -eta * W_grad[i] + M * self.dW[i]
        # デルタの計算
    
    def calc_delta(self, delta_dash, w, z, label, output):
        # delta_dash : 1つ先の層のデルタ
        # w : pre_deltaとdeltaを繋ぐネットの重み
        # z : wへ向かう出力
        if(output):
            delta = z - label
        else:
            delta = np.dot(delta_dash, w.T) * self.F.dtanh(z)
        return delta
    
    def train(self, dataset, N, interations=1000, minibatch=4, eta=0.5, M=0.5):
        print ("_____Train_____")
        #入力データ
        inputs = dataset[:, :self.unit[0]]
        #訓練データ
        label = dataset[:, self.unit[0]:]

        errors = []

        for val in range(interations):
            minibatch_errors = []
            for index in range(0, N, minibatch):
                _inputs = inputs[index: index + minibatch]
                _label = label[index: index + minibatch]
                self.forward(_inputs)
                self.backPropagate(_label, eta, M)

                loss = self.calc_loss(_label)
                minibatch_errors.append(loss)
            En = sum(minibatch_errors) / len(minibatch_errors)
            print ("epoch", val + 1, " : Loss = ", En)
            errors.append(En)
        print("\n")
        errors = np.asarray(errors)
        plt.plot(errors)

    def getWeight(self):#パラメータの値を取得
        for i in range(len(self.W)):
            print("W", i + 1, ":")
            print(self.W[i])
            print("\n")
            print("B", i + 1, ":")
            print(self.B[i])
            print("\n")

    def save_weight(self, name):
        import datetime
        today_detail = datetime.datetime.today()
        s = today_detail.strftime("%m-%d-%H-%M")
        np.save('Models/%s_W_%s.npy' % (name, s), self.W)
        np.save('Models/%s_B_%s.npy' % (name, s), self.B)
        print("Weight is saved!!")
        for i in range(len(self.W)):
            print("W", i + 1, ".shape = ", self.W[i].shape)
            print("B", i + 1, ".shape = ", self.B[i].shape)
        print("\n")

    def draw_test(self, x, label, W, B):
        self.W = W
        self.B = B
        x1_max = max(x[:, 0]) + 0.5
        x2_max = max(x[:, 1]) + 0.5
        x1_min = min(x[:, 0]) - 0.5
        x2_min = min(x[:, 1]) - 0.5
        xx, yy = np.meshgrid(np.arange(x1_min, x1_max, 0.01),np.arange(x2_min, x2_max, 0.01))
        Z = self.forward(np.c_[xx.ravel(),yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.subplot(2, 1 ,2)
        plt.contourf(xx, yy, Z, cmap=plt.cm.jet)
        plt.scatter(x[:, 0], x[:, 1], c=label, cmap=plt.cm.jet)
        plt.show()

import sklearn.datasets
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #データ生成
    x, label = sklearn.datasets.make_classification(n_features=2, n_samples=300, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=3)
    print(x,label)

    #グラフの描画
    plt.subplot(2, 1, 1)
    plt.scatter(x[:,0], x[:, 1], c=label, cmap=plt.cm.jet)
    #plt.show()

    unit = [2,3,3]
    minibatch = 20
    iterations = 3000
    eta = 0.1
    M = 0.1
    N = x.shape[0]
    dataset = np.column_stack((x, label))
    np.random.shuffle(dataset)

    brain = Newral_Network(unit)
    brain.train(dataset, N, iteration, minibatch, eta, M)
    brain.draw_test(x, label, brain.W, brain.B)
