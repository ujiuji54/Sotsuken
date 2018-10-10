import os
import datetime
import wave
import yaml
import numpy as np
from numpy.lib.stride_tricks import as_strided
from keras.models import Model
from keras.layers import Input, Dense, Activation, CuDNNLSTM, BatchNormalization, Multiply
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.losses import mean_squared_error

def load_wave(wave_file):
    with wave.open(wave_file, "r") as w:
        buf = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (buf / 0x7fff).astype(np.float32)

def load_aux(aux):
    buf = np.array([aux])
    return (buf / 0x7fff).astype(np.float32)

def save_wave(buf, wave_file, sampling_rate):
    _buf = (buf * 0x7fff).astype(np.int16)
    with wave.open(wave_file, "w") as w:
        w.setparams((1, 2, sampling_rate, len(_buf), "NONE", "not compressed"))
        w.writeframes(_buf)

def flow(dataset, timesteps, batch_size):
    n_data = len(dataset)
    while True:
        i = np.random.randint(n_data)
        x = dataset[i][0]
        y = dataset[i][1]
        aux = dataset[i][2]
        yield random_clop(x, aux, y, timesteps, batch_size)

def random_clop(x, aux, y, timesteps, batch_size):
    max_offset = len(x) - timesteps
    offsets = np.random.randint(max_offset, size=batch_size)
    batch_x = np.stack((x[offset:offset+timesteps] for offset in offsets))
    batch_y = np.stack((y[offset:offset+timesteps] for offset in offsets))
    return [batch_x, aux], batch_y 

def build_model(timesteps):
    main_input = Input((timesteps, 1))
    aux_input = Input((1,1))
    x = CuDNNLSTM(64, return_sequences=True)(main_input)
    y = Dense(64, activation="sigmoid")(aux_input)
    x = Multiply()([x,y])
    x = CuDNNLSTM(64, return_sequences=True)(x)
    output = CuDNNLSTM(1, return_sequences=True)(x)
    model = Model([main_input, aux_input], output)
    return model

class LossFunc:

    def __init__(self, timesteps):
        self.__name__ = "LossFunc"
        self.timesteps = timesteps
    
    def __call__(self, y_true, y_pred):
        return mean_squared_error(
            y_true[:, -self.timesteps:, :],
            y_pred[:, -self.timesteps:, :])

def train(model, train_dataflow, val_dataflow, max_epochs, patience):
    timestamp = datetime.datetime.now()

    cp_dir = "./checkpoint/{:%Y%m%d_%H%M%S}".format(timestamp)
    if not os.path.exists(cp_dir):
        os.makedirs(cp_dir)
    cp_filepath = os.path.join(cp_dir, "model_{epoch:06d}.h5")
    cb_mc = ModelCheckpoint(filepath=cp_filepath, monitor="val_loss", period=1, save_best_only=True)

    cb_es = EarlyStopping(monitor="val_loss", patience=patience)

    tb_log_dir = "./tensorboard/{:%Y%m%d_%H%M%S}".format(timestamp)
    cb_tb = TensorBoard(log_dir=tb_log_dir)

    model.fit_generator(
        generator=train_dataflow,
        steps_per_epoch=100,
        validation_data=val_dataflow,
        validation_steps=10,
        epochs=max_epochs,
        callbacks=[cb_mc, cb_es, cb_tb])

def sliding_window(x, window, slide):
    n_slide = (len(x) - window) // slide
    remain = (len(x) - window) % slide
    clopped = x[:-remain]
    return as_strided(clopped, shape=(n_slide + 1, window), strides=(slide * 4, 4))
