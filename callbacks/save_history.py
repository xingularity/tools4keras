from tensorflow.keras.callbacks import Callback
import datetime as dt
import time
import pickle


class HistorySaver(Callback):
    def __init__(self, file_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_name = file_name

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            return

        self.model.history.history['epoch'] = epoch
        self.model.history.history['timstamp'] = dt.datetime.now()
        with open(self.file_name, 'wb') as hist:
            pickle.dump(model.history.history, hist)
