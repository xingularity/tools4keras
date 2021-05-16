from tensorflow.keras.callbacks import Callback
import datetime as dt
import time
import pickle


class HistorySaver(Callback):
    def __init__(self, file_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_name = file_name
        print(self.file_name)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            return
        
        if not isinstance(self.model.history.history.get('epoch'), list):
            self.model.history.history['epoch'] = list()
        self.model.history.history['epoch'].append(epoch)
        
        if not isinstance(self.model.history.history.get('timestamp'), list):
            self.model.history.history['timestamp'] = list()
        self.model.history.history['timestamp'].append(dt.datetime.now())
        with open(self.file_name, 'wb') as hist:
            pickle.dump(self.model.history.history, hist)
        print('HistorySaver writes file \'' + self.file_name + '\'\n')
            
    def load_history(self):
        with open(self.file_name, 'rb') as hist:
            return pickle.load(hist)
