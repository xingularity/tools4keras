from tensorflow.keras.callbacks import Callback
import time


class SleepAWhile(Callback):
    def __init__(self, setup, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if 'epoch' in setup.keys():
            self.epoch_sleep_seconds = setup['epoch']
        if 'train_batch' in setup.keys():
            self.train_batch_sleep_interval = setup['train_batch'][0]
            self.train_batch_sleep_seconds = setup['train_batch'][1]

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            return
        if hasattr(self, 'epoch_sleep_seconds'):
            print("Start epoch {} of training; sleep {} seconds".format(epoch, self.epoch_sleep_seconds))
            time.sleep(self.epoch_sleep_seconds)

    def on_train_batch_begin(self, batch, logs=None):
        if batch == 0:
            return
        if hasattr(self, 'train_batch_sleep_interval'):
            if (batch % self.train_batch_sleep_interval) == 0:
                time.sleep(self.train_batch_sleep_seconds)
