import os
import tensorflow as tf

def Callback(cfg):
    cfg_callback = cfg['CALLBACK']

    callbacks = list()
    checkpoint_callback(callbacks, cfg_callback['checkpoint'])
    plateau_callback(callbacks, cfg_callback['plateau'])
    #scheduler_callback(callbacks, cfg_callback['scheduler'])

    return callbacks

def checkpoint_callback(callbacks, cfg):
    if not cfg['flag_run'] : return

    outputFolder = cfg['output_dir']
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    filepath=outputFolder + '/' + cfg['save_name'] + '-{epoch:02d}.hdf5'

    callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath,
                        monitor=cfg['monitor'],
                        verbose=cfg['verbose'],
                        save_best_only=cfg['save_best_only'],
                        save_weights_only=cfg['save_weights_only'],
                        mode=cfg['mode'],
                        save_freq=cfg['save_freq'],
                        save_frequency=cfg['save_frequency']
                )

    callbacks.append(callback)

def plateau_callback(callbacks, cfg):
    if not cfg['flag_run'] : return

    callback = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=cfg['factor'],
                        patience=cfg['patience'],
                        verbose=cfg['verbose'],
                        mode=cfg['mode'],
                        min_delta=cfg['min_delta'],
                        cooldown=cfg['cooldown'],
                        min_lr=cfg['min_lr']
    )

    callbacks.append(callback)

def scheduler_callback(callbacks, cfg):
    if not cfg['flag_run'] : return

    def scheduler(epoch):
        if epoch < 10:
            return 0.001
        else:
            return 0.001 * tf.math.exp(0.1 * (10 - epoch))

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    callbacks.append(callback)
