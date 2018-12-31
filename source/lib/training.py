import keras as K
import keras.callbacks as kc
import keras.optimizers as ko

from utils.naming import *
from utils.logging import *


def train_model(model_generator, train, valid, loss,
                max_epochs=20,
                patience=0,
                model_name=None,
                additional_callbacks=None,
                learning_rate=1e-3) -> K.models.Model:
    """
    Train a model with all kinds of log services and optimizations we could come up with.
    Clears completely the session at each call to have separated training sessions of different models

    :param model_generator: a function returning an instance of the model to be trained
    :param loss: the loss function to be used. Must conform to the keras convention for losses:
                    - must accept two arguments:
                        1) y_true: the ground truth
                        2) y_pred: the network prediction
                    - must return a value to be minimized
    :param model_name: the model name used for saving checkpoints and the final model on file.
    :param learning_rate: The learning rate to use on Adam.
    :param max_epochs:  The maximum number of epochs to perform.
    :param patience: The early stopping patience. If None, disables early stopping.
    :param additional_callbacks: Any additional callbacks to add to the fitting functoin
    :return: The trained model, if early stopping is active this is the best model selected.
    """

    K.backend.clear_session()

    model = model_generator()

    if model_name is None:
        model_name = model.name

    checkpoint_path = models_path(model_name+".ckp")
    h5model_path = models_path(model_name+".h5")
    log("Model:", level=COMMENTARY)
    model.summary(print_fn=lambda s: log(s, level=COMMENTARY))

    callbacks = []
    if checkpoint_path is not None:
        log("Adding callback for checkpoint...", level=COMMENTARY)
        callbacks.append(kc.ModelCheckpoint(filepath=checkpoint_path,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='min',
                                            period=1))
    if patience > 0:
        log("Adding callback for early stopping...", level=COMMENTARY)
        callbacks.append(kc.EarlyStopping(patience=patience,
                                          verbose=1,
                                          monitor='val_loss',
                                          mode='min',
                                          min_delta=2e-4))

    # if tb_path is not None:
        # TBManager.set_path(tensorboard_path(model_name))
        # log("Setting up tensorboard...", level=COMMENTARY)
        # log("Clearing tensorboard files...", level=COMMENTARY)
        # TBManager.clear_data()

        # log("Adding tensorboard callbacks...", level=COMMENTARY)
        # callbacks.append(ScalarWriter())
        # if tb_plots is not None:
        #     callbacks.append(ImageWriter(data_sequence=train_data,
        #                                  image_generators=tb_plots,
        #                                  name='train',
        #                                  max_items=10))
        #     callbacks.append(ImageWriter(data_sequence=valid_data,
        #                                  image_generators=tb_plots,
        #                                  name='validation',
        #                                  max_items=10))
    if additional_callbacks is not None:
        callbacks += additional_callbacks

    # Training tools
    optimizer = ko.adam(lr=learning_rate)

    log("Compiling model...", level=COMMENTARY)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    history = model.fit(train[0], train[1],
                        validation_data=valid,
                        verbose=1,
                        epochs=max_epochs,
                        callbacks=callbacks)

    if h5model_path is not None:
        log("Saving H5 model...", level=COMMENTARY)
        model = model_generator()
        model.load_weights(checkpoint_path)
        model.save(h5model_path)
        os.remove(checkpoint_path)

    log("Training completed!", level=COMMENTARY)

    return model
