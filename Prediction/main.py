from util import parser, construct_dataset
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
import os
from model import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



if __name__ == '__main__':

    parser = parser()
    parser = parser.parser
    args = parser.parse_args()

    train_df, test_df = construct_dataset(args)

    if args.model == 'Full':
        input_loader = full_input_loader
        create_dataset = create_full_dataset
        create_model = create_full_model
    else:
        input_loader = img_only_input_loader
        create_dataset = create_img_only_dataset
        create_model = create_img_only_model

    dataset_test = create_dataset(test_df, input_loader=input_loader, batch_size=args.batch_size)

    if args.train:

        print("Start to Train")

        # split train data into train : validation
        # This split ratio --> train : validation : test = 8 : 1 : 1
        train_df, val_df = train_test_split(train_df, test_size=0.11, random_state=args.seed)
        dataset_tr = create_dataset(train_df, input_loader=input_loader, batch_size=args.batch_size)
        dataset_val = create_dataset(val_df, input_loader=input_loader, batch_size=args.batch_size)

        if len(args.gpus) > 1:
            mirrored_strategy = tf.distribute.MirroredStrategy(devices=args.gpus)
            with mirrored_strategy.scope():
                model = create_model(args.train)
                model.compile(loss="mse", optimizer="adam",
                              metrics=[keras.metrics.mse, keras.metrics.mean_absolute_error])
        else:
            model = create_model(args.train)
            model.compile(loss="mse", optimizer="adam", metrics=[keras.metrics.mse, keras.metrics.mean_absolute_error])

        cb_earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=0, mode="min")
        cb_bestSave = tf.keras.callbacks.ModelCheckpoint(args.save_model, save_best_only=True, monitor="val_loss", mode="min")
        cb_lr_scheduling = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode="min")

        train_log = model.fit(dataset_tr,
                              epochs=args.epochs,
                              validation_data=dataset_val,
                              callbacks=[cb_earlyStopping, cb_bestSave, cb_lr_scheduling])

    # Evaluate the trained model with test dataset
    print("Start to Evaluate")
    evaluation_model = tf.keras.models.load_model(args.save_model)
    acc = evaluation_model.evaluate(dataset_test, batch_size=args.batch_size)

