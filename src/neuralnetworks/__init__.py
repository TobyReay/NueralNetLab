import pathlib
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
import numpy as np



def countClasses(
    path: pathlib.Path,
    types: list[str] = ["glioma", "notumor", "pituitary"]
) -> list[tuple[str, int]]:
    classes = []

    for type_name in types:
        directory = path / type_name
        count = sum(1 for file in directory.glob('*') if file.is_file())

        classes.append((type_name, count))

    return classes



def displayImage(images: list[pathlib.Path]):

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for i, image in enumerate(images):
        img = Image.open(image)
        print(image)
        print("image width " + str(img.width))
        print("image height " + str(img.height))

        axes[i].imshow(img)
        axes[i].set_title(image.name)
        axes[i].axis("off")

    for i in range(len(images), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def plot_loss_accuracy(history, name: str, smooth_sigma=2):
    # Extract loss and accuracy values
    loss_values = history.history['loss']
    acc_values = history.history['accuracy']

    # Validation (if available)
    val_loss = history.history.get('val_loss')
    val_acc = history.history.get('val_accuracy')

    epochs = np.arange(1, len(loss_values) + 1)

    # --- Plot Training & Validation Loss ---
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b', alpha=0.5, label='Training Loss')
    # plt.plot(epochs, gaussian_filter1d(loss_values, sigma=smooth_sigma), color='b', linewidth=2, label='Trend (Train)')

    if val_loss is not None:
        plt.plot(epochs, val_loss, marker='x', linestyle='--', color='r', alpha=0.5, label='Validation Loss')
        # plt.plot(epochs, gaussian_filter1d(val_loss, sigma=smooth_sigma), color='r', linewidth=2, label='Trend (Val)')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.grid(True)
    plt.legend()

    # --- Plot Training & Validation Accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_values, marker='o', linestyle='-', color='g', alpha=0.5, label='Training Accuracy')
    # plt.plot(epochs, gaussian_filter1d(acc_values, sigma=smooth_sigma), color='g', linewidth=2, label='Trend (Train)')

    if val_acc is not None:
        plt.plot(epochs, val_acc, marker='x', linestyle='--', color='orange', alpha=0.5, label='Validation Accuracy')
        # plt.plot(epochs, gaussian_filter1d(val_acc, sigma=smooth_sigma), color='orange', linewidth=2, label='Trend (Val)')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
   
    plt.savefig(name)


def main() -> None:
    SEED = 42
    NUM_EPOCHS = 120
    tf.keras.utils.set_random_seed(SEED)
    print('TensorFlow version:', tf.__version__)
    print('GPU devices:', tf.config.list_physical_devices('GPU'))


    #T1
    testing_dir = pathlib.Path("./BrainTumorDataset/Testing")
    training_dir = pathlib.Path("./BrainTumorDataset/Training")
    validation_dir = pathlib.Path("./BrainTumorDataset/Validation")

    test_classes = countClasses(testing_dir)
    train_classes = countClasses(training_dir)
    val_classes = countClasses(validation_dir)

    print("testing" + str(test_classes)) # Equal Distrubution
    print("training"+ str(train_classes)) # Non equal distribution no tumor has alot higher count of data could lead to overfitting and false reports on actual tumors predicting no tumor alot more
    print("validation" + str(val_classes)) # Equal Distribution

    #t1a we have 501 training for positive glioma

    #t1b displaying 1 image per class in validation
    images = [pathlib.Path("BrainTumorDataset/Validation/glioma/Tr-gl_1181.jpg"), pathlib.Path("BrainTumorDataset/Validation/notumor/Te-no_0310.jpg"), pathlib.Path("BrainTumorDataset/Validation/pituitary/Tr-pi_1317.jpg")]
    displayImage(images)
    # Images should have a common resolution in a dataset as differences can affect the training as when you define a model you need to define the input size 
    # one issue with resizing images is that features can be lost and streched affecting the training of the model

    batch_size = 32

    image_width = 256
    image_height = 256

    train_set = keras.utils.image_dataset_from_directory(
        training_dir,
        seed=None,
        image_size=(image_width, image_height),
        batch_size=batch_size,
        shuffle=False
    )

    validation_set = keras.utils.image_dataset_from_directory(
            validation_dir,
            seed=None,
            image_size=(image_width, image_height),
            batch_size=batch_size,
            shuffle=False
        )

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_normalized_ds = train_set.map(lambda x, y: (normalization_layer(x), y))
    validate_normalize_ds = validation_set.map(lambda x, y: (normalization_layer(x), y))

    image_batch, labels_batch = next(iter(train_normalized_ds))
    first_image= image_batch[0]
    print(np.min(first_image), np.max(first_image), first_image[0][0])

    y_train = np.concatenate([y for x, y in train_set], axis=0)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = dict(enumerate(class_weights))

    #Model Definitions 

   
    model_a = tf.keras.models.Sequential([
    # Input
    tf.keras.layers.Input(
        shape=train_normalized_ds.element_spec[0].shape[1:]
    ),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
   
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
  
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
  
    tf.keras.layers.Dense(3, activation="softmax")
    ])


    model_a.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )


    print(model_a.summary())

    model_b = tf.keras.models.Sequential([
    tf.keras.layers.Input(
        shape=train_normalized_ds.element_spec[0].shape[1:]
    ),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(3, activation="softmax")
]) 
    # Own experiement Global Average pooling
    model_d = tf.keras.models.Sequential([
        tf.keras.layers.Input(
            shape=train_normalized_ds.element_spec[0].shape[1:]
        ),
    
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
    
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
    
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
    
        tf.keras.layers.GlobalAveragePooling2D(),
    
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
    
        tf.keras.layers.Dense(3, activation="softmax")
    ]) 

    model_c = model_b

    model_b.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    print(model_b.summary())


    model_c.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"
          ]
        )

    print(model_c.summary())

    model_d.compile(
                optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"
              ]
            )

    print(model_d.summary())



    model_a_history = model_a.fit(
        train_normalized_ds,
        validation_data = validation_set,
        epochs = NUM_EPOCHS,
        batch_size = batch_size,
        verbose = 1,
        
    )

    model_b_history = model_b.fit(
        train_normalized_ds,
        validation_data = validation_set,
        epochs = NUM_EPOCHS,
        batch_size = batch_size,
        verbose = 1,
    )

    model_c_history = model_c.fit(
        train_normalized_ds,
        validation_data = validation_set,
        epochs = NUM_EPOCHS,
        batch_size = batch_size,
        verbose = 1,
        class_weight = class_weights
    )

    model_d_history = model_d.fit(
            train_normalized_ds,
            validation_data = validation_set,
            epochs = NUM_EPOCHS,
            batch_size = batch_size,
            verbose = 1,
            class_weight = class_weights
        )
    
    


    plot_loss_accuracy(model_a_history, "model_a")
    plot_loss_accuracy(model_b_history, "model_b")
    plot_loss_accuracy(model_c_history, "model_c")
    plot_loss_accuracy(model_d_history, "model_d")


if __name__ == "__main__":
    main()
