
import tensorflow as tf
from tensorflow.keras import applications, layers, models, optimizers, datasets
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# Set random seed
tf.random.set_seed(42)
np.random.seed(42)


class PretrainedImageClassifier:

    def __init__(self, model_name='MobileNetV2',
                 img_size=(96, 96), num_classes=10):

        self.model_name = model_name
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def build_model(self, trainable_layers=5, dropout_rate=0.3):

        model_dict = {
            'VGG16': applications.VGG16,
            'ResNet50': applications.ResNet50,
            'MobileNetV2': applications.MobileNetV2,
            'EfficientNetB0': applications.EfficientNetB0
        }

        if self.model_name not in model_dict:
            raise ValueError("Unsupported model name")

        base_model = model_dict[self.model_name](
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size[0],
                         self.img_size[1], 3)
        )

        base_model.trainable = False

        inputs = layers.Input(
            shape=(self.img_size[0], self.img_size[1], 3)
        )

        # Model-specific preprocessing
        if self.model_name == 'MobileNetV2':
            x = applications.mobilenet_v2.preprocess_input(inputs)
        elif self.model_name == 'VGG16':
            x = applications.vgg16.preprocess_input(inputs)
        elif self.model_name == 'ResNet50':
            x = applications.resnet50.preprocess_input(inputs)
        else:
            x = inputs

        x = base_model(x, training=False)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)

        outputs = layers.Dense(
            self.num_classes,
            activation='softmax'
        )(x)

        self.model = models.Model(inputs, outputs)

        # Fine-tuning selected layers
        if trainable_layers > 0:
            base_model.trainable = True

            fine_tune_at = max(
                0, len(base_model.layers) - trainable_layers
            )

            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False

            print(f"Fine-tuning {trainable_layers} layers")

        return self.model

    def compile_model(self, learning_rate=0.001):

        self.model.compile(
            optimizer=optimizers.Adam(
                learning_rate=learning_rate
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def prepare_data(self):

        (x_train, y_train), (x_test, y_test) = (
            datasets.cifar10.load_data()
        )

        y_train = y_train.flatten().astype('int32')
        y_test = y_test.flatten().astype('int32')

        # Split training data into training and validation
        split = int(0.8 * len(x_train))

        x_val = x_train[split:]
        y_val = y_train[split:]

        x_train = x_train[:split]
        y_train = y_train[:split]

        # Resize images
        x_train = tf.image.resize(x_train, self.img_size)
        x_val = tf.image.resize(x_val, self.img_size)
        x_test = tf.image.resize(x_test, self.img_size)

        return (
            (x_train, y_train),
            (x_val, y_val),
            (x_test, y_test)
        )

    def train(self, x_train, y_train, x_val, y_val,
              epochs=5, batch_size=32):

        callbacks = [
            EarlyStopping(
                patience=2,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                factor=0.2,
                patience=1
            )
        ]

        self.history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def evaluate(self, x_test, y_test):

        results = self.model.evaluate(
            x_test, y_test, verbose=0
        )

        print(
            f"Test Loss: {results[0]:.4f}, "
            f"Test Accuracy: {results[1] * 100:.2f}%"
        )

        predictions = self.model.predict(
            x_test, verbose=0
        )

        predicted_classes = np.argmax(
            predictions, axis=1
        )

        class_names = [
            'airplane', 'automobile', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predicted_classes,
                target_names=class_names,
                zero_division=0
            )
        )

        return results

    def predict_single(self, image):

        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        predictions = self.model.predict(
            image, verbose=0
        )

        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        return predicted_class, confidence

    def plot_training_history(self):

        if self.history is None:
            print("No training history available")
            return

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 4)
        )

        ax1.plot(
            self.history.history['accuracy'],
            label='Training Accuracy'
        )
        ax1.plot(
            self.history.history['val_accuracy'],
            label='Validation Accuracy'
        )
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        ax2.plot(
            self.history.history['loss'],
            label='Training Loss'
        )
        ax2.plot(
            self.history.history['val_loss'],
            label='Validation Loss'
        )
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()
        plt.show()


# Main program
if __name__ == "__main__":

    class_names = [
        'airplane', 'automobile', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    classifier = PretrainedImageClassifier(
        model_name='MobileNetV2',
        img_size=(96, 96),
        num_classes=10
    )

    print("Building model...")
    model = classifier.build_model(
        trainable_layers=5,
        dropout_rate=0.3
    )

    print("Compiling model...")
    classifier.compile_model(learning_rate=0.001)

    print("Preparing CIFAR-10 dataset...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = (
        classifier.prepare_data()
    )

    print("Training data shape:", x_train.shape)
    print("Validation data shape:", x_val.shape)
    print("Test data shape:", x_test.shape)

    print("Training model...")
    classifier.train(
        x_train, y_train,
        x_val, y_val,
        epochs=5,
        batch_size=32
    )

    print("Evaluating model...")
    classifier.evaluate(x_test, y_test)

    classifier.plot_training_history()

    # Save model
    model.save('cifar10_classifier.keras')
    print("Model saved successfully!")

    # Predict one sample image
    sample_idx = 0
    sample_image = x_test[sample_idx]
    true_label = y_test[sample_idx]

    predicted_class, confidence = (
        classifier.predict_single(sample_image)
    )

    print("\nSample Prediction:")
    print("True class:", class_names[true_label])
    print("Predicted class:", class_names[predicted_class])
    print("Confidence:", f"{confidence:.4f}")

    # Display image
    plt.figure(figsize=(4, 4))
    plt.imshow(sample_image.numpy().astype('uint8'))
    plt.title(
        f"True: {class_names[true_label]}\n"
        f"Predicted: {class_names[predicted_class]}"
    )
    plt.axis('off')
    plt.show()

    print("\nPre-trained Image Classification Completed!")
