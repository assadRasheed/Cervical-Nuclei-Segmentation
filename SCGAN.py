import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, Model
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np
import matplotlib.pyplot as plt

# ========================== CSFU Module ==========================
def CSFU_module(inputs, target_shape=(512, 512), target_channels=512):
    resized_features = [layers.Conv2D(target_channels, (1, 1), padding='same')(f) for f in inputs]
    resized_features = [layers.Resizing(*target_shape)(f) for f in resized_features]

    fused = layers.Add()(resized_features)
    fused = layers.BatchNormalization()(fused)
    fused = layers.DepthwiseConv2D((3, 3), padding='same', activation='swish')(fused)

    return fused

# ========================== TFA Block ==========================
def TFA_block(inputs):
    conv_v = layers.Conv2D(512, (15, 1), padding='same', activation='relu')(inputs)
    conv_h = layers.Conv2D(512, (1, 15), padding='same', activation='relu')(conv_v)
    combined = layers.Add()([inputs, conv_h])
    return combined

# ========================== UNet Generator for Segmentation ==========================
def build_generator(input_shape=(512, 512, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), padding='same', activation='mish')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), padding='same', activation='mish')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), padding='same', activation='mish')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), padding='same', activation='mish')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), padding='same', activation='mish')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), padding='same', activation='mish')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    conv4 = layers.Conv2D(512, (3, 3), padding='same', activation='mish')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), padding='same', activation='mish')(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)

    conv5 = layers.Conv2D(1024, (3, 3), padding='same', activation='mish')(pool4)
    conv5 = layers.Conv2D(1024, (3, 3), padding='same', activation='mish')(conv5)

    # Decoder
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = layers.Concatenate()([conv4, up6])
    conv6 = layers.Conv2D(512, (3, 3), padding='same', activation='mish')(merge6)
    conv6 = layers.Conv2D(512, (3, 3), padding='same', activation='mish')(conv6)

    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = layers.Concatenate()([conv3, up7])
    conv7 = layers.Conv2D(256, (3, 3), padding='same', activation='mish')(merge7)
    conv7 = layers.Conv2D(256, (3, 3), padding='same', activation='mish')(conv7)

    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = layers.Concatenate()([conv2, up8])
    conv8 = layers.Conv2D(128, (3, 3), padding='same', activation='mish')(merge8)
    conv8 = layers.Conv2D(128, (3, 3), padding='same', activation='mish')(conv8)

    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = layers.Concatenate()([conv1, up9])
    conv9 = layers.Conv2D(64, (3, 3), padding='same', activation='mish')(merge9)
    conv9 = layers.Conv2D(64, (3, 3), padding='same', activation='mish')(conv9)

    # Final Convolutional Layer
    conv10 = layers.Conv2D(64, (3, 3), padding='same', activation='mish')(conv9)
    conv10 = layers.Conv2D(64, (3, 3), padding='same', activation='mish')(conv10)

    # CSFU and TFA Integration
    csfu_output = CSFU_module([conv1, conv2, conv3, conv4, conv5], target_shape=(512, 512), target_channels=512)
    tfa_output = TFA_block(csfu_output)

    # Output Layer
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(tfa_output)

    model = Model(inputs, output)
    return model


# ========================== Synergic Discriminators for Segmentation ==========================
def build_discriminator(input_shape=(512, 512, 1)):
    # Define the input layer
    input_layer = layers.Input(shape=input_shape)

    # ResNet-50 based discriminator
    resnet_discriminator = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=input_layer)
    resnet_features = layers.GlobalAveragePooling2D()(resnet_discriminator.output)

    # EfficientNet-B2 based discriminator
    efficientnet_discriminator = tf.keras.applications.EfficientNetB2(include_top=False, weights=None, input_tensor=input_layer)
    efficientnet_features = layers.GlobalAveragePooling2D()(efficientnet_discriminator.output)

    # Concatenate the outputs of the two discriminators
    merged_features = layers.Concatenate()([resnet_features, efficientnet_features])

    # Synergic network
    synergic_output = layers.Dense(512, activation='relu')(merged_features)
    synergic_output = layers.Dense(1, activation='sigmoid')(synergic_output)

    synergic_model = Model(inputs=input_layer, outputs=synergic_output)

    return synergic_model

# ========================== SCGAN Model for Segmentation ==========================
class SCGAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(SCGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.cross_entropy = BinaryCrossentropy(from_logits=True)

    def compile(self, gen_optimizer, disc_optimizer):
        super(SCGAN, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def call(self, inputs, training=None):
        gen_images = self.generator(inputs)
        disc_real_output = self.discriminator([inputs, gen_images], training=True)
        return gen_images, disc_real_output

    def train_step(self, data):
        real_images, _ = data

        # Generate fake segmentation maps
        fake_images = self.generator(real_images)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            real_output = self.discriminator([real_images, real_images], training=True)
            fake_output = self.discriminator([real_images, fake_images], training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}

# ========================== Training Configuration ==========================
# Create generator and discriminator
generator = build_generator(input_shape=(512, 512, 3))
discriminator = build_discriminator(input_shape=(512, 512, 1))

# Create SCGAN model
scgan = SCGAN(generator, discriminator)

# Compile model
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.0002, decay_steps=530, end_learning_rate=0.0)
gen_optimizer = optimizers.Adam(learning_rate=lr_schedule)
disc_optimizer = optimizers.Adam(learning_rate=lr_schedule)

scgan.compile(gen_optimizer, disc_optimizer)

# ========================== Data Loading and Preprocessing ==========================
def preprocess_image(image, size):
    image = tf.image.resize(image, size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize the image to [0, 1] range
    return image

def load_dataset(image_paths, label_paths, image_size):
    images = [preprocess_image(tf.io.read_file(img_path), image_size) for img_path in image_paths]
    labels = [preprocess_image(tf.io.read_file(lbl_path), image_size) for lbl_path in label_paths]
    return tf.data.Dataset.from_tensor_slices((images, labels))

# Load your datasets here
# train_dataset = load_dataset(train_image_paths, train_label_paths, (512, 512))
# validation_dataset = load_dataset(val_image_paths, val_label_paths, (512, 512))

# ========================== Training Loop ==========================
# scgan.fit(train_dataset, epochs=530, batch_size=16)

# ========================== Evaluation ==========================
def evaluate_model(dataset, model):
    y_true = []
    y_pred = []

    for images, labels in dataset.batch(16):
        predictions = model(images, training=False)
        y_true.extend(labels.numpy().flatten())
        y_pred.extend(predictions.numpy().flatten() > 0.5)  # Threshold to binary

    # Calculate metrics
    sensitivity = recall_score(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)  # Custom function needed
    f1 = f1_score(y_true, y_pred)

    print(f'Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, F1-Score: {f1:.4f}')

def specificity_score(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp)

# After training, evaluate the model on the validation dataset
# evaluate_model(validation_dataset, scgan)

# ========================== Plot Training Dynamics ==========================
def plot_training_dynamics(history):
    plt.figure(figsize=(12, 8))

    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history.history['gen_loss'], label='Generator Loss')
    plt.plot(history.history['disc_loss'], label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # If metrics are tracked during training, you can plot them here

    plt.tight_layout()
    plt.show()

# Assuming history is available
# plot_training_dynamics(history)





