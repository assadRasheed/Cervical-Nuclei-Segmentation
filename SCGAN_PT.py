import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, Model
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import matplotlib.pyplot as plt

# ========================== CSFU Module ==========================
def R_BiFPN_module(inputs, num_blocks=2):
    feature_pyramids = inputs
    for _ in range(num_blocks):
        new_features = []
        for i in range(len(feature_pyramids)):
            current_feature = feature_pyramids[i]
            if i > 0:
                upsampled = layers.UpSampling2D()(new_features[-1])
                current_feature = layers.Add()([current_feature, upsampled])
            if i < len(feature_pyramids) - 1:
                downsampled = layers.MaxPooling2D()(feature_pyramids[i+1])
                current_feature = layers.Add()([current_feature, downsampled])
            new_features.append(layers.Conv2D(512, (3, 3), padding='same', activation='swish')(current_feature))
        feature_pyramids = new_features
    return feature_pyramids

def CSFU_module(inputs):
    # First R-BiFPN
    first_bifpn = R_BiFPN_module(inputs)

    # Second R-BiFPN
    csfu_output = R_BiFPN_module(first_bifpn)

    return csfu_output

# ========================== TFA Block ==========================
def TFA_block(inputs):
    conv_v = layers.Conv2D(1024, (15, 1), padding='same', activation='relu')(inputs)
    conv_h = layers.Conv2D(1024, (1, 15), padding='same', activation='relu')(conv_v)
    combined = layers.Add()([inputs, conv_h])
    return combined

# ========================== UNet Generator with CSFU ==========================
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

    # Apply TFA Block at the deepest layer (conv5)
    tfa_output = TFA_block(conv5)

    # CSFU Module (R-BiFPN) between Encoder and Decoder
    csfu_output = CSFU_module([conv1, conv2, conv3, conv4, tfa_output])

    # Decoder
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(csfu_output[3])
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

    # Output Layer
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv10)

    model = Model(inputs, output)
    return model

# ========================== Synergic Discriminators for Segmentation ==========================
def build_discriminator(input_shape=(512, 512, 1)):
    input_layer = layers.Input(shape=input_shape)

    resnet_discriminator = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=input_layer)
    resnet_features = layers.GlobalAveragePooling2D()(resnet_discriminator.output)

    efficientnet_discriminator = tf.keras.applications.EfficientNetB2(include_top=False, weights=None, input_tensor=input_layer)
    efficientnet_features = layers.GlobalAveragePooling2D()(efficientnet_discriminator.output)

    merged_features = layers.Concatenate()([resnet_features, efficientnet_features])

    synergic_output = layers.Dense(512, activation='relu')(merged_features)
    synergic_output = layers.Dense(1, activation='sigmoid')(synergic_output)

    synergic_model = Model(inputs=input_layer, outputs=synergic_output)
    return synergic_model

# ========================== SCGAN Model for Progressive Resizing ==========================
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
        disc_real_output = self.discriminator(inputs)
        disc_fake_output = self.discriminator(gen_images)
        return {"gen_images": gen_images, "disc_real_output": disc_real_output, "disc_fake_output": disc_fake_output}

    def train_step(self, data):
        real_images = data

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(real_images, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}

# ========================== Progressive Resizing and Transfer Learning ==========================
def train_progressive_resizing(initial_model, dataset, image_sizes, epochs_per_size=5):
    models = []
    current_model = initial_model

    for size in image_sizes:
        print(f"Training with image size: {size}x{size}")

        # Resize dataset
        resized_dataset = dataset.map(lambda x, y: (tf.image.resize(x, [size, size]), tf.image.resize(y, [size, size])))

        # Create a new model and transfer weights
        new_model = build_generator(input_shape=(size, size, 3))
        new_model.set_weights(current_model.get_weights())

        # Create SCGAN with new generator
        discriminator = build_discriminator(input_shape=(size, size, 1))
        scgan = SCGAN(new_model, discriminator)

        scgan.compile(
            gen_optimizer=optimizers.Adam(1e-4),
            disc_optimizer=optimizers.Adam(1e-4)
        )

        # Train model
 notably adjust
        scgan.fit(resized_dataset, epochs=epochs_per_size)

        # Add the trained model to the list
        models.append(scgan.generator)

        # Update current model
        current_model = new_model

    return models

# Usage Example:
# image_sizes = [256, 325, 512]
# initial_generator = build_generator(input_shape=(256, 256, 3))
# trained_models = train_progressive_resizing(initial_generator other adjusting
# other real_dataset, image_sizes)