
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, Model
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np
import matplotlib.pyplot as plt

# ========================== R-BiFPN Module ==========================
def R_BiFPN_module(inputs, num_blocks=1):
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

# ========================== CSFU Module ==========================
def CSFU_module(inputs):
    # Apply the first R-BiFPN block
    features = R_BiFPN_module(inputs, num_blocks=1)
    # Apply the second R-BiFPN block on the output of the first
    features = R_BiFPN_module(features, num_blocks=1)
    return features

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

    # Apply TFA Block at the deepest layer (conv5)
    tfa_output = TFA_block(conv5)

    # CSFU Integration: Apply the CSFU (two R-BiFPN modules) after the encoder
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
        disc_real_output = self.discriminator(inputs)
        disc_fake_output = self.discriminator(gen_images)
        return {"gen_images": gen_images, "disc_real_output": disc_real_output, "disc_fake_output": disc_fake_output}

    def train_step(self, data):
        real_images = data

        # Generate fake images
        with tf.GradientTape() as gen_tape:
            fake_images = self.generator(real_images, training=True)

        # Train the discriminator
        with tf.GradientTape() as disc_tape:
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as gen_tape:
            fake_images = self.generator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            gen_loss = self.generator_loss(fake_output)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}

# Instantiate the generator and discriminator models
generator = build_generator()
discriminator = build_discriminator()

# Instantiate the SCGAN model and compile it
scgan = SCGAN(generator, discriminator)
scgan.compile(
    gen_optimizer=optimizers.Adam(learning_rate=1e-4),
    disc_optimizer=optimizers.Adam(learning_rate=1e-4),
)

# ========================== Training ==========================
# Define training data and labels
train_images = np.random.rand(10, 512, 512, 3)  # Placeholder for training data
train_labels = np.random.rand(10, 512, 512, 1)  # Placeholder for training labels

# Train the SCGAN model
scgan.fit(train_images, epochs=10)