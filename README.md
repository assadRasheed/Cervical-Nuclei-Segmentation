This repository contains an implementation of a Segmentation Conditional Generative Adversarial Network (SCGAN) built using TensorFlow and Keras. The SCGAN is designed for image segmentation tasks, combining several advanced neural network modules and architectures such as the U-Net generator, Synergic Discriminators, TFA Block, and CSFU (R-BiFPN) Module to enhance the performance of segmentation tasks.
**Prerequisites**
Python 3.7
TensorFlow 2.0
NumPy
Matplotlib
Scikit-learn

**Clone the repository**:git clone https://github.com/assadRasheed/Cervical-Nuclei-Segmentation.git
cd scgan-segmentation

**Model Compilation:**
Run the main script to build and compile the SCGAN model. The model summary will be printed to the console.
python main.py
history = model.fit(train_dataset, epochs=50)
plot_training_progress(history)
Note: You need to replace train_dataset with your actual dataset.

**Visualization:**
Use the plot_training_progress function to visualize the generator and discriminator losses during training.
**Customization**
Adjusting Model Architecture: You can modify the generator, discriminator, and SCGAN model architecture by editing the build_generator, build_discriminator, and SCGAN class.

**Training Hyperparameters:** Modify the learning rate and other hyperparameters by adjusting the gen_optimizer and disc_optimizer in the build_and_compile_model function.
