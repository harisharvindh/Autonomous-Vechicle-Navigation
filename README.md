# Autonomous-Vechicle-Navigation
Utilizing UDEMY Simulator to train a model for autonomous driving

The Udemy self-driving car simulator, encapsulated within the PowerMode_autopilot class, is an advanced and sophisticated machine learning framework designed to train an autonomous driving model. This Python-based solution utilizes a combination of computer vision and deep learning techniques to interpret the driving environment and make decisions accordingly.

At its core, the simulator processes data collected from simulated driving sessions, where images from three different camera angles (left, center, right) are used alongside steering angles as the primary dataset. This data is meticulously parsed from a driving_log.csv file, which is then divided into training and validation sets to ensure the model learns effectively and generalizes well to unseen data.

The model architecture is constructed using Keras, a high-level neural networks API, known for its user-friendliness and modular approach. The model starts with a normalization layer to standardize input images, followed by a series of convolutional layers designed to extract spatial hierarchies of features from the images. Each convolutional layer is accompanied by activation functions, pooling, dropout, and batch normalization layers to enhance learning efficiency and mitigate overfitting.

A distinctive feature of the model is its use of Exponential Linear Units (ELU) for activations, providing smoother non-linear transformations that help with faster convergence. The model also employs spatial dropout and L2 regularization as techniques to prevent overfitting, ensuring the model learns robust and generalized features.

The training process is supported by dynamic data augmentation techniques such as random flips, translations, brightness adjustments, and shadow introductions to simulate a variety of driving conditions. This not only enriches the dataset but also equips the model to handle real-world lighting and environmental variations.

Further preprocessing steps, including cropping irrelevant sections of the images, resizing to meet the input requirements of the neural network, and converting RGB images to YUV color space, are crucial for reducing computational load and focusing the model’s learning on relevant features.

A batch generator efficiently handles memory usage by dynamically loading and processing images in batches during the training process, facilitating the training of large datasets on hardware with limited memory.

The training loop leverages Keras’ fit_generator method, allowing for real-time data augmentation and parallel processing of data enhancement and model training. This approach optimizes GPU utilization for model training while simultaneously preparing the next batch of data on the CPU, ensuring efficient use of computational resources
