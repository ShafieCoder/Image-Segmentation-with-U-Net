# Image-Segmentation_with_U-Net

We are going to build our own U-Net, a type of CNN designed for quick, precise image segmentation, and using it to predict a label for every single pixel in an image - in this case, an image from a self-driving car dataset.

This type of image classification is called semantic image segmentation. It's similar to object detection in that both ask the question: "What objects are in this image and where in the image are those objects located?," but where object detection labels objects with bounding boxes that may include pixels that aren't part of the object, semantic image segmentation allows you to predict a precise mask for each object in the image by labeling each pixel in the image with its corresponding class. The word “semantic” here refers to what's being shown, so for example the “Car” class is indicated below by the dark blue mask, and "Person" is indicated with a red mask:

<p align="center">
  <img width="500" src="https://github.com/ShafieCoder/Image-Segmentation-with-U-Net/blob/main/images/carseg.png" alt="Example of segmented image">
</p>

 Region-specific labeling is a pretty crucial consideration for self-driving cars, which require a pixel-perfect understanding of their environment so they can change lanes and avoid other cars, or any number of traffic obstacles that can put peoples' lives in danger.
 
 ## 1- U-Net
 
U-Net, named for its U-shape, was originally created in 2015 for tumor detection, but in the years since has become a very popular choice for other semantic segmentation tasks.

U-Net builds on a previous architecture called the Fully Convolutional Network, or FCN, which replaces the dense layers found in a typical CNN with a transposed convolution layer that upsamples the feature map back to the size of the original input image, while preserving the spatial information. This is necessary because the dense layers destroy spatial information (the "where" of the image), which is an essential part of image segmentation tasks. An added bonus of using transpose convolutions is that the input size no longer needs to be fixed, as it does when dense layers are used.

Unfortunately, the final feature layer of the FCN suffers from information loss due to downsampling too much. It then becomes difficult to upsample after so much information has been lost, causing an output that looks rough.

U-Net improves on the FCN, using a somewhat similar design, but differing in some important ways. Instead of one transposed convolution at the end of the network, it uses a matching number of convolutions for downsampling the input image to a feature map, and transposed convolutions for upsampling those maps back up to the original input image size. It also adds skip connections, to retain information that would otherwise become lost during encoding. Skip connections send information to every upsampling layer in the decoder from the corresponding downsampling layer in the encoder, capturing finer information while also keeping computation low. These help prevent information loss, as well as model overfitting.

#### 1.1- Model Details
<p align="center">
  <img width="700" src="https://github.com/ShafieCoder/Image-Segmentation-with-U-Net/blob/main/images/unet.png" alt="U-Net Architecture">
</p>

__Contracting path__ (Encoder containing downsampling steps):

Images are first fed through several convolutional layers which reduce height and width, while growing the number of channels.

The contracting path follows a regular CNN architecture, with convolutional layers, their activations, and pooling layers to downsample the image and extract its features. In detail, it consists of the repeated application of two 3 x 3 unpadded convolutions, each followed by a rectified linear unit (ReLU) and a 2 x 2 max pooling operation with stride 2 for downsampling. At each downsampling step, the number of feature channels is doubled.

**Crop function**: This step crops the image from the contracting path and concatenates it to the current image on the expanding path to create a skip connection.

**Expanding path** (Decoder containing upsampling steps):

The expanding path performs the opposite operation of the contracting path, growing the image back to its original size, while shrinking the channels gradually.

In detail, each step in the expanding path upsamples the feature map, followed by a 2 x 2 convolution (the transposed convolution). This transposed convolution halves the number of feature channels, while growing the height and width of the image.

Next is a concatenation with the correspondingly cropped feature map from the contracting path, and two 3 x 3 convolutions, each followed by a ReLU. You need to perform cropping to handle the loss of border pixels in every convolution.

**Final Feature Mapping Block**: In the final layer, a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. The channel dimensions from the previous layer correspond to the number of filters used, so when you use 1x1 convolutions, you can transform that dimension by choosing an appropriate number of 1x1 filters. When this idea is applied to the last layer, you can reduce the channel dimensions to have one layer per class.

The U-Net network has 23 convolutional layers in total.

#### 1.2- Encoder (Downsampling Block)

<p align="center">
  <img width="700" src="https://github.com/ShafieCoder/Image-Segmentation-with-U-Net/blob/main/images/encoder.png" alt="U-Net Encoder">
</p>

The encoder is a stack of various conv_blocks:

Each conv_block() is composed of 2 **Conv2D** layers with ReLU activations. We will apply **Dropout**, and **MaxPooling2D** to some conv_blocks, as you will verify in the following sections, specifically to the last two blocks of the downsampling.

The function will return two tensors:
* next_layer: That will go into the next block.
* skip_connection: That will go into the corresponding decoding block.

**Note**: If max_pooling=True, the next_layer will be the output of the MaxPooling2D layer, but the skip_connection will be the output of the previously applied layer(Conv2D or Dropout, depending on the case). Else, both results will be identical.

**conve_block**
Here are the instructions for each step in the conv_block, or contracting block:
* Add 2 Conv2D layers with n_filters filters with kernel_size set to 3, kernel_initializer set to ['he_normal'](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal), padding set to 'same' and 'relu' activation.
* if dropout_prob > 0, then add a Dropout layer with parameter dropout_prob
* If max_pooling is set to True, then add a MaxPooling2D layer with 2x2 pool size

#### 1.3 - Decoder (Upsampling Block)
The decoder, or upsampling block, upsamples the features back to the original image size. At each upsampling level, you'll take the output of the corresponding encoder block and concatenate it before feeding to the next decoder block.
<p align="center">
  <img width="700" src="https://github.com/ShafieCoder/Image-Segmentation-with-U-Net/blob/main/images/decoder.png" alt="U-Net decoder">
</p>

There are two new components in the decoder: up and merge. These are the transpose convolution and the skip connections. In addition, there are two more convolutional layers set to the same parameters as in the encoder.

Here you'll encounter the Conv2DTranspose layer, which performs the inverse of the Conv2D layer. You can read more about it [here](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose). 

**upsampling_block**

To implement the function `upsampling_block(...)`: 
* Takes the arguments `expansive_input` (which is the input tensor from the previous layer) and `contractive_input` (the input tensor from the previous skip layer)
* The number of filters here is the same as in the downsampling block you completed previously
* Your `Conv2DTranspose` layer will take `n_filters` with shape (3,3) and a stride of (2,2), with padding set to `same`. It's applied to `expansive_input`, or the input tensor from the previous layer. 

This block is also where you'll concatenate the outputs from the encoder blocks, creating skip connections. 

* Concatenate your Conv2DTranspose layer output to the contractive input, with an `axis` of 3. In general, you can concatenate the tensors in the order that you prefer. But for the grader, it is important that you use `[up, contractive_input]`

For the final component, set the parameters for two Conv2D layers to the same values that you set for the two Conv2D layers in the encoder (ReLU activation, He normal initializer, `same` padding). 

#### 1.4 - Build the Model
This is where we'll put it all together, by chaining the encoder, bottleneck, and decoder! We'll need to specify the number of output channels, which for this particular set would be 23. That's because there are 23 possible labels for each pixel in this self-driving car dataset.

**unet_model**

For the function `unet_model`, specify the input shape, number of filters, and number of classes (23 in this case).

For the first half of the model:

* Begin with a `conv block` that takes the inputs of the model and the number of filters
* Then, chain the first output element of each block to the input of the next convolutional block
* Next, double the number of filters at each step
* Beginning with `conv_block4`, add `dropout_prob` of 0.3
* For the final `conv_block`, set `dropout_prob` to 0.3 again, and turn off max pooling

For the second half:

* Use cblock5 as `expansive_input` and cblock4 as `contractive_input`, with n_filters * 8. This is your `bottleneck layer`.
* Chain the output of the previous block as `expansive_input` and the corresponding `contractive block output`.
* Note that you must use the second element of the `contractive block` before the `max pooling layer`.
* At each step, use half the number of filters of the previous block
* conv9 is a `Conv2D layer` with `ReLU` activation, He normal initializer, `same` padding
* Finally, conv10 is a Conv2D that takes the number of classes as the filter, a kernel size of 1, and `"same"` padding. The output of conv10 is the output of your model.

#### 1.4 - Loss Function
In semantic segmentation, you need as many masks as you have object classes. In the dataset you're using, each pixel in every mask has been assigned a single integer probability that it belongs to a certain class, from 0 to num_classes-1. The correct class is the layer with the higher probability.

This is different from `categorical crossentropy`, where the labels should be one-hot encoded (just 0s and 1s). Here, you'll use sparse categorical crossentropy as your loss function, to perform pixel-wise multiclass prediction. Sparse categorical crossentropy is more efficient than other loss functions when you're dealing with lots of classes.

#### 1.5 - Dataset Handling

Below, define a function that allows you to display both an input image, and its ground truth: the true mask. The true mask is what your trained model output is aiming to get as close to as possible.

## 2 - Train the Model
In this stage we train the model, and by defining a function uses `tf.argmax` in the axis of the number of classes to return the index with the largest value and merge the prediction into a single image. Finally we can check our predicted masks against the true mask and the original input image to chack final results.






