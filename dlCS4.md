# Week 1

- If you only use a standard NN for larger images, the number of features might get overwhelming. For example, a 1000 pixels image would have $1000 \times 1000 \times 3 = 3 \cdot 10^6$ features, which is a bit unfeasible.
- Vertical edge detection with **Convolution**:
  \[
  \begin{bmatrix}
    3 & 0 & 1 & 2 & 7 & 4 \\
    1 & 5 & 8 & 9 & 3 & 1 \\
    2 & 7 & 2 & 5 & 1 & 3 \\
    0 & 1 & 3 & 1 & 7 & 8 \\
    4 & 2 & 1 & 6 & 2 & 8 \\
    2 & 4 & 5 & 2 & 3 & 9
  \end{bmatrix} \ast
  \begin{bmatrix}
    1 & 0 & -1 \\
    1 & 0 & -1 \\
    1 & 0 & -1
  \end{bmatrix}
  =
  \begin{bmatrix}
     -5 & -4 &  0 &   8 \\
    -10 & -2 &  2 &   3 \\
      0 & -2 & -4 &  -7 \\
     -3 & -2 & -3 & -16
  \end{bmatrix}
  \]
  - Alternatively, you can use:
    - Sobel Filter:
      \[
      \begin{bmatrix}
        1 & 0 & -1 \\
        2 & 0 & -2 \\
        1 & 0 & -1
      \end{bmatrix}
      \]
    - Scharr Filter:
    \[
    \begin{bmatrix}
       3 & 0 &  -3 \\
      10 & 0 & -10 \\
       3 & 0 &  -3
    \end{bmatrix}
    \]
  - CNNs can learn the weights of the filters by themselves, instead of you handpicking them.
  - The dimension in the convolution will be:
    \[
    M_{(n,n)} \ast W_{(f,f)} = Y_{(n-f+1,n-f+1)}
    \]
    - Your image will shrink very fast after some convolutions.
    - You overlap the edge pixels a lot.
- **Padding**: this lets you take care of the shrinking and overlapping problems. You're just adding pixels to the external edges, making the image bigger artificially without changing its core.
  - "Valid": without padding.
  - "Same": output has the same size of the original image.
    \[
    n + 2p - f + 1 = n \\
    p = \frac{f-1}{2}
    \]
    - $f$ is usually odd.
- **Strided Convolutions**: if $stride = 2$, instead of stepping 1 pixel at a time, we are going to make it 2 steps. The dimensions are now (ceil to avoid non integers):
  \[
  \Bigl\lfloor \frac{n+2p-f}{s} + 1 \Bigl\rfloor \times \Bigl\lfloor \frac{n+2p-f}{s} + 1 \Bigl\rfloor
  \]
- Mathematically, this convolution would actually be called **cross-correlation**. In math, there is a prior horizontal and vertical flipping.
- **Convolutions over Volumes**: instead of convolving planes, you're convolving cubes.
  \[
  M_{(6,6,3)} \ast W_{(3,3,3)} = Y_{(4,4,1)}
  \]
  - The output would have a third dimension to denote the amount of filters that processed the image: $(n-f+1, n-f+1, n^{\prime}_c)$, where $n^{\prime}_c$ is the number of filters in a given layer.
  - Finally, we add a bias and a non-linearity to configure an NN layer.
- Suppose we have 10 filters that are 3x3x3 (if someone says the filter has size fxf, it's actually, fxfx3, because it deals with RGB images) in one layer of an NN, how many parameters does that layer have?
  - $(27 + 1) \cdot 10 = 280$ parameters
- **Summary of Notation**:
  - $f^{[l]}$ is the filter size.
  - $p^{[l]}$ is the padding.
  - $s^{[l]}$ is the stride.
  - $n^{[l]}_C$ is the number of filters in a layer.
  - $(n^{[l-1]}_H, n^{[l-1]}_W, n^{[l-1]}_C)$ is the input's dimensions.
  - $(n^{[l]}_H, n^{[l]}_W, n^{[l]}_C)$ is the outputs' dimension, where
    \[
    n^{[l]}_H = \Bigl\lfloor \frac{n^{[l-1]}_H + 2p^{[l]} - f^{[l]}}{s^{[l]}} + 1 \Bigl\rfloor
    \]
  - $(f^{[l]}, f^{[l]}, n^{[l-1]}_c)$ is the filters' dimensions.
  - $(n^{[l]}_H, n^{[l]}_W, n^{[l]}_C)$ is the activations' dimensions $a^{[l]}$; and $(m, n^{[l]}_H, n^{[l]}_W, n^{[l]}_C)$ is the dimension of $A^{[l]}$.
  - $(f^{[l]}, f^{[l]}, n^{[l-1]}_C, n^{[l]}_C)$ is the dimension of the filters in each layer.
  - $(1, 1, 1, n^{[l]}_C)$ is the dimension of the biases.
- An example of CNN:
  \[
  \begin{align*}
    39 \times 39 \times 3 & \rightarrow 37 \times 37 \times 10 \rightarrow 17 \times 17 \times 20 \rightarrow 7 \times 7 \times 40 \\

    n^{[0]}_H = n^{[0]}_W = 39 & \rightarrow n^{[1]}_H = n^{[1]}_W = 39 \rightarrow n^{[2]}_H = n^{[2]}_W = 17 \rightarrow n^{[3]}_H = n^{[3]}_W = 7\\

    n^{[0]}_C = 3 & \rightarrow n^{[1]}_C = 10 \rightarrow n^{[2]}_C = 20 \rightarrow n^{[3]}_C = 40
  \end{align*}

  \]
  where
  \[
  \begin{align*}
    f^{[1]} = 3 & \rightarrow f^{[2]} = 20 \rightarrow f^{[3]} = 40 \\

    s^{[1]} = 1 & \rightarrow s^{[2]} = 2 \rightarrow s^{[3]} = 2 \\

    p^{[1]} = 0 & \rightarrow p^{[2]} = 0 \rightarrow p^{[3]} = 0 \\

    n^{[1]}_C = 10 & \rightarrow n^{[2]}_C = 20 \rightarrow n^{[3]}_C = 40
  \end{align*}
  \]
  - The trend with CNNs is to:
    \[
    \uparrow n^{[l]}_C \\
    \downarrow n^{[l]}_H, n^{[l]}_W
    \]
  - In the last layer, you flatten the $7 \times 7 \times 40$ tensor in a 1960 vector to input it to input it to a logits or softmax.
- **Pooling Layers**:
  - **Max Pooling**: for example, in a 4x4 matrix and 2x2 output, you take the max of squares of 2x2 in the input.
    - There are two hyperparameters: $f = 2$ and $s = 2$, which are not learnable. Most of the time you don't use padding as a hyperparameter.
    - This tries to keep the main feature of a block and still reduce the size of the input.
  - **Average Pooling**: just take the average inside a block. Usually this is only used sometimes, to compress the image.
  - There are no parameters to learn.
- The typical **scheme** for a CNN is to have something like:
  \[
  \begin{align*}
    Conv & \rightarrow Pool \rightarrow Conv \rightarrow Pool \rightarrow \\

    & \rightarrow Fully \ Connected \rightarrow FC \rightarrow Softmax
  \end{align*}
  \]
  - The activation size will drop gradually, if it is too fast, it can hurt performance.
  - The pooling layers have no parameters.
  - If there is no padding, the formulas binding the output shape of the pooling are:
    \[
    \begin{align*}
      n_H & = \Bigl\lfloor \frac{n_{H_{prev}} - f}{stride} \Bigl\rfloor + 1 \\

      n_W & = \Bigl\lfloor \frac{n_{W_{prev}} - f}{stride} \Bigl\rfloor + 1 \\

      n_c & = n_{C_{prev}}
    \end{align*}
    \]
- **Why Convolutions?**
  - If you wanted to use NNs directly at the first layers, the size would be unfeasibly large.
  - **Parameter Sharing**: a conv filter makes the pixels to share its parameters.
    - Additionally, it reduces the number of parameters, thus reducing overfitting.
  - **Sparse connections**: in each layer, each output depends only on a small number of inputs.
  - The Convolutional structure is very robust to shifts in the image also.


# Week 2

- **Why look at case studies?**
  - You can use examples to solve similar problems.
  - Some examples:
    - LeNet-5
    - AlexNet
    - VGG-16
    - ResNet
    - Inception
- **LeNet-5**, LeCun (1998): recognize handwritten digits.
  - 32x32x1 $\rightarrow$ f = 5, s = 1 $\rightarrow$ 28x28x6 $\rightarrow$ f = 2, s = 2, avg pool $\rightarrow$ 14x14x6 $\rightarrow$ f = 5, s = 1 $\rightarrow$ 10x10x16 $\rightarrow$ f = 2, s = 2, avg pool $\rightarrow$ 5x5x16 (400) $\rightarrow$ FC 120 $\rightarrow$ FC 84 $\rightarrow$ Different Classifier.
  - **Advanced Comments**: it had a non-linearity after the pooling layers. If you want to read it, focus on sections **II** and **III**.
  - ~60k parameters.
- **AlexNet**, Alex Krizhevsky and Geoff Hinton: first very successful image recognition.
  - 227x227x3 $\rightarrow$ f = 11, s = 4 $\rightarrow$ 55x55x96 $\rightarrow$ max pool f = 3, s = 2 $\rightarrow$ 27x27x96 $\rightarrow$ Conv f = 5 same $\rightarrow$ 27x27x256 $\rightarrow$ max pool f = 3, s = 2 $\rightarrow$ 13x13x256 $\rightarrow$ Conv f = 3 same $\rightarrow$ 13x13x384 $\rightarrow$ Conv Same  f = 3 $\rightarrow$ 13x13x384 $\rightarrow$ Conv Same f = 3 $\rightarrow$ 13x13x384 $\rightarrow$ max pool f = 3, s = 2 $\rightarrow$ 6x6x256 $\rightarrow$ 9216 FC $\rightarrow$ 4096 FC $\rightarrow$ 4096 FC $\rightarrow$ Softmax 1000.
  - Similar to Lenet, but much bigger $\rightarrow$ ~60M parameters.
  - Used ReLU.
  - Multiple GPUs.
  - Local Response Normalization: not used anymore. Normalizing across the volume.
  - Turning Point for Deep Learning.
- **VGG-16**: Simonyan and Zisserman (2015). Instead of having a bunch of hyperparameters, let's focus on the convolution layers.
  - 224x224x3 $\rightarrow$ 64 Conv x2 f = 3, s = 1, same $\rightarrow$ max pool f = 2, s = 2 $\rightarrow$ 224x224x64 $\rightarrow$ max pool 112x112x64 $\rightarrow$ 128 Conv x2 $\rightarrow$ 112x112x128 $\rightarrow$ max pool $\rightarrow$ 56x56x128 $\rightarrow$ Conv x3 $\rightarrow$ 56x56x256 $\rightarrow$ max pool $\rightarrow$ 28x28x256 $\rightarrow$ Conv x3 512 $\rightarrow$ 28x28x512 $\rightarrow$ max pool $\rightarrow$ 14x14x512 $\rightarrow$ Conv x3 512 $\rightarrow$ 14x14x512 $\rightarrow$ max pool $\rightarrow$ 7x7x512 $\rightarrow$ FC 4096 $\rightarrow$ FC 4096 $\rightarrow$ Softmax 1000.
  - ~138M parameters, big by modern standards, but very homogeneous.
  - VGG-19 is a slightly bigger version which is only slightly better.
- **ResNets**:
  - Very Deep Networks are difficult to train because of exploding and vanishing gradients.
  - **Residual Block**: In a two layer NN, instead of the last activation being $a^{[l+2]} = g(z^{[l+2]})$ (Plain Network), it's going to be $a^{[l+2]} = g(z^{[l+2]} + a^{[l]})$, which can be seen as a "shortcut" or "skip connection" to the initial activation.
  - If you make networks which are incredibly deep, you will find that the training error increases after a certain number of layers. If you use ResNets instead, you still get a somewhat smoothly monotonically descendent curve.
  - **Intuition**: if you use regularization, $W^{[l]}$ will be closer and closer to zero, so $g(z^{[l+2]} + a^{[l]}) = g(W^{[l+2]}a^{[l+1]} + b^{[l+2]} + a^{[l]}) = g(a^{[l]}) = a^{[l]}$. This way, making a deep NN more robust to damage to $w^{[l]}$ and $b^{[l]}$.
- **1x1 Convolution**: not quite multiplying by a scalar.
  - If you take into account that it is a 1x1xn_c convolution, it actually makes more sense.
  - "Network in Network", Lin (2013).
  - Adds more linearity.
  - Can convert a lot of channels into a smaller amount of channels: 28x28x192 $\rightarrow$ 28x28x32, for example.
