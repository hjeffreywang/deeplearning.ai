# Course 1

## Week 1

- Taking off of DL: performance + data
- **Cycle of Programming**: idea $\rightarrow$ code $\rightarrow$ experimenation

**Small training set regime**:
- No difference in the size of the NN
- Ability of the programmer

Advancements:
- ReLU makes GD run faster.

## Week 2

- Usually, we transform the images in $n \times n \times 3$ to vectors $x_{64 \times 64 \times 3,1}$
- $m$ is the number of examples, as in $m_{train}$ and $m_{test}$. Matrix $X_{n \times m}$,
- $Y_{1 \times m}$

#### Logistic Regression

- A very small neural network.
- Given $x$, we want $\hat{y} = P(y=1|x)$.
- Linear functions don't make sense for probabilities
- $\hat{y} = \sigma (w^T x + b)$, where $\sigma = \frac{1}{1 + e^{-z}}$
- In our notation, $w \in \mathbb{R}^{n_x}$ and $b \in \mathbb{R}$

Cost Function

- Don't use the normal loss function in logistic regression: $\mathcal{L}(\hat{y},y)=\frac{1}{2} (\hat{y} - y)^2$
- Use instead: $\mathcal{L}(\hat{y},y)= -(y \log\hat{y} + (1-y)\log(1-\hat{y}))$
- If $y = 1$: $\mathcal{L}(\hat{y},y)= -\log \hat{y}$, want $\hat{y}$ to be large.
- If $y = 0$: $\mathcal{L}(\hat{y},y)= -\log (1 - \hat{y})$, want $\hat{y}$ to be small.
- Cost Function: $J(w,b) = \frac{1}{m} \sum^{m}_{i=1} \mathcal{L} (\hat{y}^{(i)},y^{(i)})$
- **Difference between the Functions**: The loss function computes the error for a single training example; the cost function is the average of the loss functions of the entire training set.

#### Gradient Descent

- Algorithm:
    \[
    \begin{align*}
      &Repeat: \\

      &w := w - \alpha \frac{\partial (w,b)}{\partial w} \\

      &(w := w - \alpha dw) \\

      &(b := b - \alpha db)
    \end{align*}
    \]
- For the **computation graph**, check the iPad. Computing derivatives is done with backward computation.
- In Python, we will use:
  \[
  dvar \equiv \frac{dFinalOutputVar}{dvar}
  \]
  For example:
  \[
  da \equiv \frac{dJ}{da} = \frac{dJ}{dv} \cdot \frac{dJ}{dv}
  \]

#### Logistic Regression with GD

\[
\begin{align*}
  x_1,w_1;x_2,w_2;...;b &\rightarrow z = w_1 x_1 + w_2 x_2 + ... + b \rightarrow \\

  &\rightarrow \hat{y} = a = \sigma(z) \rightarrow \mathcal{L}(a,y)
\end{align*}
\]
\[
\begin{align*}
  &da \equiv \frac{d \mathcal (a,y)}{da} = -\frac{y}{a} + \frac{1-y}{1-a} \\

  &dz \equiv \frac{d \mathcal{L}}{dz} = \frac{d\mathcal{L}}{da} \frac{da}{dz} = a-y \\

  &\frac{\partial \mathcal{L}}{\partial w_1} = x_1 dz \equiv dw_1 \\

  &\frac{\partial \mathcal{L}}{\partial w_2} = x_2 dz \equiv dw_2 \\

  &\frac{\partial \mathcal{L}}{\partial b} = dz \equiv db
\end{align*}
\]

- Update equations
  ``` {r,eval=FALSE} # for pseudocoding
  w_1 := w_1 - alpha*dw1
  ...
  b := b - alpha*db
  ```
- Algorithm with $m$ examples (non-vectorized, takes 200-300x more):
  ```
  J = 0; dw1 = 0; dw2 = 0; db = 0
  for i = 1 to m
    z[i] = wTx[i] + b
    a[i] = sigmoid(z[i])
    J += -(y[i]log(a[i]) + (1-y[i])log(1-a[i]))
    dz[i] = a[i] - y[i]
    dw1 += x1[i]dz[i]
    dw2 += x2[i]dz[i]
    db += dz[i]

  J /= m; dw1 /= m; dw2 /= m; db /= m
  w1 := w1 - alpha*dw1
  w2 := w2 - alpha*dw2
  b := b - alpha*db
  ```
- Vectorized Version:
  - `np.dot()` uses parallel instructions to make the code faster (SIMD). Both the CPU and the GPU have this parallelism, but GPUs are way better.
  - You can do transposing with `w.T`.
  - Vectorized Code:
    ``` python
    for i in range(1000):
      dw = np.zeros(n_x,1)
      Z = np.dot(w.T,X) + b
      # if b is ill-defined,
      # Python will broadcast it.
      A = sigmoid(z)
      dz = A - Y
      dw = (1/m)*X*dz.T
      db = (1/m)*np.sum(dz)
      w = w - alpha*dw
      b = b - alpha*db
    ```
  - **Main Loop of GD**:
    1. Calculate the Current Loss (Forwardpropagation)
    1. Calculate the Gradient (Backpropagation)
    1. Update Parameters (Gradient Descent)
- **Broadcasting**: Python will match the dimensions of different matrices (similar to Matlab/Octave `bsxfun`). Example:
  ``` python
  cal = A.sum(axis=0) # Sum over columns
  percentage = 100*A/(cal.reshape(1,4))
  ```
- **Dont's of Python**:
  - Don't use **rank 1 arrays**, like `np.random.randn(5)`, specify the dimensions: `np.random.randn(5,1)`. Otherwise strange results for matrix operations will occur.
  - **Assertion Statements** will help you make sure the matrices have the intended dimensions (`assert(a.shape == (5,1))`). These are extremely useful to find bugs.
- You can reshape an image simply with `x = img.reshape((32*32*3,1))`
- If you add **restarts** to the Logistic Regression NN, accuracy gets a slight increase from 70% to 74%.

#### Jupyter Notebooks

- **Python Basics**:
  - Reshaping Image Tensors with `.reshape()`
  - Normalizing Matrices with `np.linalg.norm()`
  - Coded softmax: $\frac{e^{x_{ij}}}{\sum_{j=0}^{j=N} e^{x_{ij}}}$
  - `*` or `np.multiply()` performs elementwise multiplication.
- **Logistic Regression with NN**:
  - Get the dimensions of the images in the data set and **reshape** it to vectors
  - Usually, people center and divide by the std. But for images, it's more convenient to divide everything by the max, 255 (**standardize**).
  - Next...
  - **Initialize the parameters** of the model
  - **Learn the parameters** by minimizing cost.

## Week 3

- **Notation**: $z^{[1]} = W^{[1]}x + b^{[1]}$ where the brackets refer to the NN layer.
- **Hidden Layer** refers to parameters which are not seen in the training set, i.e., different from the input and output parameters. The superscript starts at 0 (and lowerscript is the node in the layer), $a^{[0]}_{i}$ for the input and so on.
- Transform it into a matrix: $(w^{[l]}_i)^T$ (rows of the weights for each node inside the layer). If you have $x_{3x1}$, then we have $W^{[1]} W_{4x3}$ (if we also have 4 nodes in the hidden layer).
- The second layer will have $z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$
- $a^{[1](1)}$ means the first layer and first example. Horizontally, different examples, and vertically, hidden units.
- **Vectorized Implementation of Forward Prop**:
  \[
  \begin{align*}
    &Z^{[1]} = W^{[1]}X + b^{[1]} \\

    &A^{[1]} = \sigma(Z^{[1]}) \\

    &Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]} \\

    &A^{[2]} = \sigma(Z^{[2]})
  \end {align*}
  \]
  - The dimensions shouls be of the form: $W^{[n]} \rightarrow (n^{[n]},n^{[n-1]})$, and then $Z^{[n]} \rightarrow (n^{[n]},m)$
- **Vectorized Implementation of GD**:
  \[
  \begin{align*}
    dZ^{[2]} &= A^{[2]} - Y \\

    dW^{[2]} &= \frac{1}{m}dZ^{[2]}A^{[1] T} \\

    db^{[2]} &= \frac{1}{m} np.sum(dZ^{[2]},axis = 1,keepdims = True) \\

    dZ^{[1]} &= W^{[2] T} dZ^{[2]} \ast g^{[1] \prime} (Z^{[1]}) \\

    dW^{[1]} &= \frac{1}{m}dZ^{[1]}X^T \\

    db^{[1]} &= \frac{1}{m} np.sum(dZ^{[1]},axis = 1,keepdims = True)
  \end{align*}
  \]
  - $dZ^{[2]}$ comes from the fact that the last layer uses the *sigmoid* funcion.
  - If $tanh$ is the activation function, then $g^{[1] \prime}(z) = 1 - a^2$, which, in Python, will be $g^{[1] \prime}(Z^{[1]}) =$ `(1 - np.power(A1,2))`
- **Activation Function**: $tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ almost always is better than the sigmoid, because it centers the data better. The **only exception** is the output when with binary classification. The **most popular function**, though is ReLU: $a = max(0,z)$. There is also the **leaky ReLU**, $a = max(0.01z,z)$ which sometimes works even better.
- **Why do we need a nonlinear activation function?** If you don't have any linearity, all you're doing is a linear operation. So you might as well not have any layers. The **only case** you use linear activation function, is in the output layer when $y \in \mathbb{R}$, like housing prices.
- Derivatives:
  \[
  \begin{align*}
    \frac{d \sigma(z)}{dz} &= \sigma(z) (1 - \sigma(z)) \\

    \frac{d tanh(z)}{dz} &= 1 - (tanh(z))^2 \\

    \frac{d ReLU(z)}{dz} &= \begin{cases} 0, \ if \ z < 0 \\  1, \ if \ z \geq 0\end{cases} \\
  \end{align*}
  \]
- **Initialize the Weights Randomly**: if you initialize with zeros, the examples will have the same back and forward prop. It's ok to initialize `b` with zeros (though better with random values). Initialize the weights with small values though: `w = np.random.randn((2,2))*0.01`. If you have deeper NN, you should choose other constant rather than `0.01`.
  - You choose a small constant so numbers cannot be too large, which would make the `tanh` activation function to saturate, slowing down the gradient process.
- In the programming assignment for Week 3, with more than 20 units, some weird non-linearities appear, which is kind of good.
  - Many Units NNs start to overfit the data. You would need **regularization** to diminish that.

## Week 4

- **Notation**: `L` for the number of layers; $a^{[l]}$ where $l$ goes from `0` to  `L`.
  - **Number of Layers**: Input and Output layers are not counted as hidden layers. The number of layers is equal to the number of hidden layers + 1.
- **General Equations of Deep NN**:
  \[
  \begin{align*}
    Repeat& \ (Forward \ Prop): \\

    Z^{[l]} &= W^{[l]}A^{[l-1]} + b^{[l]} \\

    A^{[l]} &= g^{[l]}(Z^{[l]}) \\

    Repeat& \ (Back \ Prop): \\

    dZ^{[l]} &= dA^{[l]} \ast g^{[l] \prime} (Z^{[l]}) \\

    dW^{[l]} &= \frac{1}{m} dZ^{[l]} A^{[l-1]T} \\

    db^{[l]} &= \frac{1}{m} np.sum(dZ^{[l]},axis=1,keepdims=True) \\

    dA^{[l-1]} &= W^{[l]T} dZ^{[l]}
  \end{align*}
  \]
  - The **dimensions** still stand: $Z^{[l]}, A^{[l]} \in \mathbb{R}^{n^{[l]}, m}$ and $W^{[l]} \in \mathbb{R}^{n^{[l]}, n^{[l-1]}}$. The dimensions of the derivatives should be consistent to the original values as well.
  - For the backprop, the first and fourth equation are a more succint way of writing $dZ^{[l]} = W^{[l+1]T} dZ^{[l+1]} \ast g^{[l] \prime} (Z^{[l]})$
- **Intuition** of Deep NN:
  - Layers detect different characteristics, with increasing complexity.
  - Circuit Theory of DL: informally, there are functions you can compute with a "small" L-layer deep neural network that shallower networks require exponentially more hidden units to compute. E.g., XOR tree $\rightarrow$ $O(\log n)$; while with one layer $\rightarrow$ $O(2^n)$.
- **Hyperparameters**: learning rate; iterations; hidden layers; hidden units; choice of activation functions; momentum; minibatch size; regularization. They control the internal parameters, like weights and individual biases.
  - It's an empirical process: $Idea \rightarrow Code \rightarrow Experiment \rightarrow Idea$.
- In the last section of part 2 of Week 4 Assignment, you can have an idea of how to deal with images of different size. They use a function called `scipy.misc.imresize()`.
  - There is apparently a mistake in that session. You should divide your image by 255 to normalize it, otherwise the algorithm will think it's all white pixels.
  - The system still overfits a lot though, the difference is of almost 20%, which is unacceptable. It can't work properly with external images.
