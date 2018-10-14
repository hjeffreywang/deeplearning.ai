# Course 2: DL Optimization

## Week 1

- **Experimentation is king**.
- Training follows the sets:
  1. Training Set.
  2. Dev Set (or Hold Out or Cross Validation).
  3. Test Set.
- The **proportions** would be 80/20 or 60/20/20.
- But with Big Data, you need a smaller percentage for the Dev and Test sets.
- Make sure the Dev and Test sets come from the same distribution. Web pics are different from Smartphone pics, for example.
- It might be ok to not have a test set.
- Bias and Variance
  - **High Bias**: Underfitting
  - **High Variance**: Overfitting
- **Basic Recipe**:
  - If **High Bias**:
    - Bigger network
    - Train set larger
    - Other NN architectures
  - If **High Variance**:
    - More data
    - Regularization
    - Other NN architecture
- Deep Learning kind of ended the Bias Variance Trade-off paradigm.
- **Random Initialization**:
  - If you initialize everything to 0s, it will fail to break symmetry.
  - You have to initialize randomly, though it's ok to initialize the bias with zeros.
- **Regularization**:
  - In Logistic Regression (usually the term with $b$ is not used):
    \[
    \begin{align*}
      J(w,b) &= \frac{1}{m} \sum^{m}_{i=1} \mathcal{L}(\hat{y}^{(i)},\hat{y}(i)) + \frac{\lambda}{2m} ||w||^2_2 + \frac{\lambda}{2m}b^2 \\

      L_2 \ Reg: \ &||w||^2_2 = \sum_{j=1}^{n_x} w^2_j = w^Tw \\

      L_1 \ Reg: \ &\frac{\lambda}{2m} \sum^{n_x}_{i=1}|w| = \frac{\lambda}{2m} ||w||_1
    \end{align*}
    \]
    - Note that new terms are now added to forward and back prop.
    - In $L_1$ regularization, $w$ will be sparse, which might be helpful for compression, but only helps a little bit.
    - $\lambda$ is the regularization parameter. In Python, use `lambd` because there is already a function called `lambda`.
    - This is actually called **weight decay**, because:
      \[
      \begin{align*}
        dw^{[l]} &= (...) + \frac{\lambda}{m}w^{[l]} \\

        w^{[l]} &= w^{[l]} - \alpha \frac{\lambda}{m} w^{[l]} - \alpha \cdot backprop \\

        w^{[l]} &= (1 - \alpha \frac{\lambda}{m})w^{[l]} - \alpha \cdot backprop
      \end{align*}
      \]
    - How does it **prevent overfitting**? $\frac{\lambda}{2m} \sum^{L}_{l=1} ||w^{[l]}||^2_F$ penalizes $w^{[l]}$ from being too large. If $w^{[l]} \approx 0$, the activation function goes to the linear region.
    - Frobenius Norm: $||w||^{2}_2 = \sum^{a^{[l-1]}}_{i=1} \sum_{j=1}^{n^{[l]}} (w_{ij}^{[l]})^2$
- **Dropout Regularization**: Eliminating randomly some nodes to simplify the network and reduce overfitting. They disappear in both forward and back prop.
  - Implementation (**Inverted Dropout**):
    ```python
    # Illustration with layer = 3
    keep_prob = 0.8
    D3 = np.random.rand(A3.shape[0], A3.shape[1]) < keep_prob
    A3 = np.multiply(A3, D3) # A3 *= D3
    A3 /= keep_prob
    ```
  - By dividing `A3 /= keep_prob` we make sure the expected value remains the same.
  - **Test time is smaller**, so no drop out on test time.
  - **Do not** keep the `1/keep_prob` factor in the calculations during training for test time.
  - **Why does drop out work?**
    - Intuition: A unit can't rely on any feature, so you have to spread the weights on every iteration, shrinking the squared norm of the weights.
  - A side effect of dropout is the **J gets less well-defined**.
- **Data Augmentation**:
  - Mirroring
  - Zooming
- **Early Stopping**: stopping at a training iteration where the dev and train errors are close to each other. **Downside**: couples GD and not overfitting, which yields less orthogonalization.
- **Normalizing Inputs**:
  \[
  \begin{align*}
    \mu &= \frac{1}{m} \sum^{m}_{i=1} x^{(i)} \\

    \sigma^2 &= \frac{1}{m} \sum^{m}_{i=1} (x^{(i)} - \mu)^2
  \end{align*}
  \]
  - You should use the same $\mu$ and $\sigma^2$ on the test set also.
  - **Why?** Because features with very different ranges will yield non-symmetric cost functions, which are more difficult to work with.
- **Vanishing/Exploding Gradients**:
  - Suppose a linear activation: if $W > \mathbb{I}$; $W^L$ will explode; if $\mathbb{0} < W < \mathbb{I}$; $W^L$ vanishes.
  - **Partial Solution**: the larger the $n$, the smaller the $w_i$.
    \[
    \begin{align*}
      ReLU: \ &var(w_i) = \frac{2}{n} \\

      tanh: \ &var(w_i) = \frac{1}{n^{[l-1]}} \ (Xavier \ Init)\\

      other: \ &var(w_i) =  \frac{2}{n^{[l-1]} + n^{[l]}} \\

      w^{[l]} = np.&random.rand(shape)*np.sqrt(\frac{2}{n^{[l-1]}})
    \end{align*}
    \]
  - He Initialization (2015): $var(w_i) = \frac{2}{n^{[l-1]}}$
- **Numerical Approximation of Gradients**: the "big triangle" definition is better because it gets $O(\epsilon^2)$
  \[
  \begin{align*}
    f^{'}(\theta) &= \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2 \epsilon} \Rightarrow O(\epsilon^2) \\

    f^{'}(\theta) &= \frac{f(\theta + \epsilon) - f(\theta)}{\epsilon} \Rightarrow O(\epsilon)
  \end{align*}
  \]
- **Gradient Checking**:
  1. $\epsilon \approx 10^{-7}$
  1. Take $W^{[1]},b^{[1]},...,W^{[L]},b^{[L]}$ and reshape it into a big vector $\theta$.
    \[
    J(W^{[1]},b^{[1]},...,W^{[L]},b^{[L]}) = J(\theta)
    \]
  1. Do the same for the derivations $d\theta$.
  1. Is $d\theta$ the slope of $J(\theta)$?
  1. for each $i$ in $\theta$:
    \[
    \begin{align*}
      d\theta_{approx}[i] &= \frac{J(\theta_1,\theta_2,...,\theta_i + \epsilon,...) - J(\theta_1,\theta_2,...,\theta_i - \epsilon,...)}{2 \epsilon} \\

      &\approx d\theta[i] = \frac{\partial J}{\partial \theta_i} \\

      d\theta_{approx} \approx d\theta \ ? &\rightarrow Check \ \frac{||d\theta_{approx} - d\theta||_2}{||d\theta_{approx}||_2 + ||d\theta||_2}
        \begin{cases}
          10^{-7} \rightarrow Great! \\

          10^{-5} \rightarrow Check \ Again \\

          10^{-3} \rightarrow Worry
        \end{cases}
    \end{align*}
    \]
  - Don't use it during training, only to debug, otherwise, it will be too slow.
  - If the algorithm fails $\rightarrow$ look at the components to try to identify the bugs.
  - Remember Regularization.
  - It doesn't work properly with dropout, because $J$ will be ill-defined.
  - Run at random initialization; perhaps again after some training.

## Week 2

- **Batch** vs **Mini Batch**: instead of letting the algorithm progress only when you process the entire data set, you separate it in batches. **Notation**: $X^{\{1\}} = (x^{(1)} x^{(2)} ... x^{(B)})$. **Epoch** is going through the entire data set.
  - Suppose 5,000,000 examples, in 5,000 batches:
    ```
    for t = 1,..., 5000
      forward prop on X_t
        Z1 = W1*X_t + b1
        A1 = g1(Z1)
        ...
        AL = gL(ZL)
      compute Cost J = (1/1000)*sum(Costs) + Regularization
      back prop on J_t
      update weights
    ```
  - Progress x Iteration **might not go down smoothly**, it will be noisier, it will trend downwards.
  - **Size**:
    1. Size = m: Batch Gradient Descent
    1. Size = 1: **Stochastic Gradient Descent**. Extremely Noisy. Won't ever Converge, it will oscillate around the minimum. It loses speed from a lack of vectorization.
    1. In between size: in practice, the best.
    - How to choose:
      1. If small, just use BGD. ($m \leq 2000$)
      1. Typical mini-batch sizes: 64, 128, 256, 512... (powers of 2)
      1. Try not to use a size bigger than GPU or CPU memory.
- **Exponentially Weighted Averages** (Moving Averages), e.g., $V_t = \beta V_{t-1} + (1-\beta) \theta_t$.
  - $V_t$ averages over $\approx \frac{1}{1-\beta}$ (how long it takes for the weights drop to $\frac{1}{e} \approx 0.35$).
    - If $\beta = 0.9 \Rightarrow Average \frac{1}{1-0.9} \approx 10 days$.
    - If $\beta = 0.98 \Rightarrow Average \approx 50 days$. More latency, the graph is smoother and, in general, it underfits the data.
    - If $\beta = 0.5 \Rightarrow Average \approx 2 days$ which gives us an overfit.
  - Example:
    \[
    V_{100} = 0.1 \theta_{100} + 0.1 \cdot 0.9 \cdot \theta_{99} + 0.1 \cdot 0.9^2 \theta_{98} + 0.1 \cdot 0.9^3 \theta^{97} + ...
    \]
  - **Bias Correction**: if the initial bias is 0, the curve will start really low. Instead, you can use: $V_t = \frac{\beta V_{t-1} + (1-\beta) \theta_t}{1 - \beta^t}$, which will make the first values bigger.
    - In Deep Learning, people usually don't bother with bias correction because it's only an initial deviance.
- Gradient Descent with **Momentum**: you want to make it faster when it moves in the direction of the minimum but slower when it oscillates perpendicularly.
  - Momentum:
    ```
    On iteration t:
      Compute dW, db on current mini-batch
      V_dW = beta*V_dW + (1-beta)*dW # Moving Average
      V_db = beta*V_db + (1-beta)*db

      w -= alpha*V_dW
      b -= alpha*V_db
    ```
  - **Why does it work?** Vertical oscillations will be averaged out, while the horizontal values will be accelerated, like a ball rolling down a bowl.
  - $\beta = 0.9$ is usually a robust value.
  - Sometimes you get in the literature: $V_{dW} = \beta V_{dW} + V_{dW}$, which also work, but is less intuitive.
- **RMSprop**:
  ```
  On iteration t:
    Compute dW, dB on current mini-batch
    S_dW = beta*S_dW + (1-beta)*dW^2
    S_dW = beta*S_dW + (1-beta)*db^2

    w -= alpha*dW/sqrt(S_dW)
    b -= alpha*db/sqrt(S_db)
  ```
  - $\frac{1}{\sqrt{S_{db}}}$ helps to dampen the oscilations, which also makes us able to use a higher learning rate.
  - In practice, you add an $\epsilon$ to get $\frac{1}{\sqrt{S_{db}} + \epsilon}$
  - RMSprop was first proposed in a Coursera course lectured by Geoff Hinton.
- **ADAM** (Adaptive Moment Estimation): basically taking Momentum and RMSprop and putting them together.
  ```
  V_dW = 0, S_dW = 0, V_db = 0, S_db = 0
  On iteration t:
    Compute dW, db using current mini-batch
    V_dW = beta1*V_dW + (1-beta1)*dW
    V_db = beta1*V_db + (1-beta1)*db
    S_dW = beta2*S_dW + (1-beta2)*dW^2
    S_db = beta2*S_db + (1-beta2)*db^2

    # In the typical implementation of ADAM,
    # you do bias correction.

    V_dW /= 1 - beta1^t
    V_db /= 1 - beta1^t
    S_dW /= 1 - beta2^t
    S_db /= 1 - beta2^t

    w -= alpha*(V_dW/(sqrt(S_dW) + eps))
    b -= alpha*(V_db/(sqrt(S_db) + eps))
  ```
  - $\alpha$ still needs to be tuned. Recommended:
    - $\beta_1 = 0.9$
    - $\beta_2 = 0.999$
    - $\epsilon = 10^{-8}$
- **Learning Rate Decay**: the closer you get to the minimum, the slower you go.
  - $\alpha = \frac{\alpha_0}{1 + decay.rate*epoch.num}$
  - There is also exponential decay: $\alpha = 0.95^{epoch.num} \alpha_0$
  - $\alpha = \frac{k}{\sqrt{epoch.num}} \alpha_0$
  - $\alpha = \frac{k}{\sqrt{t}} \alpha_0$
  - There is also the discrete staircase.
  - If you're model takes a lot of time, you can do manual decay.
- **The Problem of Local Optima**: for you to end up in a local optimum, you need all directions to be pointing upwards, which, in many dimensions, is very unlikely. Say we have 10 dimensions, then the probability of that happening would be $10^{-10}$. Instead, what we usually have are **saddle points**, where not all directions have a zero derivative. This makes local optima much less of a problem.
- **The Problem of Plateaus**: the algorithm makes learning very slow.

## Week 3

- **Tuning Process**:$\alpha, \beta, \epsilon, \#layers, \#hidden units, learning \ rate decay, mini-batch \ size$.
  - Most Important: $\alpha$
  - Second: $\beta, \#hidden units, mini-batch size$
  - Third: $\#layers, \#learning \ rate \ decay$
- Try random values: **don't use grid**.
- Coarse to fine, sampling more densely around the best values found randomly at first.
- For $\alpha$, it's much more appropriate to search for parameters in a **$\log$ scale** since it has smaller values. For example:
  ``` python
  r = -4*np.random.rand() # r in [-4,0]
  alpha = 10**r
  ```
  - Similarly, for $\beta$:
    ``` python
    r = -3*np.random.rand()
    beta = 1 - 10**r # r in [-3,-1]
    ```
- You should re-test hyperparameters occasionally.
- Two major approaches for tuning models:
  - Babysitting one model (Pandas)
  - Training many models at the same time. (Caviar)
- **Batch Normalization** (BN): instead of normalizing only the inputs ($Z$ instead of $A$) for the first layer, we do it for every layer.
  \[
  \begin{align*}
    &Given \ an \ intermediate \ values \ in \ NN \ Z^{[l](1)},...,Z^{[l](m)} \\

    &\mu = \frac{1}{m} \sum_i z^{(i)} \\

    &\sigma^2 = \frac{1}{m} \sum_i (z^{(i)} - \mu)^2 \\

    &z^{(i)}_{norm} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}} \\ \\

    &\hat{z}^{(i)} = \gamma z^{(i)}_{norm} + \beta \\

    &If
      \begin{cases}
        \gamma = \sqrt{\sigma^2 + \epsilon} \\ \beta = \mu
      \end{cases}
    \Rightarrow \hat{z}^{(i)} = z^{(i)}
  \end{align*}
  \]
  - $\gamma$ and $\beta$ are learnable parameters of the model, and they make us able to not cluster the values around the linear part of the sigmoid.
  - Parameters of the NN now: $W^{[1]},b^{[1]},..., W^{[L]},b^{[L]},\beta^{[1]},\gamma^{[1]},...,\beta^{[L]},\gamma^{[L]}$. (Not the same $\beta$ as in ADAM.)
  - Use GD to update $\beta$ and $\gamma$, like $\beta^{[l]} = \beta^{[l]} - \alpha d\beta^{[l]}$.
  - Batch Norm is usually used with mini-batches.
  - If you use Batch Norm, since you're always normalizing with zero mean, you can basically **eliminate $b^{[l]}$**.
  - Implementation:
    ```
    for t=1 ... num Mini Batches
      Compute Forward Prop on X_t
        In each hidden layer, use BN to rebuild z_l with z_tilda_l
      Use Back Prop to compute dW_l, dbeta_l, dgamma_l
      Update Parameters
        W_l = W_l - alpha*dW_l
        beta_l = beta_l - alpha*dbeta_l
        gamma_l = gamma_l - alpha*dgamma_l
    ```
  - BN makes the later layers more robust to changes in the first layers.
    - **Covariate Shift**: even if the ground function remains the same, if the distribution of your data changes, you might need to retrain your model. BN makes the model more robust to those changes.
    - By normalizing everything, BN makes sure that, at least, $\mu$ and $\sigma$ stay the same.
    - BN also has a *slight* **regularization** effect. The $z^{[l]} \rightarrow \hat{z}^{[l]}$ adds some noise (also because of the size of the mini-batch) to the calculations, which also helps the model to regularize itself, much like dropout.
    - At **test time**, you will need to adapt BN, because you're probably going to process one example at a time.
      - One try is to estimate an exponentially weighted average for each layer's $\mu$ and $\sigma$.
- **Softmax Regression**: multiclass classification. Capital `C` for the #classes.
  \[
  \begin{align*}
    &Z^{[L]} = W^{[L]} A^{[L]} + b^{[L]} \\

    &t = e^{z^{[L]}} \\

    &a^{[L]} = \frac{e^{Z^{[L]}}}{\sum_{j=1}^{C} t_i} \\

    &a^{[L]}_i = \frac{t_i}{\sum_{j=1}^{C} t_i}
  \end{align*}
  \]
  - $\sum_{j=1}^{C} t_i$ makes sure that they sum up to 1. And the exponentiation is simulating a multiclass logistic regression.
  - **Exponentiation** also makes one of the classes to stand out, since one index will cause a huge difference in the probability.
  - "Softmax" comes from a contrast to "hard max", where we force zeros and ones.
  - If $C = 2$, softmax is reduced to logistic regression.
  - **Back Prop for Softmax** (derivable):
    \[
    \frac{\partial J}{\partial z^{[L]}} = dz^{[L]} = \hat{y} - y
    \]
- **Deep Learning Frameworks**:
  - Criteria to choose:
    - Ease of programming
    - Running speed
    - Truly open (open source with good governance)
- **TensorFlow**:
  - **Motivating Problem**: $J(w) = w^2 - 10w + 25 = (w-5)^2$
    ``` python
    import numpy as np
    import tensorflow as tf

    w = tf. Variable(0,dtype=tf.float32)
    # cost = tf.add(tf.add(w**2,tf.multiply(-10,w)),25)
    cost = w**2 + 10*w + 25
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)
    print(session.run(w))
    # prints 0.0, because we haven't done anything so far

    session.run(train)
    print(session.run(w))
    # prints 0.1 after 1 iteration

    for i in range(1000):
      session.run(train)
    print(session.run(w))
    # prints 4.99999, which is very close to the minimum

    ```
  - What if you want to minimize a function of your training data?
    ``` python
    coefficients = np.array([[1.], [-20.], [100.]])

    w = tf. Variable(0,dtype=tf.float32)
    x = tf.placeholder(tf.float32, [3,1])
    cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]
    # You could replace GDOptimizer for something like ADAM
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    session.run(train, feed_dict=[x:coefficients])

    for i in range(1000):
      # You could change `feed_dict` to take mini-batches
      session.run(train, feed_dict=[x:coefficients])
    print(session.run(w))
    ```
  - Alternative idiomatic expression for the `session` part:
    ``` python
    with tf.Session() as session:
      session.run(init)
      print(session.run(w))
    ```
