# TensorFlow Tutorial

## Exploring the TensorFlow Library

### Initalizing Variables

  ``` python
  y_hat = tf.constant(36, name='y_hat')
  y = tf.constant(39, name='y')

  loss = tf.Variable((y - y_hat)**2, name='loss')

  init = tf.global_variables_initializer()

  with tf.Session() as session:
    session.run(init)
    print(session.run(loss))
  ```

### Basic Sessions

  ``` python
  a = tf.constant(2)
  b = tf.constant(10)
  c = tf.multiply(a,b)
  print(c)
  # All you did was put in a 'computation graph'
  # You need to initialize and
  # run a session to get the result
  sess = tf.Session()
  print(sess.run(c))
  ```

### Placeholders

  ``` python
  # A placeholder is an object whose value
  # you can specify only later.
  # Change the value of x in the feed_dict
  x = tf.placeholder(tf.int64, name = 'x')
  print(sess.run(2 * x, feed_dict = {x: 3}))
  sess.close()
  ```

### Linear Function

  ``` python
  X = tf.constant(np.random.randn(3,1), name = 'X')
  W = tf.constant(np.random.randn(4,3), name = 'W')
  b = tf.constant(np.random.randn(4,1), name = 'b')
  Y = tf.add(tf.matmul(W,X), b)

  sess = tf.Session()
  result = sess.run(Y)
  sess.close()
  ```

### Computing the Sigmoid

  ``` python
  x = tf.placeholder(tf.float32, name = 'x')
  sigmoid = tf.sigmoid(x)

  with tf.Session() as sess:
    result = sess.run(sigmoid, feed_dict = {x: z})
  ```

### Computing the Cost

  ``` python
  z = tf.placeholder(tf.float32, name = 'z')
  y = tf.placeholder(tf.float32, name = 'y')

  cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = y)

  sess = tf.Session()
  cost = sess.run(cost, feed_dict = {z: logits, y: labels})
  sess.close()
  ```

### Using One Hot Encodings

  - Becareful with the `axis` in the `tf.one_hot` function. For row vectors with the classes, do `axis = 0`
  ``` python
  C = tf.constant(C, name = 'C')

  one_hot_matrix = tf.one_hot(labels, depth = C, axis = 0)

  sess = tf.Session()
  one_hot = sess.run(one_hot_matrix, feed_dict = {x: labels})
  sess.close()
  ```

### Initialize with Zeros and Ones

  ``` python
  ones = tf.ones(shape)
  ```

### Building an NN

#### Create Placeholders

  ``` python
  X = tf.placeholder(tf.float32, shape = [n_x, None], name = 'X')
  Y = tf.placeholder(tf.float32, shape = [n_y, None], name = 'Y')
  ```

#### Initialize Parameters

  ``` python
  tf.set_random_seed(1)

  W1 = tf.get_variable("W1", [25,12288], \
                       initializer = tf.contrib.layers.xavier_initializer(seed = 1))
  b1 = tf.get_variable("b1", [25,1], \
                       initializer = tf.zeros_initializer())
  W2 = tf.get_variable("W2", [12,25], \
                       initializer = tf.contrib.layers.xavier_initializer(seed = 1))
  b2 = tf.get_variable("b2", [12,1], \
                       initializer = tf.zeros_initializer())
  W3 = tf.get_variable("W3", [6,12], \
                       initializer = tf.contrib.layers.xavier_initializer(seed = 1))
  b3 = tf.get_variable("b3", [6,1], \
                       initializer = tf.zeros_initializer())
  ```

#### Forward Prop

  ``` python
  Z1 = tf.add(tf.matmul(W1,X),b1)                
  A1 = tf.nn.relu(Z1)
  Z2 = tf.add(tf.matmul(W2,A1),b2)   
  A2 = tf.nn.relu(Z2)
  Z3 = tf.add(tf.matmul(W3,A2),b3)

  return Z3
  ```

  - This doesn't output any cache, you will soon discover why.

#### Compute Cost

  ``` python
  # Fit TF requirement for the shapes,
  # it should be (number of examples, num_classes)
  logits = tf.transpose(Z3)
  labels = tf.transpose(Y)

  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

  ```
