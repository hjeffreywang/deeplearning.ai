# Sequence Models

## Week 1

- **Why Sequence Models?** Ex.: speech recognition, music generation, sentiment classification, DNA analysis, machine translation, video activity recognition, name entity recognition.
- **Notation**:
  - x: Harry Potter and Hermione Granger invented a new spell.
  - x: $x^{\langle 1 \rangle} x^{\langle 2 \rangle} ... x^{\langle t \rangle} ... x^{\langle 9 \rangle}$
  - y: 1 1 0 1 1 0 0 0 0 (people's names in a sentence)
  - y: $y^{\langle 1 \rangle} y^{\langle 2 \rangle} ... y^{\langle t \rangle} ... y^{\langle 9 \rangle}$
  - $t$th example of a sequence: $x^{(i) \langle t \rangle}$
  - Length of the input sequence of training example $i$: $T^{i}_x$
  - Vocabulary (small dictionary):
    \[
    \begin{bmatrix}
      a \\
      aaron \\
      \vdots \\
      and \\
      \vdots \\
      harry \\
      \vdots \\
      potter \\
      \vdots \\
      zulu
    \end{bmatrix}

    \Rightarrow
    Index =

    \begin{bmatrix}
      1 \\
      2 \\
      \vdots \\
      367 \\
      \vdots \\
      4,075 \\
      \vdots \\
      6,830 \\
      \vdots \\
      10,000
    \end{bmatrix}
    \]
    - Use **one-hot** representations of the dictionary to represent the words in the phrase.
    - **Unknown Word**: $x^{\langle unk \rangle}$
- **Recurrent Networks**:
  - **Why not a standard network?**:
    - Inputs and outputs can be of different length.
    - Doesn't share features learned across different positions of text.
  - The activation of the first time step $a^{\langle 1 \rangle}$ gets passed to the second time step.
  - **Weights**: in $w_{ax}$, a $w$ quantity is going to be multiplied by an $x$-like quantity to compute an $a$-like quantity.
    - From $x$ to the network: $w_{ax}$.
    - From the network to the next time step: $w_{aa}$
    - From the network to $\hat{y}$: $w_{ya}$
  - One **downside** about this architecture is that it isn't using information about the next words.
    - He said, "Teddy Roosevelt was a great President."
    - He said, "Teddy bears are on sale!"
  - To correct for that, use Bidirectional RNNs or **BRNN**.
  - Computing:
    \[
    \begin{align*}
      a^{\langle 0 \rangle} &= \vec{0} \\

      a^{\langle 1 \rangle} &= g_1(w_{aa} a^{\langle 0 \rangle} + w_{ax} x^{\langle 1 \rangle} + b_a) \leftarrow tanh/ReLU \\

      \hat{y}^{\langle 1 \rangle} &= g_2(w_{ya} a^{\langle 1 \rangle} + b_y) \leftarrow sigmoid \\

      \vdots \\

      a^{\langle t \rangle} &= g_t(w_{aa} a^{\langle t-1 \rangle} + w_{ax} x^{\langle t \rangle} + b_a) \leftarrow tanh/ReLU \\

      \hat{y}^{\langle t \rangle} &= g_t(w_{ya} a^{\langle t \rangle} + b_y) \leftarrow sigmoid
    \end{align*}
    \]
    - Writing things in a simpler way:
      \[
      \begin{align*}
        a^{\langle t \rangle} &= g(W_a [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_a) \\

        [W_{aa} \ W_{ax}] &= W_a \\

        [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] &= \begin{bmatrix} a^{\langle t-1 \rangle} \\ x^{\langle t \rangle} \end{bmatrix} \\

        \hat{y}^{\langle t \rangle} &= g(W_y a^{\langle t \rangle} + b_y)
      \end{align*}
      \]
  - **Backprop in RNNs**:
    - Backprop through time:
      \[
      \begin{align*}
      \mathcal{L}^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}) &= -y^{\langle t \rangle} \log{\hat{y}^{\langle t \rangle}} - (1-y^{\langle t \rangle}) \log{(1-\hat{y}^{\langle t \rangle})} \\

      \mathcal{L}(\hat{y}, y) &= \sum^{T_y}_{t=1} \mathcal{L}^{\langle t \rangle} (\hat{y}^{\langle t \rangle}, y^{\langle t \rangle})
      \end{align*}
      \]
- **Different types of RNNs**:
  - Sentiment Classification: you can simplify the architecture into a "many-to-one" architecture, because only the last logits is meaningful.
  - Music Generation: "one-to-many". (There are feed forwards in this case.)
  - Machine Translation: "many-to-many", the input and the output have different lengths. You usually divide the architecture into encoders and decoders.
- **Speech Recognition**:
  \[
  \begin{align*}
    & P(The \ apple \ and \ pair \ salad) = 3,2 \cdot 10^{-13} \\
    & P(The \ apple \ and \ pear \ salad) = 5,7 \cdot 10^{-10} \\
    & P(sentence) = P(y^{\langle 1 \rangle},..., y^{\langle t \rangle}) = ?
  \end{align*}
  \]
- **Language Modelling with an RNN**:
  - Training Set: large corpus of english text.
  - Tokenize your sentences.
    - Add extra tokens for the ending of a sentence `<EOS>`
    - Can maybe add punctuation as tokens also.
  - What if the word is **not** in the training set?
    - Model it as `<UNK>`.
  - **Instead** of feeding the second stage of the architecture with a new word, you feed it with the correct answer of the previous guess.
  - **Cost Function**: Softmax
    \[
    \begin{align*}
      \mathcal{L}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}) &= - \sum_{i} y^{\langle t \rangle}_i \log{\hat{y}^{\langle t \rangle}_i} \\

      \mathcal{L} &= \sum_t \mathcal{L}^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle})
    \end{align*}
    \]
  - **Sampling**: `np.random.choice` and then force your RNN to calculate $P(\_\_\_\_ \ | \ previous \ sample)$.
  - **Character-Level**: instead of your vocabulary being words, you do it with only characters.
    - Usually doesn't capture long-range dependencies.
    - Much more computationally expensive.
- **Vanishing Gradients with RNNs**: exploding gradients can be catastrophic, but vanishing gradients are much more common in RNNs.
  - The cat, (...), was full.
  - The cats, (...), were full.
  - For exploding gradients, you can apply **gradient clipping**.
- **Gated Recurrent Unit (GRU)**: create a memory cell that will force the RNN to remember an important value.
  - Simplified:
  \[
  \begin{align*}
    & \tilde{c}^{\langle t \rangle} = \tanh{(W_c [c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c)} \\

    & \Gamma_u = \sigma(W_u [c^{\langle t-1 \rangle}, x^{t}] + b_u) \\

    & c^{\langle t \rangle} = \Gamma_u \ast \tilde{c}^{\langle t \rangle} + (1 - \Gamma_u) \ast c^{\langle t-1 \rangle}
  \end{align*}
  \]
  - This way, when $\Gamma_u = 0 \Rightarrow c^{\langle t \rangle} = c^{\langle t-1 \rangle}$.
  - $c^{\langle t \rangle}$ can be a vector, that's why we have $\ast$, for element-wise multiplication.
    	- This can also be used to tell the system which bits should be updated.
  - **Full GRU**: add a $\Gamma_r$ factor for relevance.
  \[
  \begin{align*}
    & \tilde{c}^{\langle t \rangle} = \tanh{(W_c [\Gamma_r \ast c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c)} \\

    & \Gamma_u = \sigma(W_u [c^{\langle t-1 \rangle}, x^{t}] + b_u) \\

    & \Gamma_r = \sigma(W_r[c^{\langle t \rangle}, x^{\langle t \rangle}] + b_r) \\

    & c^{\langle t \rangle} = \Gamma_u \ast \tilde{c}^{\langle t \rangle} + (1 - \Gamma_u) \ast c^{\langle t-1 \rangle}
  \end{align*}
  \]
- **LSTM**: Hochreiter (1997). There are 3 gates now, including an output gate.
  \[
  \begin{align*}
    & \tilde{c}^{\langle t \rangle} = \tanh{(W_c [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c)} \\

    & \Gamma_u = \sigma(W_u [a^{\langle t-1 \rangle}, x^{t}] + b_u) \\

    & \Gamma_f = \sigma(W_f [a^{\langle t-1 \rangle}, x^{t}] + b_f) \\

    & \Gamma_o = \sigma(W_o [a^{\langle t-1 \rangle}, x^{t}] + b_o) \\

    & c^{\langle t \rangle} = \Gamma_u \ast \tilde{c}^{\langle t \rangle} + \Gamma_f \ast c^{\langle t-1 \rangle} \\

    & a^{\langle t \rangle} = \Gamma_o \ast \tanh{(c^{\langle t \rangle})} \\

    & y^{\langle t \rangle} = softmax{(W_y a^{\langle t \rangle} + b_y)}
  \end{align*}
  \]
  - Chris Olah (and Andrej Karpathy) has a blog post on trying to understand LSTMs.
  - Some people have also tried to replace $x^{\langle t \rangle}$ for $c^{\langle t-1 \rangle}$, which yields the so called **peephole connection**.
  - There is no universally superior way when it comes to GRUs and LSTMs.
    - If you want deeper networks, use GRUs.
    - But the more robust choice is usually LSTMs.
- **Bidirectional RNNs**:
  - You will need the person to stop talking to process the speech.
  \[
  \hat{y}^{\langle t \rangle} = g(W_y [\overrightarrow{a}^{\langle t \rangle}, \overleftarrow{a}^{\langle t \rangle}] + b_y)
  \]
- **Deep RNNs**:
  - Same architecture, but now with layers, i.e., $a^{[l] \langle t \rangle}$.
  - For RNNs, 3 layers is already quite a lot.
  - What you see sometimes is layers that are not connected horizontally (recurrently) in only the first stages.

## Week 2

- **Word Representation**:
  - One hot representations only allow the words to represent themselves separately, not allowing for connections between them.
    - The **inner product** between words is always zero.
  - **Featurized Representation**: word embedding:
    \[
    \begin{center}
    \begin{tabular}{c | c c c c c c }
      & Man & Woman & King & Queen & Apple & Orange \\
      \hline \\
      Gender & -1 & 1 & -0.95 & 0.97 & 0.00 & 0.01 \\
      Royal & 0.01 & 0.02 & 0.93 & 0.95 & -0.01 & 0.00 \\
      Age & 0.03 & 0.02 & 0.7 & 0.69 & 0.03 & -0.02 \\
      Food & 0.04 & 0.01 & 0.02 & 0.01 & 0.95 & 0.97
    \end{tabular}
    \end{center}
    \]
  - One way of **visualizing word embeddings** is the **t-SNE** (t-distributed stochastic neighbor embedding) algorithm by Maaten & Hinton (2008). It takes, for example, a 300 dimensional vector and makes a visualization in 2D.
  - Word embeddings can also boost your network with **transfer learning**. (They will be a layer in your network, an embedding layer.)
    1. Learn word embeddings from large text corpus ($10^{9}-10^{11}$ words). (Or download pre-trained embeddings online.)
    1. Transfer embedding to new task with smaller training set.
    1. Optional: Continue to finetune the word embeddings with new data.
  - Word embedding also has also interesting implications with CNNs, though there we would rather say "encoding", since the "vocabulary" isn't restricted to a dictionary.
- **Notation**:
  - $O_{5391}$ is the one-hot embedding of a word in position 5391 of the dictionary.
  - $e_{5391}$ is the word embedding of a word in position 5391 of the dictionary.
- **Linguistic Regularities** (Mikolov, 2013):
  - Example: Man $\rightarrow$ Woman; King $\rightarrow$ ?
    - **Interesting Property** (and it works):
      \[
      \begin{align*}
        & e_{man} - e_{woman} \approx \begin{bmatrix} -2 \\ 0 \\ 0 \\ 0 \end{bmatrix} = Gender \ Vector \\

        & e_{king} - e_{queen} \approx \begin{bmatrix} -2 \\ 0 \\ 0 \\ 0 \end{bmatrix} = Gender \ Vector \\ \\

        & Find \ word \ such \ that: \\

        & e_{man} - e_{woman} \approx e_{king} - e_{queen} \\ \\

        & Find \ word \ w: \\

        & \arg \max (sim (e_w, e_{king} - e_{man} + e_{woman}))
      \end{align*}
      \]
      - The mappings of the 4 words will configure approximately a parallelogram, i.e., the vectors from woman to queen and man to king will also be parallel.
      - After t-SNE, it will probably not hold true.
      - **Similarity Functions**:
        - Cosine Similarity (more often used):
          \[
          sim(u,v) = \frac{u^{T} v}{||u||_2 ||v||_2}
          \]
- **Embedding Matrix**:
  - **Notation**: $(300, 10000)$ dimensional. Each word takes a column. If you multiply the one-hot representation, you end up with a vector $(300, 1)$, $E_{(300, 10000)} \cdot O_{6257_{(10000, 1)}} = e_{6257_{(300, 1)}}$, where $e_{6257}$ is the embedding.
  - Not efficient to multiply matrices, so, in practice, we use a specialized function.
- **Algorithms for learning the matrix $E$**:
  - Bengio, 2003: give $E$ and the $e$'s (with a fixed history window, i.e., context) of the phrase to an NN and have it run GD.
    - Context:
      - Last 4 words.
      - 4 words left & right.
      - Last 1 word
      - Nearby 1 word (super simple and works remarkably well)
  - **Word2Vec**: Mikolov (2013), efficient estimation of word representation in vector space.
    - **Skip-grams**:
      - Context: randomly pick a word in the phrase.
      - Target: randomly pick a word within some window.
      - $c$ and $t$ are chosen to be nearby words.
    \[
    Context \ c \rightarrow Target \ t \\
    Softmax: \ p(t|c) = \frac{e^{\theta_{t}^{T} e_c}}{\sum^{10000}_{j=1} e^{\theta_{j}^{T} e_c}}
    \]
    - where $\theta_t$ = parameter associated with the chance of a particular word $t$ being labeled.
    - $\theta_t$ and $e_c$ are both 500 dimensional vectors.
    - They are both trained with an optimization algorithm such as Adam or GD.
    - The loss function: ($y_i$ is a one-hot vector)
    \[
    \mathcal{L}(\hat{y}, y) = - \sum^{10000}_{i=1} y_i \log{\hat{y}_i}
    \]
    - **Problems**:
      - **Computational speed**: $\sum^{10000}_{j=1}$ can be quite slow. Usually people use a hierarchical tree, rather than a full sum. This way, the complexity goes to $O(\log{v})$
      - How to sample context c? **Remove stopwords**.
  - **Negative Sampling**: Mikolov (2013).
    - From a word, say, juice, we sample a context, say, orange, which will receive a target of True or 1. Then, we randomly sample other $k$ words from the dictionary, which will be our negative examples, having a target of False or 0.
      - It's okay if the negative samples are also in the phrase.
      - We have a ratio of $k:1$ negative to positive examples.
    \[
    P(y=1|c,t) = \sigma(\theta^{T}_c e_c)
    \]
    - Instead of training 10,000 logistic regression units. We are going to train only those related to the sampled targets, i.e., $k+1$.
    - **Selecting Negative Examples**: a middle ground between sampling uniformly among all the words and the most frequent ones.
      \[
      P(w_i) = \frac{f(w_i)^{3/4}}{\sum^{10000}_{j=1} f(w_j)^{3/4}}
      \]
  - **GloVe**: global vectors for word represantion, Pennington (2014).
    \[
    \begin{align*}
      & c, t \\
      & x_{ij} = \# times \ i \ appears \ in \ context \ of \ j \\

      & (Sometimes \ x_{ij} = x_{ji}) \\ \\

      & Minimize: \\

      & \sum^{10000}_{i=1} \sum^{10000}_{j=1} f(x_{ij})(\theta_{i}^{T} e_{j} + b_i + b_{j}^{\prime} - \log{x_{ij}})^2 \\

      & e_{w}^{(final)} = \frac{e_w + \theta_w}{2}
    \end{align*}
    \]
    - $f(x_{ij})$ is a weighting term, it is useful for very frequent and infrequent words. The choice of $f$ is curious.
      - $0 \cdot \log{0} = 0$
    - $\theta_i$ and $e_j$ should be initialized randomly at the beginning of the training.
    - $x_{ij}$ is the number of times the word $i$ appears in the context of word $j$.
  - A **note** on the featurization view of word embeddings:
    - You cannot guarantee that the features are interpretable.
      \[
      (A \theta_i)^T (A^{-T} e_j) = \theta_{i}^T A^T A^{-T} e_j
      \]
- **Sentiment Classification**:
  - Use transfer learning to get the matrix $E$.
  - Feed $e_i$'s to an NN and then softmax it.
    - Ignores word order.
      - "Completely lacking in good taste, good service, etc."
  - Feed it into an RNN. Many-to-one architecture.
- **Debiasing Word Embeddings**:
  - "Man is to computer as woman is to homemaker".
  1. Identity bias direction (in higher dimensions, SVD is used)
    \[
    \begin{align*}
      & e_{he} - e_{she} \\
      & e_{male} - e_{female} \\
      & \vdots \\
      & average
    \end{align*}
    \]
  2. Neutralize: for every word that is not definitional (most words are not definitional), project them on the symmetry axis to get rid of bias.
  3. Equalize pairs. For example, you want the only difference between grandfather and grandmother to be in gender.
  \[
  \begin{align*}
    & e^{bias \ component} = \frac{e \cdot g}{||g||_2^2} * g \\
    & e^{debiased} = e - e^{bias \ component}
  \end{align*}
  \]

- **Keras and Mini-Batching**: most frameworks only allow for mini-batches to have the same length. With phrases, this wouldn't work because of the different lengths.
  - The common solution is to pad the phrases: $(e_{I}, e_{love}, e_{you}, \vec{0}, \vec{0}, \ldots, \vec{0})$.
    - This way the final dimensions for the input will be $(m, max\_length, embedding\_feature\_size)$.

## Week 3

- **Sequence to Sequence Models**: using an encoder-decoder scheme. This also works for captioning. $P(english|french)$.
  - Sampling: do **not** use random sampling.
    - Why not **Greedy Search**(after finding the most likely first word, find the next and so on...)? Doesn't work with language.
    - **Beam Search**:
      \[
      \begin{align*}
        \arg \max(y) \prod^{T_y}_{t=1} &P(y^{\langle t \rangle}|x, y^{\langle 1 \rangle}, \ldots, y^{\langle t-1 \rangle}) = \\

        & P(y^{\langle 1 \rangle}|x) P(y^{\langle 2 \rangle}|x, y^{\langle 1 \rangle}) \ldots P(y^{\langle T_y \rangle}|x, y^{\langle 1 \rangle}, \ldots y^{\langle T_y \rangle})
      \end{align*}
      \]
      1. Run the sentence through the encoder.
      1. Start with the first word: $P(y^{\langle 1 \rangle}|x)$
      1. Set a Beam, for example $B = 3$ (if $B = 1$, then Beam Search becomes Greedy Search). It will keep in memory the best $B$ choices for the first word.
      1. For each of the best choices, do the same as step 2: $P(y^{\langle 2 \rangle}|x, jane) \Rightarrow P(y^{\langle 1 \rangle}, y^{\langle 2 \rangle}|x) = P(y^{\langle 1 \rangle}|x) P(y^{\langle 2 \rangle}|x, y^{\langle 1 \rangle})$. This way, you've cut out almost all the 10,000 possibilities of the dictionary to only $B$ possibilities.
      - **Refinements**:
        - **Length Normalization**: the probabilities may become so small that they might cause computer underflow. So, instead, we use $\log$ normalized by the number of words you have in your translation:
        \[
        \begin{align*}
          & \arg \max(y) \frac{1}{T_{y}^{\alpha}} \sum^{T_y}_{t=1} \log{(P(y^{\langle t \rangle}|x, y^{\langle 1 \rangle}, \ldots, y^{\langle t-1 \rangle}))} \\

          & where \ \alpha \ is \ usually = 0.7
        \end{align*}
        \]
        - How to choose beam width $B$?
          - Large B: better result, slower.
          - Small B: worse result, faster.
          - In production systems, $B$ goes from 10 to 100, sometimes 1000 or 3000, depending on the application.
      - Unlike exact search algorithms like BFS (Breadth First Search) or DFS (Depth First Search), Beam Search runs faster but is not guaranteed to find the exact maximum $\arg \max(y) P(y|x)$.
      - **Error Analysis in Beam Search**:
        - Jane visite l'Afrique en septembre.
        - $y^{*}$: Human: Jane visits Africa in September.
        - $\hat{y}$: Algorithm: Jane visited Africa last September.
        - To assess the error, we can compute both $P(y^{*}|x)$ and $P(\hat{y}|x)$ using the RNN to get an idea of where the error is.
          - Case 1: $P(y^{*}|x) > P(\hat{y}|x)$.
            - Beam search chose $\hat{y}$. But $y^{*}$ attains higher $P(y|x)$.
            - Conclusion: Beam search is at fault.
          - Case 2: $P(y^{*}|x) \leq P(\hat{y}|x)$.
            - $y^{*}$ is a better translation than $\hat{y}$. But RNN predicted $P(y^{*}|x) \leq P(\hat{y}|x)$.
            - Conclusion: RNN model is at fault.
        - Through this process you can do error analysis step by step in the translation. And then, the most frequent error will be the one you correct first.
      - **Evaluating Machine Translation**: Bleu Score (Papineni, 2002).
        - Example:
          - French: Le chat est sur le tapis.
          - Reference 1: the cat is on the mat.
          - Reference 2: there is a cat on the mat.
          - MT Output: the the the the the the the.
          - Modified Precision: $max(\# \ appearances / total \# \ words) = 2/7$.
        - Bigrams: the cat; cat the; cat on; on the; the mat.
        - Count Clip: the number of times the bigram appears in at most one of the references.
        \[
        \begin{align*}
          & p_1 = \frac{\sum_{unigram \in \hat{y}} Count_{clip}(unigram)}{\sum_{unigram \in \hat{y}} Count(unigram)} \\

          & p_n = \frac{\sum_{ngram \in \hat{y}} Count_{clip}(ngram)}{\sum_{ngram \in \hat{y}} Count(ngram)}
        \end{align*}
        \]
        - Combined Bleu Score:
          \[
          CBS = BP \cdot exp(\frac{1}{m} \sum^{m}_{n=1} p_n)
          \]
          - where $BP$ is the brevity penalty, because short translations are much easier.
          \[
          BP =
            \begin{cases}
              1 \ if \ MT\_output\_length > reference\_output\_length \\
              exp(1 - MT\_output\_length/reference\_output\_length) \ otherwise
            \end{cases}
          \]
- **Attention Model**: Bahdanau (2014).
  - For very long sentences, the Bleu Score usually drops rapidly. So maybe we should crack the long sentences into smaller parts, much like human translators do.
  - In it each word will get a context weight $\alpha^{\langle t,x \rangle}$, where $x$ is the other item in the phrase.
    \[
    \begin{align*}
      & a^{\langle t^{\prime} \rangle} = (\overrightarrow{a}^{\langle t \rangle}, \overrightarrow{a}^{\langle t \rangle}) \\ \\

      & \sum_{t^{\prime}} \alpha^{\langle 1,t^{\prime} \rangle} = 1 \\ \\

      & c^{\langle 1 \rangle} = \sum_{t^{\prime}} \alpha^{\langle 1,t^{\prime} \rangle} a^{\langle t^{\prime} \rangle}
    \end{align*}
    \]
    - Computing $\alpha^{\langle t,t^{\prime} \rangle}$, we basically use a softmax, with $a^{\langle t^{\prime} \rangle}$ and $s^{\langle t-1 \rangle}$ as inputs.
      \[
      \alpha^{\langle t,t^{\prime} \rangle} = \frac{exp(e^{\langle t,t^{\prime} \rangle})}{\sum^{T_x}_{t^{\prime}=1} exp(e^{\langle t,t^{\prime} \rangle})}
      \]
  - You can also visualize the attention weights to see if everything is ok.
  - Con: It takes $O(N^2)$ to compute the algorithm.
- **Speech Recognition**:
  - Phoneme Representations are no longer necessary.
  - **CTC Cost** for Speech Recognition: connectionist temporal classification (Graves, 2006).
    - ttt\_h\_eee\_\_\_(space)\_\_\_\_qqq\_\_\_ $\rightarrow$ the(space)q
      - Basic Rule: collapse repeated characters not separated by "blank".
  - **Trigger Word Detection**:
    - Set the target label to be one right after the trigger word is said.
      - One of the problems is that the training data is very sparse. You could improve it to output more ones around when the word is said.
