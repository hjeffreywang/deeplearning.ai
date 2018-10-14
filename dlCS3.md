# ML Strategy

## Week 1

- **Why ML?** Sometimes more data won't do no good.
- **Orthogonalization**: what to tune to be more effective in improving your model. Different knobs for different stages of the design.
  - **Chain Assumptions of ML**:
    - Fit training set well on cost function
      - Bigger network
      - ADAM
    - Fit dev set well on cost function
      - Regularization
    - Fit test set well on cost function
      - Bigger dev set
    - Perform well in real world
      - Change dev set or cost function
  - Try to avoid **early stopping** because it simultaneously affects the training and dev sets performances.
- Using a **single number evaluation metric**:
  - **Precision**: of examples recognized as cats, what % were actually cats?
  - **Recall**: what % of actual cats were actually recognized.
  - There is usually a **trade-off** between precision and recall.
  - Instead of picking one evaluation parameter, you should probably use a combination of them. Usually, the **F1 Score** is used, the harmonic mean of $P$ and $R$.
    \[
    F1 \ Score = \frac{2}{\frac{1}{P} + \frac{1}{R}}
    \]
  - **Dev + Single Number Evaluation Metric** $\Rightarrow$ speedy implementation.
- **Satisficing and Optimizing Metric**:
  - Ex.: maximize accuracy, subject to running time $\leq$ 100 ms.
    - Accuracy is the optimizing metric, and running time is the satisficing metric.
    - $N$ metrics:
      - $1$ optimizing
      - $N-1$ satisficing
- **Setting up the Dev/Test Sets**
  - They should come from the same distributions.
    - Randomly shuffle data from different regions.
  - Sometimes, having no dev set could be ok.
  - **When to change the dev/test** sets and metrics:
    - Say, in the cat classifier, the better algorithm is letting through some pornographic images. That means the metric is not good yet. So define a new error metric:
      \[
      \begin{align*}
        &\frac{1}{m_{dev}} \sum^{m_{dev}}_{i=1} w^{(i)} \mathcal{L} (y^{(i)}_{pred} + y^{(i)}) \\

        &w^{(i)} =
          \begin{cases}
            1, \ if \ x^{(i)} \ is \ non-porn \\
            10, \ if \ x^{(i)} \ is \ porn
          \end{cases}
      \end{align*}
      \]
- **Comparing to Human-Level Performance**: the performance quickly surpasses human performance but then plateaus after a short while and you won't be able to surpass the theoretical error, i.e., *Bayes Error* ($\leq$ human level performance).
  - Humans are quite good at a lot of tasks.
  - So long as ML is worse than humans, you can:
    - Get labeled data from humans.
    - Gain insight from manual error analysis: Why did a person get this right?
    - Better analysis of bias/variance.
  - **Avoidable Bias**: the difference between human level performance and the training performance.
  - **Variance**: proportional to the difference between the training and the dev errors.
  - **Surpassing Human-Level Performance**:
    - When you surpass human performance, you lose sight of where the *bayes error* is. The ways of making progress are less clear.

### Flight Simulator - Birds

1. The City Council tells you that they want an algorithm that

  1. Has high accuracy

  1. Runs quickly and takes only a short time to classify a new image.

  1. Can fit in a small amount of memory, so that it can run in a small processor that the city will attach to many different security cameras.

  Having three evaluation metrics makes it harder for you to quickly choose between two different algorithms, and will slow down the speed with which your team can iterate. True/False?

  - True

2. The City Council tells you that they want an algorithm that

  - Has high accuracy

  - Runs quickly and takes only a short time to classify a new image.

  - Can fit in a small amount of memory, so that it can run in a small processor that the city will attach to many different security cameras.

  - Test: 98%/ Runtime: 9sec/ Memory: 9MB

3. Based on the city’s requests, which of the following would you say is true?

  - Accuracy is an optimizing metric; running time and memory size are a satisficing metrics.

4. Before implementing your algorithm, you need to split your data into train/dev/test sets. Which of these do you think is the best choice?

  - Train: 9,500,000/ Dev: 250,000/ Test: 250,000

5. You should not add the citizens’ data to the training set, because this will cause the training and dev/test set distributions to become different, thus hurting dev and test set performance. True/False?

  - False

6. One member of the City Council knows a little about machine learning, and thinks you should add the 1,000,000 citizens’ data images to the test set. You object because:

  - The test set no longer reflects the distribution of data (security cameras) you most care about.

  - This would cause the dev and test set distributions to become different. This is a bad idea because you’re not aiming where you want to hit.

7. (Train Error: 4% / Dev Set Error: 4.5%) This suggests that one good avenue for improving performance is to train a bigger network so as to drive down the 4.0% training error. Do you agree?

  X No, because this shows your variance is higher than your bias.

  X Yes, because having 4.0% training error shows you have high bias.

  - No, because there is not enough information to tell.

8. If your goal is to have “human-level performance” be a proxy (or estimate) for Bayes error, how would you define “human-level performance”?

  - 0.3% (accuracy of expert \#1)

9. Which of the following statements do you agree with?

  - A learning algorithm’s performance can be better than human-level performance but it can never be better than Bayes error.

10. (Human: 0.1% / Training: 2% / Dev Set: 2.1%) Based on the evidence you have, which two of the following four options seem the most promising to try? (Check two options.)

  - Try decreasing regularization.

  - Train a bigger model to try to do better on the training set.

11. (Human: 0.1% / Training: 2% / Dev: 2.1% / Test: 7%) What does this mean? (Check the two best options.)

  - You should try to get a bigger dev set.

  - You have overfit to the dev set.

12. (Human: 0.1% / Train: 0.05% / Dev: 0.05%) What can you conclude? (Check all that apply.)

  - If the test set is big enough for the 0.05% error estimate to be accurate, this implies Bayes error is ≤0.05

  - It is now harder to measure avoidable bias, thus progress will be slower going forward.

13. Because even though you have higher overall accuracy, you have more false negatives (failing to raise an alarm when a bird is in the air). What should you do?

  X Ask your team to take into account both accuracy and false negative rate during development.

  - Rethink the appropriate metric for this task, and ask your team to tune to the new metric.

14. You have only 1,000 images of the new species of bird. The city expects a better system from you within the next 3 months. Which of these should you do first?

  X Add the 1,000 images into your dataset and reshuffle into a new train/dev/test split.

  - Use the data you have to define a new evaluation metric (using a new dev/test set) taking into account the new species, and use that to drive further progress for your team.

15. Because of years of working on Cat detectors, you have such a huge dataset of 100,000,000 cat images that training on this data takes about two weeks. Which of the statements do you agree with? (Check all that agree.)

  - Buying faster computers could speed up your teams’ iteration speed and thus your team’s productivity.

  - If 100,000,000 examples is enough to build a good enough Cat detector, you might be better of training with just 10,000,000 examples to gain a $\approx$10x improvement in how quickly you can run experiments, even if each model performs a bit worse because it’s trained on less data.

  - Needing two weeks to train will limit the speed at which you can iterate.

## Week 2

- Say your algorithm is missing out on certain breeds of dogs that look like cats. Should you reshape the cost function to work better with these dogs?
  - The suggestion: do error analysis first:
    - Get ~100 mislabeled dev set examples.
    - Count up how many are dogs.
    - It might be that the mistakes on these dogs are not relevant to the total amount of errors.
- Evaluate multiple ideas in parallel. (You can use a **spread sheet** to see common points in the error samples.)
  - Improve great cats (lions, panthers, etc) recognition.
  - Improve performance on blurry images.
  - Improve dog recognition.
- Incorretly labeled training examples:
  - DL algorithms are quite **robust to random errors** in the training set.
  - Add another column to the spread sheet analysis.
- Apply corrections to both the test and dev sets to make sure they come from the same distribution.
- Examine what your algorithm got right also.
- Build your first system quickly, then iterate.
  - Use Bias/Variance analysis & Error analysis to prioritize next steps.
- Training and Testing on **Different Distributions**:
  - Option \#1: Shuffle distributions.
  - Option \#2: (**Better**) Shuffle on the train set, but keep the dev and test set as the intended distribution.
- If you have different distributions for the dev and test sets, you can create a **Train-Dev** set.
  - If you have a good performance on training and bad on the train-dev, it means you have a **data-mismatch**.
  - Strange things might happen depending on the difficulty of the training examples.
- **Addressing Data Mismatch**:
  - There aren't completely systematic solutions.
  - Carry out error analysis between the training and dev/test sets.
  - Make training and dev/test sets more similar.
  - You can use **artificial data synthesis** with artificial noise or other effects.
    - Try not to use a small dataset for the synthesis effect, the model might overfit to the artificial noise.
    - Even realistic videogame synthesis might end up with an overfitting model.
- **Transfer Learning**:
  - If low level features in one task are useful in another, use the base part of the NN and then reinitialize or add ending layers.
  - Is **very useful** when you don't have enough data, so you can use data from other problems to help solve the problem.
- **Multi-task Learning**:
  - Loss Function:
    \[
    J(\hat{y}) = \frac{1}{m} \sum^{m}_{i=1} \sum^{\#tasks}_{j=1} \mathcal{L}(\hat{y}^{(i)}_j, y^{(i)}_j)
    \]
  - You can even give question marks in the training set.
  - For a big enough network, the performance should be close to separate tasks.
- **End-to-End DL**:
  - Replacing many separate algorithms with a NN, which simplifies the problem.
  - In many systems, though, a hybrid system might be best. For example, in face recognition, first you zoom in the face and then classify the face.
  - Pros:
    - Lets the data speak.
    - Less hand-designing of components needed.
  - Cons:
    - May need large amount of data.
    - Excludes potentially useful hand-designed components.

## Case Study - Autonomous Driving Study

1. Your 100,000 labeled images are taken using the front-facing camera of your car. This is also the distribution of data you care most about doing well on. You think you might be able to get a much larger dataset off the internet, that could be helpful for training even if the distribution of internet data is not the same.

You are just getting started on this project. What is the first thing you do? Assume each of the steps below would take about an equal amount of time (a few days).

  - Spend a few days training a basic model and see what mistakes it makes.

2. For the output layer, a softmax activation would be a good choice for the output layer because this is a multi-task learning problem. True/False?

  - False

3. You are carrying out error analysis and counting up what errors the algorithm makes. Which of these datasets do you think you should manually go through and carefully examine, one image at a time?

  - 500 images on which the algorithm made a mistake

4. Because this is a multi-task learning problem, you need to have all your $y^{(i)}$ vectors fully labeled. If one example is equal to $[0,?,1,1,?]$ then the learning algorithm will not be able to use that example.

  - False.

5. The distribution of data you care about contains images from your car’s front-facing camera; which comes from a different distribution than the images you were able to find and download off the internet. How should you split the dataset into train/dev/test sets?

  - Choose the training set to be the 900,000 images from the internet along with 80,000 images from your car’s front-facing camera. The 20,000 remaining images will be split equally in dev and test sets.

6.

  - Train 8.8%: 940,000 images randomly picked from (900,000 internet images + 60,000 car’s front-facing camera images)
  - Train-dev 9.1%: 20,000 images randomly picked from (900,000 internet images + 60,000 car’s front-facing camera images)
  - Dev 14.3%: 20,000 images from your car’s front-facing camera
  - Test 14.8%: 20,000 images from the car’s front-facing camera

  - You also know that human-level error on the road sign and traffic signals classification task is around 0.5%. Which of the following are True? (Check all that apply).

    - You have a large avoidable-bias problem because your training error is quite a bit higher than the human-level error.

    - You have a large data-mismatch problem because your model does a lot better on the training-dev set than on the dev set

7. Based on table from the previous question, a friend thinks that the training data distribution is much easier than the dev/test distribution. What do you think?

  - There’s insufficient information to tell if your friend is right or wrong.

8. You decide to focus on the dev set and check by hand what are the errors due to. Here is a table summarizing your discoveries:

  - Dev Error: 14.3%
  - Mislabeled Data: 4.1%
  - Foggy Pictures: 8%
  - Rain Drops: 2.2%
  - Other: 1%

  - The results from this analysis implies that the team’s highest priority should be to bring more foggy pictures into the training set so as to address the 8.0% of errors in that category. True/False?

    - False because this would depend on how easy it is to add this data and how much you think your team thinks it’ll help.

9. You can buy a specially designed windshield wiper that help wipe off some of the raindrops on the front-facing camera. Based on the table from the previous question, which of the following statements do you agree with?

  - 2.2% would be a reasonable estimate of the maximum amount this windshield wiper could improve performance.

10. You decide to use data augmentation to address foggy images. You find 1,000 pictures of fog off the internet, and “add” them to clean images to synthesize foggy days, like this:

  - So long as the synthesized fog looks realistic to the human eye, you can be confident that the synthesized data is accurately capturing the distribution of real foggy images (or a subset of it), since human vision is very accurate for the problem you’re solving.

11. After working further on the problem, you’ve decided to correct the incorrectly labeled data on the dev set. Which of these statements do you agree with? (Check all that apply).

  - You should also correct the incorrectly labeled data in the test set, so that the dev and test sets continue to come from the same distribution

  - You should not correct incorrectly labeled data in the training set as it does not worth the time.

12. So far your algorithm only recognizes red and green traffic lights. One of your colleagues in the startup is starting to work on recognizing a yellow traffic light. (Some countries call it an orange light rather than a yellow light; we’ll use the US convention of calling it yellow.) Images containing yellow lights are quite rare, and she doesn’t have enough data to build a good model. She hopes you can help her out using transfer learning.

  - She should try using weights pre-trained on your dataset, and fine-tuning further with the yellow-light dataset.

13. Another colleague wants to use microphones placed outside the car to better hear if there’re other vehicles around you. For example, if there is a police vehicle behind you, you would be able to hear their siren. However, they don’t have much to train this audio system. How can you help?

  - Neither transfer learning nor multi-task learning seems promising.

14.

  - (A) Input an image (x) to a neural network and have it directly learn a mapping to make a prediction as to whether there’s a red light and/or green light (y).
  - (B) In this two-step approach, you would first (i) detect the traffic light in the image (if any), then (ii) determine the color of the illuminated lamp in the traffic light.

  - Between these two, Approach B is more of an end-to-end approach because it has distinct steps for the input end and the output end. True/False?

    - False

15. Approach A (in the question above) tends to be more promising than approach B if you have a ________ (fill in the blank).

  - Large training set
