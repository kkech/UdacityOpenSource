# flashcard_recommender

## Plan:
### 1) Object
### 2) Recommandation with Reinforcement Learning 
### 3) Privacy problematics
### 4) Data securing with PySyft

## 1) Object

The goal of this code is to design a machinery able to gather data of a set of users, in order to find relationships between exercise flashcards. With those data, it will then be able to order a given set of flashcard by increasing difficulty, so that the user will have the better pedagogical experience as possible in a session.

### About Flashcards:

Flashcards are an old popular tool for students to revise their lessons. One flashcard is a small card containing only one single information. So when a student has created a set of flashcards for one lesson, s.he can quickly browse it anywhere when s.he has time.

Cards can be used for exercises too. One side will be the question, the other side - the answer.
**Exemples:**
*Maths* : Q : 2 + 2, A : 4
*French* : Q : Chat, A : Cat
*Geography* : Q : Capital of Germany, A : Berlin
...

### Flashcards on smartphones, Improve the learning experience

Smartphones apps have become very popular for their ability to store huge sets of smartcards, share them in user communities, and also for their efficient revision management systems. Indeed, the app memorises which cards the user has seen and when, her or his preferences of apparition frequency and other options depending on the app. We propose to do another step by making the app able to determine which exercise card should be proposed knowing the user answer history.

**Smartphones as pocket computers provide powerful flashcard management**. But it maystill be improved. Let us take a math exemple. If a teenager has difficulty with integer addition, it appears evident that it should focus on them before trying to resolve substraction between fractions. So if the user answered wrong to "48 + 27" and "17 + 34", it may be better to propose "7 + 4" than "12.6 - 14.8".
So, an app should benefit from learning which card progression would help the user to learn better. For a given card set, if the app has no previous indication about the relationship between them, it has to learn them. For instance, if given two cards A and B, the rate of good answers for B does not depends on A being solved or not, then the two cards will appear to be independent. If 70% of the users who were right at A are right at B, and 30% of those who were wrong at A are right at B, then you may think that A should be proposed prior to B. 
If the app has a large community, then distributed machine learning can improve any user app on its mobile phone by using the data of the whole community. However we have to pay attention to user privacy.

## 2) Recommandation with Reinforcement Learning

Reinforcement Learning is a promising solution for a dynamical recommandation system in permanent evolution with a user community. Indeed:
- the user community makes a wide environment providing a constant stream of data,
- the score (percentage of good answer) makes an good reward function for the agent,
- the action is the ordering of cards for given subsets

**Deep learning setting:**
The agent takes in input a set or cards from a fixed collection (for instance, 10 cards from 1.000 short geography questions) and propose the best difficulty order (For instance in maths exercises, propose additions before substractions, but maybe substractions with integer before additions with fractions, depending on users' success)
An **actor neural network** takes this subset as an input, and return a list of float (softmax output function), which ordering provides the estimated best progression for this subset (for instance, if the output is (7.1, 0.4, 6.2, 12.5, 1.3), then the best ordering of the cards, leading to higher expected score from the user who gains experience by seeing the cards, would be 4-1-3-5-2)
A **critic neural network** learns the average score from the users for given cards ordering and predicts the average score depending on a given ordering, generating the q-values for the actor policy.

## 3) Privacy problematics

When the user allows the app to use her results to improve the learning algorithm, privacy questions appears. For instance:
The app records the user's stats over a cards for french learning. Indirectly, it gives to external watchers the possibility to evaluates her level and compare it with other users. If the user learns french to apply for a job, she may want to keep her data hidden. If the app has a large community, without privacy securing, it could sell abusively records of users' level to hiring companies requiring french speaking employees, providing them excessive power to negociate working conditions.

## 4) Data securing with PySyft

3 solutions will be proposed to improve the data privacy:
- **Federated Learning:**
Users data never leave their device for a data center. The neural networks for actor-critic models are stored on each device, update are done there and only the update steps are centralized
- **Data encryption**
To avoid data to leak during the models update, encryption will secure the weights transfer
- **Local differential privacy**
The randomness of answering a set of exercises can used with high benefice to secure users' privacy. Indeed, if the app stores randomly modified scores according to local differential privacy protocols (for each result, randomly decide to enter the right score, or to enter a random score) then watching individual data won't say anything about her, but in average over the community, the app will be able to learn the correlation between cards be cancelling the errors statisticaly.
