# Neural Question Generation

## Introduction

The main objective of our project is to employ a novel approach for generating a comprehensive set of questions from a given text using an advanced computational model. Specifically, we aim to outline a methodology that can facilitate the generation of a diverse and relevant set of questions, which in turn can assist in comprehending the text better.

## Dataset

The dataset used by the paper was the SQuAD dataset (The Stanford Question Answering Dataset) which consisted of over 100,000+ question-answer pairs sampled from the top 10000 wikipedia articles.

However, the dataset we used was the Question Answering in Context, provides three different types of training and testing data (corpus, questions and answers). There are 14k unique corpa, 100k unique questions, and 14k unique answers in this dataset.

## Model

- After pre-processing the data, we first concatenate the word embeddings along with a smaller "answer tag", which is an embedding that stores information about the answer associated with the target question. We do this using two concatenated $\texttt{tf.keras.layers.Embedding}$ layers.

- This encoding passage gets fed into a stacked bidirectional LSTM, which eventually outputs a hidden representation of the initial passage. This is accomplished in our code using the $\texttt{Bidrectional}$ module which wraps a $\texttt{tf.keras.layers.LSTM}$ stack. The LSTM also has a dropout parameter, and all of these can be found inside $\texttt{config.py}$.

- We compute self attention on this passage, and pass it through a set of feature fusion gating layers, allowing the model to learn the relevance of certain input features. This may include sentence structure, overall theme of the passage, and the tone of the author. We accomplish this in $\texttt{gated_self_attention}$. This function computes energy and score tensors that are passed through hyperbolic tangent and softmax activations to compute the output, as described in the source literature.

- We then move to the decoder stack, where we use the target context sequence as a partial decoding. This gets passed into a unidirectional $\texttt{tf.keras.layers.LSTM}$ layer, which simply outputs a decoded state.

- We take this decoded state and along with the final passage-answer representation, compute an attention in the decoder stack. We also compute a attention vector, which is outputted by the $\texttt{attention}$ function in the decoder module. These eventually get fed into a maxout pointer network. This is a new technique - in the past, copy/pointer methods have shown exceptional results, but in the context of our model, this often leads to lots of word repetition, which we want to avoid. Therefore, the maxout pointer applies a hard cap on the scores of certain words, allowing for more diversity. 

- Finally, we acquire a final probability distribution for each word, and we use a beam search algorithm to generate outputs, which can be seen in $\texttt{inference.py}$. Beam search is a heuristic algorithm and computes a best-first search, meaning that it takes the best option from a limited scope, and extrapolates from there. In our case, our model evaluates words in the future as well as the current word, while a traditional greedy search would only look at the word right ahead. Using this time window allows for more stable inference.

## Challenges

- One severe limitation was that there was no official implementation of our paper so we had to make the model based of unofficial github repositories which is why there would be implementation differences in our model and the one used by the paper.

- Following up from the previous limitation, we also found it difficult to convert the code in torch1 to tensorflow2 due to a ton of non-overlapping and depricated methods with no alternatives, as well as converting the Blue evaluation script from python2 to python3 due to some depreicated methods which wasn't as bad as the former.

- Another limitation we ran into was the fact that we were only able to train our model on 4000 corpa as apposed to the original datasets 14000 corpa. The reason for this was because we were unable to train on higher number of corpa due to the memory constraints of our laptops.

- Following up from the previous limitation, we ran into lack of RAM issues both while training, as well as while attempting to generate questions hence our $\texttt{generated.txt}$ file only has around 100 questions that have been generated.

## Results

Bleu scores to display our results in a summary:

- Bleu_1: 0.26599
- Bleu_2: 0.09290
- Bleu_3: 0.03793
- Bleu_4: 0.01669

## Conclusion

Through the use of of a bi-directional LSTM with a gated self-attention encoder and maxout pointer decoder, we created an advanced computational model to generate a comprehensive set of relevant questions based on the textual contents of a given corpus. Even though we weren't able to generate amazing Bleu scores due to the challenges we faced, we believe that the model displays a positive trend which is expected to display positive results when properly trained on the entire dataset.

## PS

List of files we were unable to push to git repo due to size:
- Quac dataset
    - para-dev.txt
    - para-test.txt
    - para-train.txt
    - tgt-dev.txt
    - tgt-test.txt
    - tgt-train.txt
    - Each file is around 250 MB hence couldn't be uploaded
- Squad dataset
    - para-dev.txt
    - para-test.txt
    - para-train.txt
    - tgt-dev.txt
    - tgt-test.txt
    - tgt-train.txt
    - Each file is around 250 MB hence couldn't be uploaded
- Trained model
    - Each trained model was around 5 GB in size and hence couldn't be uploaded 