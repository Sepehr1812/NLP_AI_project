In this project, Unigram and Bigram modeling with back-off smoothing are implemented.

At first, train text file is read and unigram and bigram models will be saved.
Then sentences of test file will be examined with train dataset and its catagory will be predicted.
At last, precision, recall and f-measure parameters will be printed.

In this project back-off smoothing is implemented; that works with conditional probability (by Bayes law) and two coefficiant: landa and 1 - landa.

Train an dtest files are news sentences in Persian and the category of each sentence is written at the first of that sentence before '@' characters.
