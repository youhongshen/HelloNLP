from string import punctuation

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize
from nltk.collocations import *

# download the dictionary
from nltk.wsd import lesk

nltk.download('punkt')

text = 'mary had a little lamb. Her fleece was white as snow.'

# tokenize
sentences = sent_tokenize(text)
print sentences

words = nltk.word_tokenize(text)
print words

# remove stop words

nltk.download('stopwords')
custom_stop_words = set(stopwords.words('english') + list(punctuation))

words_no_stopword = [word for word in words if word not in custom_stop_words]
print words_no_stopword

# identify bigrams
# bigram is subset of n-gram
# where n-gram is multiple words that occur together that has a diff meaning
# than its components
# for example, new york

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = nltk.BigramCollocationFinder.from_words(words_no_stopword)
s = sorted(finder.ngram_fd.items())
# this prints out each of the bigrams along with its occurrences
# I suppose we identify bigrams by the number of occurrences of the diff bigrams
print s

# stemming and part of speech tagging
text2 = 'mary closed on closing night when she was in the mood to close'

stemmer = nltk.LancasterStemmer()
stemmed_words = [stemmer.stem(word) for word in nltk.word_tokenize(text2)]
print stemmed_words

nltk.download('averaged_perceptron_tagger')
pos = nltk.pos_tag(nltk.word_tokenize(text2))
print pos

# disambiguation word meaning
nltk.download('wordnet')

# diff meaning of a word
for ss in wordnet.synsets('bass'):
    print(ss, ss.definition())

# disambiguation
sensel = lesk(nltk.word_tokenize('sing in a lower tone, along with the '
                                 'bass'), 'bass')
print(sensel, sensel.definition())
sensel = lesk(nltk.word_tokenize('this bass is really hard to catch'), 'bass')
print(sensel, sensel.definition())

