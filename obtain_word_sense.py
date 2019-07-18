"""
This is a simple demo obtaining word senses (word sense inference)
with LDA (Latent Dirichlet Allocation)

Later on also Word Sense inference maybe with benchmark: http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark

Improvements:
- calculate scores
- adapt all parameters (when score calculator there, think of automatic adaption)
- remove some parts of speech in dataset which not seem relevant for recognition
- additional features implementation (instead of only using the words as such)
- implement gibbs sampler
"""



import re

import gensim
from gensim.models import Word2Vec

from sd_core.configuration_handler import ConfigurationHandler
from sd_core.conditional_print import ConditionalPrint

CODED_CONFIGURATION_PATH = './configurations/word_sense_detector.conf'
config_handler = ConfigurationHandler(first_init=True, fill_unkown_args=True,  \
                                      coded_configuration_paths=[CODED_CONFIGURATION_PATH])
config = config_handler.get_config()
cpr = ConditionalPrint(config.PRINT_MAIN, config.PRINT_EXCEPTION_LEVEL, config.PRINT_WARNING_LEVEL,
                       leading_tag="main")


# evaluation dataset is "DeWSD - Resources for German WSD" from University of Heidelberg, German dataset with annotated senses
# it can be obtained here: http://projects.cl.uni-heidelberg.de/dewsd/files.shtml#gold
from sd_core.text_file_loader import TextFileLoader
import os.path as path
text_file_loader = TextFileLoader()

# load input information
all_filepaths = text_file_loader.load_multiple_txt_paths_from_folder(config.INPUT_FILE_FOLDER_PATH)
overview_file = text_file_loader.load_txt_text(path.join(config.INPUT_FILE_FOLDER_PATH,"goldstandardSenses.txt"))
noun_filepaths = text_file_loader.load_multiple_txt_paths_from_folder(config.INPUT_FILE_FOLDER_PATH, ".n.txt")
verb_filepaths = text_file_loader.load_multiple_txt_paths_from_folder(config.INPUT_FILE_FOLDER_PATH, ".v.txt")
adjective_filepaths = text_file_loader.load_multiple_txt_paths_from_folder(config.INPUT_FILE_FOLDER_PATH, ".a.txt")
# extract the words to look in corpus

# Dataset for the clustering of non-annotated words is taken from "Leipzig Copora Collection"
# Link: http://wortschatz.uni-leipzig.de/en/download/
# Format description: http://pcai056.informatik.uni-leipzig.de/downloads/corpora/Format_Download_File-eng.pdf
# taken dataset is german from mixed sources from 2011 with 100K words

# load word list
# Filename: *_words.txt
# Format: Word_ID Word Word Frequency
# the 'filtered' word list was created manually and contains only the relevant words, which are in evaluation dataset
USE_FILTER= True
#word_filter = ["arbeiten"]  # only example sentences with these words get used
# word_filter = ["Autorität","Autoritäten"]       # has 2 senses in annotated set
word_filter = ["Schaltung", "Schaltungen"]        # has 3 senses in annotated set
NUMBER_OF_SENSES = 3
SIZE_CONTEXT_WINDOW = 3 #5     # was tested in paper 5 to 10 (+- window)
FILTER_SPECIAL_CHARS = True # filter , . etc only alphanumerical chars in final sentences


word_list_filtered = text_file_loader.load_txt_text(path.join(config.INPUT_FILE_FOLDER_PATH_2,"deu_mixed-typical_2011_1M-words_filtered.txt"), encoding="utf-8").split("\n")

parsed_words = {}
for line in word_list_filtered:
    line_split = line.split('\t')
    if line_split is None or len(line_split) <= 2: continue
    word_id, word, word2, word_freq = tuple(line_split)
    if USE_FILTER:
        if word in word_filter:
            parsed_words[word_id] = (word, word_freq)
    else:
        parsed_words[word_id] = (word, word_freq)

# Inverted list
# The file contains information about the occurrences of words in sentences (and optional theirposition in the sentence).
# Filename: *_inv_w.txtFormat: Word_ID Sentence_ID (Position_in_Sentence)
inverted_list = text_file_loader.load_txt_text(path.join(config.INPUT_FILE_FOLDER_PATH_2, "deu_mixed-typical_2011_1M-inv_w.txt"), encoding="utf-8").split("\n")

used_sentence_ids = {}
for line in inverted_list:
    line_split = line.split('\t')
    if line_split is None or len(line_split) <= 2: continue
    word_id, sentence_id, pos_in_sentence = tuple(line_split)
    # if the sentence has a word from the selected ones note this sentence
    if word_id in parsed_words.keys():
        used_sentence_ids[sentence_id] = True

# From the filtered words list and the inverted list obtain sentences to evaluate ( can be big file)
sentence_list = text_file_loader.load_txt_text(path.join(config.INPUT_FILE_FOLDER_PATH_2,"deu_mixed-typical_2011_1M-sentences.txt"), encoding="utf-8").split("\n")

def filter_special_chars(sentence):
    result = re.findall("\w+", sentence)
    text = " ".join(result)
    return text

def apply_context_window(split_sentence, words_to_recognize, window_size):
    found_word_index = -1
    # search for word
    for word_index, word in enumerate(split_sentence):
        if word in words_to_recognize:
            found_word_index = word_index

    # returns in special case word to recognize is not found
    if USE_FILTER:
        if found_word_index == -1:
            return []
    else:
        return split_sentence

    # remove follow up words to recognized word
    words_with_cut_end = split_sentence[:found_word_index+1+window_size]

    # remove starting words
    start_index = found_word_index-window_size
    if start_index < 0:
        start_index = 0

    words_start_end = words_with_cut_end[start_index:]

    final_split = words_start_end[:]  # make copy of list for safety
    return final_split



used_sentences = []
# used_sentences_lengths = []
for line in sentence_list:
    line_split = line.split('\t')
    if line_split is None or len(line_split) < 2: continue
    sentence_id, sentence = tuple(line_split)
    if sentence_id in used_sentence_ids.keys():
        if FILTER_SPECIAL_CHARS:
            sentence = filter_special_chars(sentence)
        len_sentence = len(sentence)
        split_sentence = sentence.split(' ')
        split_sentence_cw = apply_context_window(split_sentence, word_filter, SIZE_CONTEXT_WINDOW)
        used_sentences.append(split_sentence_cw)
        # used_sentences_lengths.append(used_sentences_lengths)

cpr.print("Sentences for clustering obtained:", len(used_sentences))


# data is availibe now, pre-processing data



# get set for testing
used_filepaths = all_filepaths[:6] # all mapped paths
all_eval_texts = []
for path in used_filepaths:
    text_lines = text_file_loader.load_txt_text(path, encoding='utf-8').split('\n')
    for line in text_lines:
        text_wo_tag = re.sub(r"#\d{1,3}", "", line) # replace the annotation
        all_eval_texts.append(text_wo_tag.split(' '))


from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
# Create a corpus from a list of texts
sentences_dictionary = Dictionary(used_sentences)
sentences_corpus = [sentences_dictionary.doc2bow(text) for text in used_sentences]
# Create LDA model
lda = gensim.models.ldamodel.LdaModel(sentences_corpus, num_topics=NUMBER_OF_SENSES)
for topic_id in range(0, NUMBER_OF_SENSES):
    cpr.print("Topic:", topic_id, "--------------")
    word_ids = lda.show_topic(topic_id, topn=10)  # gets 10 most significant word for topic
    topic_words = []
    for rep in word_ids:
        my_dict_id = int(rep[0])
        word = sentences_dictionary[my_dict_id]
        topic_words.append(word)

    cpr.print(topic_words)

# Create a second evaluation corpus
eval_corpus = [sentences_dictionary.doc2bow(text) for text in all_eval_texts]

exit()

unseen_doc = eval_corpus[0]
vector = lda[unseen_doc]


lda_model = gensim.models.ldamodel.LdaModel(corpus=sentences_corpus,
                                           id2word=id2word,
                                           num_topics=4,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='symmetric',
                                            iterations=100,
                                           per_word_topics=True)

# Parameters for LDA                  # description paper                / description common lda

# N_c = max(used_sentences_lengths)-1   # number of words around ambigoous / Number of words in document
print("sasd")


"""Possible features 
Our experiments used a feature
set designed to capture both immediate local context,
wider context and syntactic context. Specifically,
we experimented with six feature categories:
±10-word window (10w), 
±5-word window (5w),
collocations (1w), 
word n-grams (ng), 
part-ofspeech n-grams (pg)
dependency relations (dp)."""


"""
alpha values ranging from 0.005 to 1.
beta parameter was set to 0.1 (in all layers).
The Gibbs sampler was run for 2,000 iterations.  LDA with Gibbs availible here: https://radimrehurek.com/gensim/models/wrappers/ldamallet.html
Number of Senses: same number of senses for all the words, since tuning this number individually for each word would be prohibitive. We experimented with values ranging from three to nine senses.
"""

# clustering step



# evaluation step


# test create a dirichlet prior over samples
import numpy as np

s = np.random.dirichlet((10, 5, 3), 20).transpose()
import matplotlib.pyplot as plt
plt.barh(range(20), s[0])
plt.barh(range(20), s[1], left=s[0], color='g')
plt.barh(range(20), s[2], left=s[0]+s[1], color='r')
plt.title("Lengths of Strings")
#plt.show()
