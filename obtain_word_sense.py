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

from sd_core.configuration_handler import ConfigurationHandler
from sd_core.conditional_print import ConditionalPrint
from sd_core.text_file_loader import TextFileLoader
from my_lib.leipzig_corpus_handler import LeipzigCorpusHandler
from gensim.corpora.dictionary import Dictionary
import os.path as path
import re
import gensim

CODED_CONFIGURATION_PATH = './configurations/word_sense_detector.conf'
config_handler = ConfigurationHandler(first_init=True, fill_unkown_args=True,  \
                                      coded_configuration_paths=[CODED_CONFIGURATION_PATH])
config = config_handler.get_config()
cpr = ConditionalPrint(config.PRINT_MAIN, config.PRINT_EXCEPTION_LEVEL, config.PRINT_WARNING_LEVEL,
                       leading_tag="main")

# base config ------------------------------
USE_FILTER = True
# WORD_FILTER = ["arbeiten"]
# WORD_FILTER = ["Autorität","Autoritäten"]       # has 2 senses in annotated set
WORD_FILTER = ["Schaltung", "Schaltungen"]        # has 3 senses in annotated set
NUMBER_OF_SENSES = 3
SIZE_CONTEXT_WINDOW = 3                           #5     # was tested in paper 5 to 10 (+- window)
FILTER_SPECIAL_CHARS = True                       # filter , . etc only alphanumerical chars in final sentences
PRE_FILTERED_WORD_ID_LIST = False                 # there can be a pre-filtered file if necessry, which allows to filter uneccessary idioms or combined words (like in verbs: verb is halten, but also festhalten is recognized)
BASE_FILENAME = "deu_mixed-typical_2011_1M"

# base initializations
text_file_loader = TextFileLoader()
leipzig_ch = LeipzigCorpusHandler()

# obtain the id's of the observed words in the corpus
words_with_ids = leipzig_ch.get_word_ids(BASE_FILENAME, use_word_filter=USE_FILTER, word_filter=WORD_FILTER, use_prefiltered_list=PRE_FILTERED_WORD_ID_LIST)
sentence_ids = leipzig_ch.get_sentence_ids_for_words(BASE_FILENAME,words_with_ids)

# From the filtered words list and the inverted list obtain sentences to evaluate ( can be big file)
sentence_list = leipzig_ch.get_all_sentences(base_name=BASE_FILENAME)
# with the sentence_ids filter the sentence_list for only relevant sentence with the word,
# this also applies context window and sc-filter already
used_sentences = leipzig_ch.get_sentences_with_ids(BASE_FILENAME, sentence_list, sentence_ids, FILTER_SPECIAL_CHARS, True, WORD_FILTER, SIZE_CONTEXT_WINDOW)
cpr.print("Sentences for clustering obtained:", len(used_sentences))

# Create a corpus-object from a list of texts
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


exit()  # it follows unused stuff atm
# get set for evaluation

# load input information
all_filepaths = text_file_loader.load_multiple_txt_paths_from_folder(config.INPUT_FILE_FOLDER_PATH)
overview_file = text_file_loader.load_txt_text(path.join(config.INPUT_FILE_FOLDER_PATH,"goldstandardSenses.txt"))
noun_filepaths = text_file_loader.load_multiple_txt_paths_from_folder(config.INPUT_FILE_FOLDER_PATH, ".n.txt")
verb_filepaths = text_file_loader.load_multiple_txt_paths_from_folder(config.INPUT_FILE_FOLDER_PATH, ".v.txt")
adjective_filepaths = text_file_loader.load_multiple_txt_paths_from_folder(config.INPUT_FILE_FOLDER_PATH, ".a.txt")


used_filepaths = all_filepaths[:6] # all mapped paths
all_eval_texts = []
for path in used_filepaths:
    text_lines = text_file_loader.load_txt_text(path, encoding='utf-8').split('\n')
    for line in text_lines:
        text_wo_tag = re.sub(r"#\d{1,3}", "", line) # replace the annotation
        all_eval_texts.append(text_wo_tag.split(' '))

# Create a second evaluation corpus
eval_corpus = [sentences_dictionary.doc2bow(text) for text in all_eval_texts]


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
