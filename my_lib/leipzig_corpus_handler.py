"""
Base methods for handling corpus
from Uni-Leipzig
"""
from sd_core.conditional_print import ConditionalPrint
from sd_core.configuration_handler import ConfigurationHandler
from sd_core.text_file_loader import TextFileLoader
from my_lib.text_processing import TextProcessing

import os.path as path


class LeipzigCorpusHandler(object):
    def __init__(self):
        config_handler = ConfigurationHandler(first_init=False)

        self.config = config_handler.get_config()
        self.cpr = ConditionalPrint(self.config.PRINT_LEIPZIG_CORPUS_HANDLER, self.config.PRINT_EXCEPTION_LEVEL,
                                    self.config.PRINT_WARNING_LEVEL)
        self.tfl = TextFileLoader()


    def get_word_ids(self, base_name, use_word_filter=False, word_filter=None, use_prefiltered_list=False):

        # load word list
        # Filename: *_words.txt
        # Format: Word_ID Word Word Frequency
        # the 'filtered' word list was created manually and contains only the relevant words, which are in evaluation dataset

        if use_prefiltered_list:
            # pre-filtered word id list
            word_id_list = self.tfl.load_txt_text(
                path.join(self.config.INPUT_FILE_FOLDER_PATH_2, base_name+"-words_filtered.txt"),
                encoding="utf-8").split("\n")
        else:
            # use unfiltered word word id list
            word_id_list = self.tfl.load_txt_text(
                path.join(self.config.INPUT_FILE_FOLDER_PATH_2, base_name+"-words.txt"),
                encoding="utf-8").split("\n")

        parsed_words = {}
        for line in word_id_list:
            line_split = line.split('\t')
            if line_split is None or len(line_split) <= 2: continue
            word_id, word, word2, word_freq = tuple(line_split)
            if use_word_filter:
                if word in word_filter:
                    parsed_words[word_id] = (word, word_freq)
            else:
                parsed_words[word_id] = (word, word_freq)

        return parsed_words

    def get_sentence_ids_for_words(self, base_name, words_with_ids):
        # Inverted list
        # The file contains information about the occurrences of words in sentences (and optional theirposition in the sentence).
        # Filename: *_inv_w.txtFormat: Word_ID Sentence_ID (Position_in_Sentence)
        inverted_list = self.tfl.load_txt_text(
            path.join(self.config.INPUT_FILE_FOLDER_PATH_2, base_name+"-inv_w.txt"), encoding="utf-8").split(
            "\n")

        used_sentence_ids = {}
        for line in inverted_list:
            line_split = line.split('\t')
            if line_split is None or len(line_split) <= 2: continue
            word_id, sentence_id, pos_in_sentence = tuple(line_split)
            # if the sentence has a word from the selected ones note this sentence
            if word_id in words_with_ids.keys():
                used_sentence_ids[sentence_id] = True

        return used_sentence_ids

    def get_all_sentences(self, base_name):
        return self.tfl.load_txt_text(
            path.join(self.config.INPUT_FILE_FOLDER_PATH_2, base_name+"-sentences.txt"),
            encoding="utf-8").split("\n")

    def get_sentences_with_ids(self, base_name, sentence_list, used_sentence_ids, filter_special_chars,
                               apply_context_window, word_filter, size_context_window):
        used_sentences = []
        # used_sentences_lengths = []
        for line in sentence_list:
            line_split = line.split('\t')
            if line_split is None or len(line_split) < 2: continue
            sentence_id, sentence = tuple(line_split)
            if sentence_id in used_sentence_ids.keys():
                if filter_special_chars:
                    sentence = TextProcessing.filter_special_chars(sentence)
                len_sentence = len(sentence)
                split_sentence = sentence.split(' ')
                if apply_context_window:
                    split_sentence_cw = TextProcessing.apply_context_window(split_sentence, word_filter, size_context_window)
                    used_sentences.append(split_sentence_cw)
                else:
                    used_sentences.append(split_sentence)
                # used_sentences_lengths.append(used_sentences_lengths)
        return used_sentences
