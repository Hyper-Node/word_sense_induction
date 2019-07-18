"""
Some simple methods for text processing
"""
import re

class TextProcessing(object):

     @staticmethod
     def filter_special_chars(sentence):
         """
         Removes special characters from a sentence
         :param sentence:
         :return:
         """
         result = re.findall("\w+", sentence)
         text = " ".join(result)
         return text

     @staticmethod
     def apply_context_window(split_sentence, words_to_recognize, window_size, use_filter=True):
         """
         Applies a context window around a word in a sentence and gives back
         an array of the words which are within the context windows
         :param split_sentence:
         :param words_to_recognize:
         :param window_size:
         :return:
         """

         found_word_index = -1
         # search for word
         for word_index, word in enumerate(split_sentence):
             if word in words_to_recognize:
                 found_word_index = word_index

         # returns in special case word to recognize is not found
         if use_filter:
             if found_word_index == -1:
                 return []
         else:
             return split_sentence


         # remove follow up words to recognized word
         words_with_cut_end = split_sentence[:found_word_index + 1 + window_size]

         # remove starting words
         start_index = found_word_index - window_size
         if start_index < 0:
             start_index = 0

         words_start_end = words_with_cut_end[start_index:]

         final_split = words_start_end[:]  # make copy of list for safety
         return final_split