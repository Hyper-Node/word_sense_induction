# word_sense_induction
Implementation(s) of word sense induction (WSI) method(s)

## Description
At the moment the code in this repository implements a word sense induction method from Brody and Lapata, 
which is described in detail in a [2009 paper](https://dl.acm.org/citation.cfm?id=1609078), called Bayesian Word Sense Induction. 

The implementation covers the base 'Bayesian sense induction model' introduced on page 105 in the paper on figure 1. 

This is created to understand the base concepts of WSI, it does not aim to cover a full implementation of the method 
introduced in the paper necessarily. 

Basically it creates several Latent Dirichlet Allocation (LDA) topics for the contexts around a certain word. 
These sets of associated words are indicators for the senses the word is used. 

## Data (Corpora) used
For creating the LDA-model, a corpus with 1 million german words in form of full sentences from Uni-Leipzig (2011-mixed typical)
was used. This corpus and corpora of several other languages can be found [here](http://wortschatz.uni-leipzig.de/en/download/).
Format description on this corpus can be found [here](Format description: http://pcai056.informatik.uni-leipzig.de/downloads/corpora/Format_Download_File-eng.pdf)

For later evaluation with decision on the meaning (WSD) an pre-annotated set of german words from Uni-Heidelberg can be used, 
it can be found [here](http://projects.cl.uni-heidelberg.de/dewsd/files.shtml#gold)

Several Word Sense Disambiguation Tasks where done as part of SemEval-Workshop, which is probably also a good source 
for annotated corpora. 

## Basic Workflow
1. Sentences which contain the relevant word are obtained, in this sentences a context window is applied and special chars are filtered
2. LDA Model is created with the sentences as input 
3. Detected Top-10 associated words for each sense are logged to output 

## Future Improvements
- implement more then one feature layer and combine the results, as depicted in figure 2 page 106 in the paper
- implement evaluation method with f-scores and annotated set
- implement automated parameter optimization
- implement a dictionary lookup for the obtained senses 
- only use nouns and relevant words for context?