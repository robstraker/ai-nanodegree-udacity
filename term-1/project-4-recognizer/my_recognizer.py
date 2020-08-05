import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    
    for word_id in range(test_set.num_items):
    	best_score, best_guess_word = float("-inf"), None
    	probability_dict = {}
    	
    	X, lengths = test_set.get_item_Xlengths(word_id)
    	for word, model in models.items():
    	    try:
    	        probability_dict[word] = model.score(X, lengths)    
    	    except:
    	        probability_dict[word] = float("-inf")
    	        
    	    if probability_dict[word] > best_score:
    	        best_score, best_guess_word = probability_dict[word], word
    	        
    	probabilities.append(probability_dict)
    	guesses.append(best_guess_word)
    	    
    return probabilities, guesses
