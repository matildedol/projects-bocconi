from random import choice, seed
import yaml
from collections import Counter

class Wordle():
    
    global ALLOWED_GUESSES, word_list
    ALLOWED_GUESSES = 6
    word_list = yaml.load(open('dev_wordlist.yaml'), Loader=yaml.FullLoader)
    
    # comment this out for development, use for testing / marking
    # seed(42)

    def __init__(self, random_state=None):
        # picks random word from list
        self._word = choice(word_list)                  
        # stores tried words
        self._tried = []                                

    # restarts game each time
    def restart_game(self, random_state=None):          
        self._word = choice(word_list)
        self._tried = []
        self._endgame = False 

    # Produces the feedback string, taking my guess as input
    def get_matches(self, guess):                       
        counts = Counter(self._word)
        results = []
        for i, letter in enumerate(guess):
            if guess[i] == self._word[i]:
                results+=guess[i]
                counts[guess[i]]-=1
            else:
                results+='+'
        for i, letter in enumerate(guess):
            if guess[i] != self._word[i] and guess[i] in self._word:
                if counts[guess[i]]>0:
                    counts[guess[i]]-=1
                    results[i]='-'

        return ''.join(results)

    # takes a word, my guess, as input
    def check_guess(self, guess):                       
        # initializes 'result' which will be my result, a sequence of +|-|a-z
        result = False                                  
        # initializes end_game to check when the game ends
        end_game = False                                
        guess = guess.lower().strip()

        # check guesses are valid
        if not guess.isalpha():                         
            return "Please enter only letters", False
        if len(guess) != 5:
            return "Please enter a five-letter word", False
        elif guess in self._tried:
            return "You have already tried that word", False
        else:
            # append guess to 'tried' if valid
            self._tried.append(guess)                   
            if guess == self._word:
                # ends the game if I guessed
                end_game = True                         
                # set 'result' equal to my word, I guessed
                result = self._word                     
                print('Congratulations, you guessed the word!')
            else:
                # set 'result' equal to the result I got (+|-|a-z)
                result = self.get_matches(guess)        
                if len(self._tried) == ALLOWED_GUESSES:                 
                    print(f'Sorry, you did not guess the word. The word was {self._word}', )
                    # once we get to the num of possible guesses, end the game
                    end_game = True                     
        return result, end_game
