from random import choice
import yaml
from rich.console import Console
import re
# from collections import Counter
# from wordle import Wordle
# from game import Game


class Guesser:
    '''
        INSTRUCTIONS: This function should return your next guess. 
        Currently it picks a random word from wordlist and returns that.
        You will need to parse the output from Wordle:
        - If your guess contains that character in a different position, Wordle will return a '-' in that position.
        - If your guess does not contain that character at all, Wordle will return a '+' in that position.
        - If you guesses the character placement correctly, Wordle will return the character. 

        You CANNOT just get the word from the Wordle class, obviously :)
    '''

    '''
        Notes (by Mati):
        game.GUESSES and game.RESULTS in game.py store the success rates for all the games you play! 
    '''
    def __init__(self, manual):
        self.word_list = yaml.load(open('dev_wordlist.yaml'), Loader=yaml.FullLoader)
        self._manual = manual
        self.console = Console()
        self._tried = []
        self.misplaced = set()
        self.last_guess = None
        self.mylist = None

    def restart_game(self):
        self._tried = []
        self.last_guess=None 
        self.mylist=self.word_list.copy()                                               # restore the initial complete list, last guess, and misplaced list
        self.misplaced = set()

    #def compute_follletter(self, letter):
        # function to compute the most freq word after a letter 
    def get_guess(self, result):
        '''
        This function must return your guess as a string. 
        '''
        if self._manual=='manual':
            return self.console.input('Your guess:\n')
        else:
            '''
            CHANGE CODE HERE
            '''
            count_letters={}
            print('parsed letters are:', count_letters)

            if not self.last_guess:   
                guess='mates'
                self._tried.append(guess)
                self.console.print(guess)
                self.last_guess = guess  
                ### misplaced_last = []
                return guess
                
            correct=re.findall(r'[a-z]', result)                                                                    # this is recomputed in each iteration, but not significantly inefficient and logic works (I always keep correct words)
            print('misplaced letters are:', self.misplaced)
            print('my last guess is', self.last_guess)
            for idx, (letter,outcome) in enumerate(zip(self.last_guess, result)):
                if letter not in count_letters:                                                                     ## count and store parsed letters (could use Counter() but this works efficiently)
                    count_letters[letter]=1
                else:
                    count_letters[letter]+=1
                    
                if outcome == '+':                                                                                  ## now filter the list depending on the result
                    count_letters[letter]-=1                                                                        # decrease count, I don't want it
                    if letter in correct and letter not in self.misplaced:
                        self.mylist=[word for word in self.mylist if word.count(letter)==correct.count(letter)]     # for '+' and letter only in correct, keep words that have only the correct num of that letter
                    elif letter in correct and letter in self.misplaced:                                            # for '+' and letter in correct and in misplaced, ignore (neglegible cost of avg guesses) (1)
                        pass
                    else:
                        self.mylist=[word for word in self.mylist if word.count(letter)==count_letters[letter]]     # for '+' and letter only in misplaced or none, I use the count of correct occurrences of the letter (not addressing all specific cases, but not necessary!)

                elif outcome == '-':                                                                                # for '-', keep words with that letter not in that index (doesn't depend on whether letter is appearing for the first time or not) + update misplaced list for (1)
                    self.misplaced.add(letter)
                    self.mylist=[word for word in self.mylist if word[idx]!=letter and letter in word]

                else:                                                                                               # for 'a', keep only words with that letter in that index 
                    self.mylist=[word for word in self.mylist if word[idx]==letter]     
                # print('my list is', self.mylist[:10])
                #print('my dict is', count_letters,'\n')
            guess = choice(self.mylist)
            self.last_guess = guess
            self._tried.append(guess)
            self.console.print(guess)
            ### misplaced=misplaced_last
            return guess
        
'''
    the problem is I want to call my function with the result and also with my last guess as argument. 

'''

    # def game(self, wordle, guesser):
    #     endgame = False
    #     guesses = 0
    #     result = '+++++'                                        # initializes result at +++++
    #     while not endgame:
    #         guess = guesser.get_guess(result)                   # TAKES THE GUESS FROM GUESSER !!                    
    #         guesses += 1
    #         result, endgame = wordle.check_guess(guess)
    #         print(result)                                       # prints the resulting string from wordle.check_guess
    #     return result, guesses

'''
    so what I want to do in guesser is: 
    # 1 start from a random word (for now). 
    # 2 run check_guess
    # 3 parse the result from check guess
    # 4 parse my word
    # 5 for every - in result, take the corresponding letter and filter list for words that have it
    # 6 for every + in result, take the corresponding letter and filter list for words that don't have it 
    # 7 for every a-z in result, filter list with only words that have it in that position
    # 8 pick another word at random from the filtered list
    # 9 return it as 'guess' 
    # 10 repeat from 2 until endgame False

'''