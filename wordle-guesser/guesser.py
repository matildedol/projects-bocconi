from random import choice
import yaml
from rich.console import Console
import re
from collections import Counter
import math
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
            self.last_guess = None
            self.mylist=None

        def restart_game(self):
            self._tried = []
            self.last_guess=None 
            self.mylist=self.word_list.copy()                                       # take the initial complete list

        def compute_entropy(self, word, list):                                            # compute entropy word by word (convenient in a 500 words list, as each word divides set on half and first one is determined)    
            patterns=[]                                                             # create list of possible patterns
            for vs_word in list:
                counts = Counter(vs_word)
                pattern=[]
                for i, letter in enumerate(word):
                    if word[i] == vs_word[i]:
                        pattern+=word[i]
                        counts[word[i]]-=1
                    else:
                        pattern+='+'
                for i, letter in enumerate(word):
                    if word[i] != vs_word[i] and word[i] in vs_word:
                        if counts[word[i]]>0:
                            counts[word[i]]-=1
                            pattern[i]='-'
                patterns.append(''.join(pattern))

            patterns_freq={}                                                        # create dict with pattern:abs_freq
            for pattern in patterns:
                if pattern not in patterns_freq:
                    patterns_freq[pattern]=1
                else:
                    patterns_freq[pattern]+=1
                
            entropy = 0
            tot=len(patterns)
            for pattern in patterns_freq:
                p = patterns_freq[pattern]/tot
                entropy += - (p)*math.log2(p)
            
            return entropy
        
        def choose_nextguess(self, list):
            entropy_dict={}
            for word in list:
                entropy_dict[word]= self.compute_entropy(word, list)
            next_guess =  max(entropy_dict, key=entropy_dict.get)
            return next_guess

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
                if not self.last_guess:
                    guess='tanes'
                    self._tried.append(guess)
                    self.console.print(guess)
                    self.last_guess = guess  
                    return guess
                    
                count_letters={}
                correct=re.findall(r'[a-z]', result)
                #print('correct letters are:', correct)
                misplaced=[]                                                                                                # this is restored each time, but ok! cause I only keep words with that letter 
                for idx, (letter,outcome) in enumerate(zip(self.last_guess, result)):

                    if letter not in count_letters:
                        count_letters[letter]=1
                    else:
                        count_letters[letter]+=1                                                                            # can be done with counter faster

                    if outcome == '+':                                                                                      # then you want to remove all words with that letter
                        count_letters[letter]-=1
                        if letter in correct and letter not in misplaced:
                            self.mylist=[word for word in self.mylist if word.count(letter)==correct.count(letter)]
                        elif letter in correct and letter in misplaced:
                            pass
                        else:
                            self.mylist=[word for word in self.mylist if word.count(letter)==count_letters[letter]]
                    elif outcome == '-':
                        misplaced.append(letter)
                        self.mylist=[word for word in self.mylist if word[idx]!=letter and letter in word]                  # I don't use the case of double or triple '-', but nvm
                    else:
                        self.mylist=[word for word in self.mylist if word[idx]==letter]                                     # here the list gets modified, in each loop, then with a new game we start from the if above and the list goes abck to being the complete one
                    #print('my list is', self.mylist[:10])
                    #print('my dict is', count_letters,'\n')
                guess = self.choose_nextguess(self.mylist)
                self.last_guess = guess
                self._tried.append(guess)
                self.console.print(guess)
            return guess