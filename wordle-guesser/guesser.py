from random import choice
import yaml
from rich.console import Console
import re
from collections import Counter
import math


class Guesser:
        '''
        How to use: just substitute dev_wordlist.yaml below with the desired dataset.
        '''
        def __init__(self, manual):
            self.word_list = yaml.load(open('dev_wordlist.yaml'), Loader=yaml.FullLoader)
            self._manual = manual
            self.console = Console()
            self._tried = []
            self.last_guess = None                                                  # initiallize last_guess and mylist as class instances, to be used in the next iteration
            self.mylist = None

        def restart_game(self):
            self._tried = []
            self.last_guess=None 
            self.mylist=self.word_list.copy()                                       # reset the list to the initial complete one and last_guess to None (for 1st iteration)

        def compute_entropy(self, word, list):                                      # compute entropy word by word, given a list (to compute patterns)   
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

            patterns_freq={}                                                        # create dict with pairs pattern:abs_freq
            for pattern in patterns:
                if pattern not in patterns_freq:
                    patterns_freq[pattern]=1
                else:
                    patterns_freq[pattern]+=1
                
            entropy = 0
            tot=len(patterns)
            for pattern in patterns_freq:
                p = patterns_freq[pattern]/tot                                      # p(x) = prob of pattern x to happen
                entropy += - (p)*math.log2(p)                                       # compute entropy of each word as the expected info that its patterns give (weighted by probabilities)
            
            return entropy
        
        def build_entropydict(self, list):                                          # function build the dictionary with pairs word:entropy
            entropy_dict={}
            for word in list:
                entropy_dict[word]= self.compute_entropy(word, list)
            return entropy_dict

        def choose_nextguess(self, list):
            entropy_dict=self.build_entropydict(list)                               # build the dictionary in each filtered list at each guess
            next_guess =  max(entropy_dict, key=entropy_dict.get)                   # get the next guess as the one that maximises the information we obtain (highest entropy)
            return next_guess
        
        def choose_firstguess(self, list):                                          # build function to compute the most frequent letter in each position (avg guesses per game will decrease!) (not used, 'tanes' is better on many test sets)
            firstword=[]
            for i in range(4):
                mydict={}
                for word in list:
                    if word[i] in mydict:
                        mydict[word[i]]=1
                    else:
                        mydict[word[i]]+=1
                best_letter=max(mydict, key=mydict.get)
                firstword.append(best_letter)
            firstguess=''.join(firstword)
            return firstguess
        
        def get_guess(self, result):

            if self._manual=='manual':
                return self.console.input('Your guess:\n')
            
            else:
                if not self.last_guess:                                             # in the first iteration of each game, determinstic choice of word
                    guess='tanes'                                                   # I chose a word that probably maximizes information (3b1b)
                    self._tried.append(guess)
                    self.console.print(guess)
                    self.last_guess = guess  
                    return guess

                count_letters={}
                correct=re.findall(r'[a-z]', result)
                misplaced=[]                                                                                                # these are recomputed each time, but ok! cause I filter the list with words that have correct and misaplced letter 
                for idx, (letter,outcome) in enumerate(zip(self.last_guess, result)):

                    if letter not in count_letters:
                        count_letters[letter]=1
                    else:
                        count_letters[letter]+=1                                                                            # keep count of parsed letters (could be done with Counter() faster, but ok)

                    if outcome == '+':                                                                                      ## if '+'
                        count_letters[letter]-=1                                                                            # then reduce count by 1
                        if letter in correct and letter not in misplaced:                                                   
                            self.mylist=[word for word in self.mylist if word.count(letter)==correct.count(letter)]         # if '+' and correct, keep only words with correct number of occurrences of that letter
                        elif letter in correct and letter in misplaced:                                                     # if '+' and correct and misaplced, ignore (neglegible cost in avg guesses, word with 3 occurences of same letter are rare)
                            pass
                        else:
                            self.mylist=[word for word in self.mylist if word.count(letter)==count_letters[letter]]         # if '+' and possibly in misplaced, keep only words with correct number of occurences of the letter
                    
                    elif outcome == '-':                                                                                    ## if '-', keep words with that letter in a different position
                        misplaced.append(letter)
                        self.mylist=[word for word in self.mylist if word[idx]!=letter and letter in word]                  # I don't use the case of double or triple '-', but nvm
                    
                    else:
                        self.mylist=[word for word in self.mylist if word[idx]==letter]                                     ## if 'correct', keep only words with that letter in that position 

                guess = self.choose_nextguess(self.mylist)                                                                  ## choose next guess based on entropy from mylist, which will be filtered
                self.last_guess = guess                                                                                     ## update last guess
                self._tried.append(guess)
                self.console.print(guess)

            return guess