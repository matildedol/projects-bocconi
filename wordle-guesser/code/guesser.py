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
            # initiallize last_guess and mylist as class instances, to be used in the next iteration
            self.last_guess = None                                                  
            self.mylist = None

        def restart_game(self):
            self._tried = []
            # reset the list to the initial complete one and last_guess to None (for 1st iteration)
            self.last_guess=None 
            self.mylist=self.word_list.copy()                                       

        # compute entropy word by word, given a list (to compute patterns)
        def compute_entropy(self, word, list):                                         
            # create list of possible patterns
            patterns=[]                                                             
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

            # create dict with pairs pattern:abs_freq
            patterns_freq={}                                                        
            for pattern in patterns:
                if pattern not in patterns_freq:
                    patterns_freq[pattern]=1
                else:
                    patterns_freq[pattern]+=1
                
            entropy = 0
            tot=len(patterns)
            for pattern in patterns_freq:
                # p(x) = prob of pattern x to happen
                p = patterns_freq[pattern]/tot                                      
                # compute entropy of each word as the expected info that its patterns give (weighted by probabilities)
                entropy += - (p)*math.log2(p)                                       
            return entropy
        
        # function build the dictionary with pairs word:entropy
        def build_entropydict(self, list):                                          
            entropy_dict={}
            for word in list:
                entropy_dict[word]= self.compute_entropy(word, list)
            return entropy_dict

        def choose_nextguess(self, list):
            # build the dictionary in each filtered list at each guess
            entropy_dict=self.build_entropydict(list)                               
            # get the next guess as the one that maximises the information we obtain (highest entropy)
            next_guess =  max(entropy_dict, key=entropy_dict.get)                  
            
            return next_guess
        
        # build function to compute the most frequent letter in each position (avg guesses per game will decrease!) (not used)
        def choose_firstguess(self, list):                                          
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
                # in the first iteration of each game, determinstic choice of word
                if not self.last_guess:                                             
                    # I chose a word that probably maximizes information (3b1b)
                    guess='tanes'                                                   
                    self._tried.append(guess)
                    self.console.print(guess)
                    self.last_guess = guess  
                    return guess

                count_letters={}
                # misplaced and correct are recomputed each time, but ok! cause I filter the list with words that have those letters
                correct=re.findall(r'[a-z]', result)
                misplaced=[]                                                                                                 
                for idx, (letter,outcome) in enumerate(zip(self.last_guess, result)):
                    # keep count of parsed letters (could be done with Counter() faster, but ok)
                    if letter not in count_letters:
                        count_letters[letter]=1
                    else:
                        count_letters[letter]+=1                                                                            

                    ## if '+'
                    if outcome == '+':                                                                                      
                        # then reduce count by 1
                        count_letters[letter]-=1                                                                            
                        # if '+' and correct, keep only words with correct number of occurrences of that letter
                        if letter in correct and letter not in misplaced:                                                   
                            self.mylist=[word for word in self.mylist if word.count(letter)==correct.count(letter)]         
                        # if '+' and correct and misaplced, ignore (neglegible cost in avg guesses, word with 3 occurences of same letter are rare)
                        elif letter in correct and letter in misplaced:                                                     
                            pass
                        # if '+' and possibly in misplaced, keep only words with correct number of occurences of the letter
                        else:
                            self.mylist=[word for word in self.mylist if word.count(letter)==count_letters[letter]]         
                    
                    ## if '-', keep words with that letter in a different position 
                    elif outcome == '-':                                                                                    
                        misplaced.append(letter)
                        # I don't use the case of double or triple '-', but nvm
                        self.mylist=[word for word in self.mylist if word[idx]!=letter and letter in word]                  
                    
                    ## if 'correct', keep only words with that letter in that position 
                    else:
                        self.mylist=[word for word in self.mylist if word[idx]==letter]                                     

                ## choose next guess based on entropy from mylist, which will be filtered
                guess = self.choose_nextguess(self.mylist)                                                                  
                ## update last guess
                self.last_guess = guess                                                                                     
                self._tried.append(guess)
                self.console.print(guess)

            return guess