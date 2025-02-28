import yaml
'''
Program to compute the most frequent letter in each position (avg guesses per game will decrease!)
Could be used for a first deterministic guess
Uses a-priori logic of setting 's' as the last letter, and having all different letters
'''

def choose_firstguess(list):                                  
    firstword=[]
    # secondword=[]
    for i in range(4):
        ## get all letters at position i
        letters_list=[]
        for word in list:
            letters_list.append(word[i])
        ## create dict with pairs letter:abs_freq
        mydict={}
        # take a set with single occurrences
        letters_set=set(letters_list)
        for letter in letters_set:
            if letter not in firstword and letter!='s':
                mydict[letter]=letters_list.count(letter)
        best_letter=max(mydict, key=mydict.get)
        firstword.append(best_letter)
        # secondword.append(max(mydict, key=mydict.get))

    firstguess=''.join(firstword) + 's'     
    # secondguess=''.join(secondword)
    return firstguess

mylist = yaml.load(open('chat_sample.yaml'), Loader=yaml.FullLoader)
print('the best word is \n', choose_firstguess(mylist))