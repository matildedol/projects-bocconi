import yaml
'''
Program to compute the most frequent letter in each position (avg guesses per game will decrease!)
Could be used for a first deterministic guess
'''

def choose_firstguess(list):                                  
    firstword=[]
    # secondword=[]
    for i in range(5):
        letters_list=[]
        for word in list:
            letters_list.append(word[i])
        mydict={}
        letters_set=set(letters_list)
        for letter in letters_set:
            mydict[letter]=letters_list.count(letter)
        best_letter=max(mydict, key=mydict.get)
        mydict.pop(best_letter)
        firstword.append(best_letter)
        # secondword.append(max(mydict, key=mydict.get))

    firstguess=''.join(firstword)
    # secondguess=''.join(secondword)
    return firstguess

mylist = yaml.load(open('dev_wordlist.yaml'), Loader=yaml.FullLoader)
print('the best word is \n', choose_firstguess(mylist))