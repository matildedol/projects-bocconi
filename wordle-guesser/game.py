from wordle import Wordle
from guesser import Guesser
import argparse
import time
import sys
import os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class Game:
    def __init__(self):
        self.RESULTS = []  # was the word guessed? values: true or false 
        self.GUESSES = []  # number of guesses per game

    def score(self, result, guesses):
        if '-' in result or '+' in result:
            # word was not guessed
            result = False
        else:
            result = True
        # checks if I guessed, it will take as input the result of wordle.check_guess
        self.RESULTS.append(result)                             
        self.GUESSES.append(guesses)

    def game(self, wordle, guesser):
        endgame = False
        guesses = 0
        # initializes result at +++++
        result = '+++++'                                       
        while not endgame:
            print('playing game...')
            # TAKES THE GUESS FROM GUESSER !!     
            guess = guesser.get_guess(result)                                  
            print('processing guess...')
            guesses += 1
            result, endgame = wordle.check_guess(guess)
            # prints the resulting string from wordle.check_guess
            print(result)                                       
        return result, guesses

# bla bla setting up stuff... 
def main():                                                     
    # set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=int, help='number of rounds to play')
    parser.add_argument('--p', action='store_true', help='whether to print the game or not')

    # parse command line arguments
    args = parser.parse_args()

    # Create game instance
    game = Game()

    # if --r is set, play the game automatically for r rounds
    if args.r:
        start_time = time.time() # log start time
        
        # set up game
        wordle = Wordle()
        guesser = Guesser('console')
        
        # block print if --p is not set
        if not args.p:
            blockPrint()

        for run in range(args.r):
            # reset game
            guesser.restart_game()
            wordle.restart_game()
            print('game restarted, #',run+1)

            # play the game
            results, guesses = game.game(wordle, guesser)

            # score the game
            # take the num of guesses counted in game(), use it to count guesses each time
            game.score(results, guesses)
            print('scores updated, #',run+1)

        # print('game done!')
        enablePrint()
        # print('game done!')

        # print results if --p is set
        if args.p:            
            print(f"Proportion of words guessed correctly: {game.RESULTS.count(True)/len(game.RESULTS):.2%}")
            print(f"Average number of guesses: {sum(game.GUESSES)/len(game.GUESSES):.4f} ")
            print(f"Total execution time: {time.time() - start_time:.2f} seconds")
        # else print results in csv format
        else:
            print(f"{game.RESULTS.count(True)/len(game.RESULTS):.2%},{sum(game.GUESSES)/len(game.GUESSES):.4f},{time.time() - start_time:.2f}")

    # if --r is not set, play the game manually
    else:
        # set up game
        guesser = Guesser('manual')
        wordle = Wordle()

        # play the game
        print('Welcome! Let\'s play wordle! ')
        game.game(wordle, guesser)

if __name__ == '__main__':
    main()