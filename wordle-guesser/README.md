This projects consists in producing a guesser for a cousin of the game Wordle. You can play the game [here](https://www.nytimes.com/games/wordle/index.html).

How to use:

The file **guesser.py**, what *I* worked on, is the guesser. It implements entropy analysis of the words from a given list, based on the patterns of answers they produce.  

The file **game.py** is runs the game. The file **wordle.py** implements it. Patterns are given with either '+', if the letter is not in the word, '-' if the letter is there but at a different position, or the letter itself if you guessed it. There are no words "not in the list".   

To run the game, use:

`python3 game.py --r R`

where R is the number of games you want to play. Add a `--p` option to get stats on the guesser performance. 