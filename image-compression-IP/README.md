This project consists in designing a program to compress an image by recoloring it with only k colors. 

The file **recolor.py** implements Lloyd's heuristic algorithm to solve the k-means problem. It outputs a recolored image with k colors.
Usage: 

`python3 recolor.py {input-image} {output-image} {k}`

Then, an IP model is designed for the compressing problem, to solve it exactly with glpk. 
You can find a detailed descirption of the model in IPmodel.pdf. The file **datfile_script.py** creates the .dat file of data inputs for the problem, while **kmeans.mod** specifies the model. 

They should be implemented as:

`python3 datfile_script.py {input-image} {k} {dat-file-path}`

and then to solve the problem with glpk:

`glpsol --model {.mod file} --data {.dat file} --output {optional output file}`


