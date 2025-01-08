set COLORS;  
set CENTERS; 

param k;      
param dist{CENTERS}; 
param z{COLORS,CENTERS};

var y{j in CENTERS}, binary;  # 1 if center j is selected, 0 otherwise

minimize dissimilarity: 
    sum {j in CENTERS} dist[j] * y[j];

s.t. num_centers: 
    sum {j in CENTERS} y[j] = k;

s.t. unique_assignment {i in COLORS}:
    sum {j in CENTERS} z[i, j] * y[j] = 1;

solve;
end;
