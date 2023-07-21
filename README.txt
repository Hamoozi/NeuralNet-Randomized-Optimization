Hamza Wahhas
4/3/23




All 3 algorithms used in this project have been sourced from the mlrose python library. Functions including how to use the algorithms on certain optimzation problems have also been inspired by mlrose.


Directions to run code.

In order to run the code in any of the files, the user must have python installed along with the sklearn library, pandas library, numpy library and mlrose library.

Each of the files is named after the learning algorithm implemented on it. 
When the user runs the code, they'll be prompted to input the file path of their data set. Note this is only works
for CSV files (NO QUOTATION MARKS WHEN INPUTTING CSV PATH).

In the first part of the project I use random hill climbing, simulated annealing and genetic algorithm for neural networks based off the heart disease dataset imported from kaggle (https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

Used a loop for all neural networks in order to track data at all iterations in one pass rather than creating an individual loop for each one. Once the results are collected, we can use the mathplotlib library to plot our results on a graph of training accuracy vs iterations and test accuracy vs iterations.

For the second part of the project I use two different optimzation problems. For the Queens problem I was guided by mlrose documentation (https://mlrose.readthedocs.io/en/stable/source/tutorial1.html). For the traveling salesperson problem I was guided by another mlrose doc (https://mlrose.readthedocs.io/en/stable/source/tutorial2.html). Note that are used are different from the documentaiton in order to preform best, more details in analsys.

Follows same path as the first part for collecting data and plotting, but added time in order to further investigate results using the time library.
