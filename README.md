# Optimization Algorithms Project

This project uses three optimization algorithms sourced from the `mlrose` Python library to solve various optimization problems. The first part of the project applies random hill climbing, simulated annealing, and genetic algorithm to neural networks based on the heart disease dataset imported from Kaggle. The second part of the project uses two different optimization problems: the Queens problem and the traveling salesperson problem.

## Prerequisites

To run this code, you must have the following installed on your computer:

- Python
- sklearn library
- pandas library
- numpy library
- mlrose library

## Running the code

1. Clone or download the repository to your computer.
2. Open the project in your code editor.
3. Run the desired script (each file is named after the learning algorithm implemented in it).
4. When prompted, input the file path of your data set (note that this only works for CSV files and you should not use quotation marks when inputting the CSV path).

## Usage

When you run one of the scripts, you will be prompted to input the file path of your data set. The script will then apply the selected optimization algorithm to solve the specified problem. The results will be collected and plotted using the `mathplotlib` library to show training accuracy vs iterations and test accuracy vs iterations.

For more detailed information on how each algorithm and problem is implemented, please refer to the comments in the code and the `mlrose` documentation.
