# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: By definition, constraint propagatino uses local constraints in a space to reduce the search space.  In the naked twins approach, we search for pairs of squares in the same unit which both have the same two possible digits, say A5:26 and A6:26.  We can reason that 2 and 6 must be in A5 and A6, although we don't know which goes where.  However, knowing this allows us to constrain the search space elsewhere, by elminating 2 and 6 from every other square in the unit.  In this way, we've used a local constraint (2 and 6 must be in either A5 or A6) to constrain a related search space (the unit containing the naked twins).'

# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: The diagonal sudoku problem adds an additional two units which must adhere to the Sudoku rule, where all 9 digits (1-9) must appear, and every square must take a different digit.  Establishing these local contraints for the two diagonals adds an additional constraint (Sudoku unit) for each of the diagonal boxes.  Each and every one of these boxes (A1, B2, C3, D4, E5, F6, G7, H8, I9, A9, B8, C7, D6, E5, F4, G3, H2, I1) must now be checked against 4 separate boxes: row, column, square and diagonal.  By adding a contraint, the number of choices for the diagonal boxes will be constrained, and the overall search space decreased.

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solutions.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the ```assign_values``` function provided in solution.py

### Data

The data consists of a text file of diagonal sudokus for you to solve.
