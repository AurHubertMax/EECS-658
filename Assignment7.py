# -*- coding: utf-8 -*-
"""Assignment7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1APwMx5uLeopJ9_eP0u4Dt7_tvC_TLfI-
"""

'''
Name of program: Assignment 7
Author: Aureliano Hubert Maximus
Creation Date: Nov 21 2023
Description:
  This program uses Reinforcement Learning to implement policy and value iteration on a given grid, where
  part 1 uses policy iteration to calculate V(s) and plot the results, and part 2 uses value iteration to
  calculate V(s) and plot the results
Collaborators:
Prof. David Johnson (EECS 658 Slides)
ChatGPT
'''
import matplotlib.pyplot as plt
import numpy as np

#Initialize grids
gridWithTerm = [[1,0,0,0,0], #need this grid to identify terminals so the terminals wouldnt be changed
              [0,0,0,0,0],
              [0,0,0,0,0],
              [0,0,0,0,0],
              [0,0,0,0,1]]

gridStart = [[0,0,0,0,0], #initial grid
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]]

#Print the content of the grids by row
def printGridResult(grid):
  for row in range(len(grid)):
    row = [round(value, 2) if value != 1 else value for value in grid[row]]
    print(row)
  print("\n")

#Used in Part 1, to calculate V(s) policy
def countV(grid, row, col):
  if gridWithTerm[row][col] == 1: #check if cell is at a terminal, if it is, return 0
    return 0

  else:
    if row == 0: # if its at the first row of the grid
      up = grid[row][col]
    else:
      up = grid[row - 1][col]

    if col == len(grid[0]) - 1: # if its at the right side of the grid
      right = grid[row][col]
    else:
      right = grid[row][col + 1]

    if row == len(grid) - 1: # if its at the bottom row
      down = grid[row][col]
    else:
      down = grid[row + 1][col]

    if col == 0: # if its at the left side of the grid
      left = grid[row][col]
    else:
      left = grid[row][col - 1]

    V = -1 + (0.25*up) + (0.25*right) + (0.25*down) + (0.25*left) #V(s) value stored here
    return V

#Used in Part 2, to calculate V(s) with value iteration
def countValueV(grid, row, col):
  reward = -1
  gamma = 1
  if gridWithTerm[row][col] == 1: #check if cell is at a terminal, if it is, return 0
    return 0

  else:
    if row == 0: # if its at the first row of the grid
      up = grid[row][col]
    else:
      up = grid[row - 1][col]

    if col == len(grid[0]) - 1: # if its at the right side of the grid
      right = grid[row][col]
    else:
      right = grid[row][col + 1]

    if row == len(grid) - 1: # if its at the bottom row
      down = grid[row][col]
    else:
      down = grid[row + 1][col]

    if col == 0: # if its at the left side of the grid
      left = grid[row][col]
    else:
      left = grid[row][col - 1]

    up = reward + gamma * up
    right = reward + gamma * right
    down = reward + gamma * down
    left = reward + gamma * left

    return max(up, right, down, right)


#Calculates the difference between 2 grids, returns the amount of difference
def gridDiff(grid1, grid2):
  differences = 0

  for row in range(len(grid1)):
    for col in range(len(grid1[0])):
      differences += abs(grid1[row][col] - grid2[row][col])

  return differences

#Empties the grid
def emptyGrid(row, col):
  newGrid = [[0 for _ in range(col)] for _ in range(row)]
  return newGrid

#Loops through all the cells in the grid, and calls functions that count their V(s), returns the updated grid
def updateGrid(grid, option):
  row, col = len(grid), len(grid[0])
  updatedGrid = emptyGrid(row, col)

  if (option == 1): #calls the countV if opttion is 1
    for r in range(row):
      for c in range(col):
        updatedGrid[r][c] = countV(grid, r, c)

  elif (option == 2): #calls the countValueV if option is 2
    for r in range(row):
      for c in range(col):
        updatedGrid[r][c] = countValueV(grid, r, c)


  return updatedGrid

#Plots the error vs iteration
def plotter(iteration, difference, epsilon):
  plt.plot(np.arange(0, iteration, 1), difference)
  plt.plot(np.arange(0, iteration, 1), [epsilon] * iteration, label='Epsilon')

  plt.xlabel("Iteration")
  plt.ylabel("Error")
  plt.title("Error vs Iteration")

  plt.legend()
  plt.show()

#Main function to handle everything
def iterateGrids(option):
  #initialize previous and next grids, and extra values
  prevGrid = gridStart.copy()
  nextGrid = gridStart.copy()
  epsilon = 1e-6
  iteration = 1
  difference = []

  if (option == 1):
    print("================ RL Policy Iteration Algorithm ================")

  else:
    print("================ RL Value Iteration Algorithm ================")

  while True:
    nextGrid = updateGrid(prevGrid, option) #update grid using policy iteration
    currentDiff = gridDiff(prevGrid, nextGrid) #counts the differences between current and next grid
    difference.append(currentDiff) #combine the differences to a list

    #break loop if difference is below epsilon
    if currentDiff < epsilon:
      print("i: ", iteration, "------------------")
      break

    #print policy array for the first 10 iterations
    if iteration >= 1 and iteration <= 10:
      print("i:", iteration, "------------------")
      print("policy array:")
      printGridResult(nextGrid)

    #update previous grid and increment iteration
    prevGrid = nextGrid.copy()
    iteration += 1

  #print policy array when it converges
  print("policy array:")
  printGridResult(nextGrid)
  plotter(iteration, difference, epsilon)
  print("\n\n")

# PART 1: RL Policy Iteration Algorithm
iterateGrids(1)


# PART 2: RL Value Iteration Algorithm
iterateGrids(2)