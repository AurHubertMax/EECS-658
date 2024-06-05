'''
Name of program: Assignment 8
Author: Aureliano Hubert Maximus
Creation Date: Dec 7 2023
Description:
  This program uses RL Monte Carlo First Visit, RL Monte Carlo Every Visit, Q-Learning, SARSA-Learning, and epsilon-greedy algorithms
  on the same gridworld described in Assignment 8 instructions.
Collaborators:
Prof. David Johnson (EECS 658 Slides)
ChatGPT
'''

import matplotlib.pyplot as plt
import numpy as np
import sys

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

kGrid =    [[0,1,2,3,4], # grid filled with list of states
            [5,6,7,8,9],
            [10,11,12,13,14],
            [15,16,17,18,19],
            [20,21,22,23,0]]

def printGridResult(grid): # prints the grid
  for row in range(len(grid)):
      row_str = [str(value) if value is not None else ' ' for value in grid[row]]
      print(' '.join(row_str))
  print("\n")



def printFullResult(kCount, s, r, gamma, Gs, Ns, Ss, Vs): # prints the table
  printMonteArray(kCount, s, r, gamma, Gs)
  print("N(s): ")
  printGridResult(Ns)
  print("S(s): ")
  printGridResult(Ss)
  print("V(s): ")
  printGridResult(Vs)

def plotter(epoch, difference, epsilon): # plots the error vs epoch
  plt.plot(np.arange(0, epoch, 1), difference)
  plt.plot(np.arange(0, epoch, 1), [epsilon] * epoch, label='Epsilon')

  plt.xlabel("Epoch")
  plt.ylabel("Error")
  plt.title("Error vs Epoch")

  plt.legend()
  plt.show()


def printMonteArray(k, s, r, y, Gs): # prints the table required for monte first and monte every
  print("k   s    r   y   G(s)")
  print("----------------------")
  for i in range(k + 1):
    print(f"{i: <3} {s[i]: <3} {r[i]: <3} {y: <3}  {Gs[i]: <3}")
  print("\n")



def randomAction(): # returns a random action
  return np.random.choice(["left", "right", "up", "down"])

def gridDiff(grid1, grid2): # returns the absolute difference between two grids
  grid1Arr = np.array(grid1)
  grid2Arr = np.array(grid2)
  return np.max(np.abs(grid1Arr - grid2Arr))

# PART 1============================================================================

def findGs(kCount, r): # returns the values of Gs, given kCount and its reward
  gamma = 0.9
  Gs = []
  for i in range(kCount):
    Gt = 0
    discount = 1.0
    for t in range(i, kCount):
      Gt += discount * r[t]
      discount *= gamma
    Gs.append(round(Gt, 2))
  Gs.append(0.00)

  return Gs

def findSsFirst(kCount, Gs, s): # returns the S(s) table, given kCount, Gs, and s
  Ss = np.zeros_like(gridStart, dtype=float) # initialize S(s) grid to be all zeros of type float
  firstVisitDict = {}

  for i in range(kCount): # saves the index of a cell to make sure it doesnt calculate duplicates
      stateKey = s[i]
      if stateKey not in firstVisitDict:
        firstVisitDict[stateKey] = Gs[i]

  for i in range(5): # iterates through S(s) grid to add its values, based on the indexes saved above
    for k in range(5):
      stateKey = kGrid[i][k]
      if (stateKey in firstVisitDict) and gridWithTerm[i][k] != 1:
        Ss[i][k] += firstVisitDict[stateKey]

  Ss_rounded = np.round(Ss, 3)

  return Ss_rounded

def findVs(totalSs, totalNs): # returns the V(s) grid
  Vs =  np.divide(totalSs, totalNs, out=np.zeros_like(totalSs), where=totalNs != 0) # sets V(s) to be Ss/Ns
  Vs_rounded = np.round(Vs, 2)

  return Vs_rounded


def monteFirst(totalSs, totalNs): # monte carlo first visit algorithm
  Ss, Ns = np.zeros_like(totalSs, dtype=float), np.zeros_like(totalNs, dtype=int) # initialize Ss and Ns grids

  currentState = (np.random.randint(0, 4), np.random.randint(0, 4)) # sets a random current state

  kCount = 0
  gamma = 0.9

  Gs = []
  s = []
  r = []



  while True:
    action = randomAction()

    s.append(kGrid[currentState[0]][currentState[1]]) # appends the state number from kGrid to s, used to keep track of which states are visited
    r.append(-1) # sets the reward for to -1, since havent reached terminal



    if (Ns[currentState[0]][currentState[1]] == 0 and gridWithTerm[currentState[0]][currentState[1]] != 1): # increment N(s) by at most one each episode
      Ns[currentState[0]][currentState[1]] += 1


    # rules for each random action
    if action == "left":
      nextState = (currentState[0], max(0, currentState[1] - 1))

    elif action == "right":
      nextState = (currentState[0], min(4, currentState[1] + 1))

    elif action == "up":
      nextState = (max(0, currentState[0] - 1), currentState[1])

    elif action == "down":
      nextState = (min(4, currentState[0] + 1), currentState[1])



    if gridWithTerm[nextState[0]][nextState[1]] == 1: # if it reaches a terminal state
      currentState = nextState
      kCount += 1
      s.append('t') # append 't' to s
      r.append(0) # append 0 to r

      Gs = findGs(kCount, r) # find the Gs values
      Ss = findSsFirst(kCount, Gs, s) # find the Ss values

      totalNs += Ns
      totalSs += Ss

      Vs = findVs(totalSs, totalNs) # find the Vs values


      return totalNs, totalSs, Vs, Gs, kCount, s, r, gamma
      break

    currentState = nextState
    kCount += 1

def iterateMonteFirst(): # monte carlo first visit iterator
  print("================ RL First-Visit Monte Carlo Algorithm ================")
  # initialize all the grids needed
  epoch = 0
  prevSs = np.zeros_like(gridStart, dtype=float)
  nextSs = np.zeros_like(gridStart, dtype=float)
  prevNs = np.zeros_like(gridStart, dtype=int)
  nextNs = np.zeros_like(gridStart, dtype=int)
  nextVs = np.zeros_like(gridStart, dtype=float)
  prevVs = np.zeros_like(gridStart, dtype=float)
  epsilon = 1e-6
  difference = []
  while True:
    Ns, Ss, Vs, Gs, kCount, s, r, gamma = monteFirst(prevSs, prevNs)
    nextVs = findVs(Ss, Ns)
    nextSs = Ss
    nextNs = Ns
    currentDiff = gridDiff(prevVs, nextVs) # keeps track of the current error

    if currentDiff < epsilon and epoch > 10: # if Vs converged
      print(f"Converged at epoch: {epoch} ------------------")
      break

    if epoch == 0 or epoch == 1 or epoch == 10: # print the results if at epoch 0, 1, and 10
      print(f"epoch: {epoch} ------------------")
      printFullResult(kCount, s, r, gamma, Gs, Ns, Ss, Vs)
      prevVs = nextVs.copy()
      epoch += 1
      difference.append(currentDiff)

    prevVs = nextVs.copy()
    epoch += 1
    difference.append(currentDiff)

  printFullResult(kCount, s, r, gamma, Gs, Ns, Ss, Vs)
  plotter(epoch, difference, epsilon)

# PART 2============================================================================

def findSsEvery(kCount, Gs, s): # returns the S(s) table, given kCount, Gs, and s
    Ss = np.zeros_like(gridStart, dtype=float)

    for j in range(kCount): # similar to findSsFirst, but doesnt keep track of duplicates
      stateKey = s[j]
      for i in range(5):
        for k in range(5):
          if kGrid[i][k] == stateKey and gridWithTerm[i][k] != 1:
            Ss[i][k] += Gs[j]

    Ss_rounded = np.round(Ss, 3)

    return Ss_rounded

def monteEvery(totalSs, totalNs): # monte carlo every visit algorithm
  Ss, Ns = np.zeros_like(totalSs, dtype=float), np.zeros_like(totalNs, dtype=int)

  currentState = (np.random.randint(0, 4), np.random.randint(0, 4)) # sets current state to be a randomly chosen state

  kCount = 0
  gamma = 0.9

  Gs = []
  s = []
  r = []


  while True:
    action = randomAction()

    s.append(kGrid[currentState[0]][currentState[1]])
    r.append(-1)



    if (gridWithTerm[currentState[0]][currentState[1]] != 1): # increment N(s) by one each time it is visited
      Ns[currentState[0]][currentState[1]] += 1


    # rules for each action
    if action == "left":
      nextState = (currentState[0], max(0, currentState[1] - 1))

    elif action == "right":
      nextState = (currentState[0], min(4, currentState[1] + 1))

    elif action == "up":
      nextState = (max(0, currentState[0] - 1), currentState[1])

    elif action == "down":
      nextState = (min(4, currentState[0] + 1), currentState[1])



    if gridWithTerm[nextState[0]][nextState[1]] == 1: # if reached terminal state
      currentState = nextState
      kCount += 1
      s.append('t')
      r.append(0)

      Gs = findGs(kCount, r) # get Gs values
      Ss = findSsEvery(kCount, Gs, s) # get Ss values

      totalNs += Ns
      totalSs += Ss

      Vs = findVs(totalSs, totalNs) # get Vs values


      return totalNs, totalSs, Vs, Gs, kCount, s, r, gamma
      break

    currentState = nextState
    kCount += 1

def iterateMonteEvery(): # monte carlo every visit iterator
  print("================ RL Every-Visit Monte Carlo Algorithm ================")
  # initialize all the grids needed
  epoch = 0
  prevSs = np.zeros_like(gridStart, dtype=float)
  nextSs = np.zeros_like(gridStart, dtype=float)
  prevNs = np.zeros_like(gridStart, dtype=int)
  nextNs = np.zeros_like(gridStart, dtype=int)
  nextVs = np.zeros_like(gridStart, dtype=float)
  prevVs = np.zeros_like(gridStart, dtype=float)
  epsilon = 1e-6
  difference = []
  while True:
    Ns, Ss, Vs, Gs, kCount, s, r, gamma = monteEvery(prevSs, prevNs)
    nextVs = findVs(Ss, Ns)
    nextSs = Ss
    nextNs = Ns
    currentDiff = gridDiff(prevVs, nextVs) # keeps track of the current error

    if currentDiff < epsilon and epoch > 10: # if Vs converged, print the results
      print(f"Converged at epoch: {epoch} ------------------")
      break

    if epoch == 0 or epoch == 1 or epoch == 10: # if epoch is at 0, 1, or 10, print the results
      print(f"epoch: {epoch} ------------------")
      printFullResult(kCount, s, r, gamma, Gs, Ns, Ss, Vs)
      prevVs = nextVs.copy()
      epoch += 1
      difference.append(currentDiff)

    prevVs = nextVs.copy()
    epoch += 1
    difference.append(currentDiff)

  printFullResult(kCount, s, r, gamma, Gs, Ns, Ss, Vs)
  plotter(epoch, difference, epsilon)




iterateMonteFirst()
iterateMonteEvery()

import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import math

# HELPER FUNCTIONS
kGrid =    [[0,1,2,3,4],
            [5,6,7,8,9],
            [10,11,12,13,14],
            [15,16,17,18,19],
            [20,21,22,23,24]]

def randomState(): # returns a random state
  return np.random.randint(0, 25)

def printQGridResult(matrix, title=""): # prints the grid
  print(title)
  np.savetxt(sys.stdout, matrix, fmt='%7d', delimiter=" ", newline="\n")
  print("\n")

def plotter(epoch, difference, epsilon): # plots the graph
  plt.plot(np.arange(0, epoch, 1), difference)
  plt.plot(np.arange(0, epoch, 1), [epsilon] * epoch, label='Epsilon')

  plt.xlabel("Epoch")
  plt.ylabel("Error")
  plt.title("Error vs Epoch")

  plt.legend()
  plt.show()



def initializeR(R): # sets R to the the gridworld described in the assignment
  terminals = ((1, 0), (5, 0), (19, 24), (23, 24))
  goLeft = (0, -1)
  goRight = (0, 1)
  goUp = (-1, 0)
  goDown = (1, 0)

  possibleAction = []

  for i in range(5):
    for k in range(5):
      if 0 <= i + goLeft[0] < len(kGrid) and 0 <= k + goLeft[1] < len(kGrid): #if can go left
        state = kGrid[i][k]

        action = kGrid[i][k - 1]
        stateActions = (state, action)
        possibleAction.append(stateActions)

      if 0 <= i + goRight[0] < len(kGrid) and 0 <= k + goRight[1] < len(kGrid): #if can go right
        state = kGrid[i][k]

        action = kGrid[i][k + 1]
        stateActions = (state, action)
        possibleAction.append(stateActions)

      if 0 <= i + goUp[0] < len(kGrid) and 0 <= k + goUp[1] < len(kGrid): #if can go up
        state = kGrid[i][k]

        action = kGrid[i - 1][k]
        stateActions = (state, action)
        possibleAction.append(stateActions)

      if 0 <= i + goDown[0] < len(kGrid) and 0 <= k + goDown[1] < len(kGrid): #if can go down
        state = kGrid[i][k]

        action = kGrid[i + 1][k]
        stateActions = (state, action)
        possibleAction.append(stateActions)


  for i in range(25): # sets cell to terminal actions to be 100 and cell to its possible actions to be 0
    for k in range(25):

      if (i, k) in terminals:
        R[i][k] = 100


      elif (i, k) in possibleAction and (i, k) not in terminals:
        R[i][k] = 0

  return R

def findRandomAction(R, currentState): # returns a random action where its corresponding R is not -1
  nextAction = randomState()

  while R[currentState][nextAction] == -1:
    nextAction = randomState()
  return nextAction



def findQ(R, Q, state, action, gamma): # returns the Q value in the specific state and action
  maxState = []

  for i in range(25):
    if R[action][i] != -1:

      maxState.append(Q[action][i])

  maxQ = max(maxState)
  return R[state][action] + gamma * maxQ


def gridDiffQ(grid1, grid2): # returns the absolute difference between two grids
  grid1Arr = np.array(grid1)
  grid2Arr = np.array(grid2)
  return np.abs(grid1Arr - grid2Arr)

# PART 3============================================================================

def qLearning(R, Q):
  gamma = 0.9
  totalQ = Q

  currentState = randomState()
  currentAction = findRandomAction(R, currentState)
  nextAction = currentAction

  while R[currentState][currentAction] != 100: # if R at current state and current action is not 100, hasnt reached end of episode
    Q[currentState][currentAction] = findQ(R, Q, currentState, currentAction, gamma)

    currentState = currentAction
    currentAction = findRandomAction(R, currentState) # find a random action
  Q[currentState][currentAction] = findQ(R, Q, currentState, currentAction, gamma)


  return np.round(Q, 2)







# PART 4============================================================================

def findSARSAAction(R, currentState, Q):
  actions = []
  qActions = []
  maxQValue = 0


  for i in range(25):
    if R[currentState][i] == 0 or R[currentState][i] == 100:
      actions.append(i) #Actions where R is not -1

  for i in range(len(actions)):
    qActions.append(Q[currentState][actions[i]]) #value of Q cells stored here
    if maxQValue < Q[currentState][actions[i]]: # if there is a value in Q's cell that is greater than the current max value
      maxQValue = Q[currentState][actions[i]] # save the index of the max value
      k = i
      if i == len(actions) - 1: # if reach end of list of actions, return the max value saved at index k
        nextAction = actions[k]
        return nextAction

  nextAction = random.choice(actions) # if multiple max values are the same value, pick a random action

  return nextAction




def SARSA(R, Q):
  gamma = 0.9
  epsilon = 1e-6

  currentState = randomState()
  currentAction = findSARSAAction(R, currentState, Q)

  nextAction = currentAction


  while R[currentState][currentAction] != 100: # if R at current state and current action is not 100, hasnt reached end of episode

    Q[currentState][currentAction] = findQ(R, Q, currentState, currentAction, gamma)


    currentState = currentAction
    currentAction = findSARSAAction(R, currentState, Q) # find an action based on SARSA policy

  Q[currentState][currentAction] = findQ(R, Q, currentState, currentAction, gamma)

  return np.round(Q, 2)






# PART 5============================================================================
def decayEpsilon(epoch):
  c = 100
  e = math.e
  t = epoch

  return math.exp(-(t - 1) / c)

def uniformRandomNumber():

  return random.uniform(0, 1)

def greedy(R, Q, epsilonGreedy): # epsilon greedy algorithm episode iterator
  gamma = 0.9
  epsilon = 1e-6


  currentState = randomState() # set current state to a random state
  currentAction = findSARSAAction(R, currentState, Q) # set current action based on SARSA algorithm

  nextAction = currentAction # initialize current action

  while R[currentState][currentAction] != 100: # if R at current state and current action is not 100, keep iterating
    if uniformRandomNumber() < epsilonGreedy: # use q-learning if a uniform random number 0 < x < 1 is less than epsilon

      Q[currentState][currentAction] = findQ(R, Q, currentState, currentAction, gamma)

      currentState = currentAction
      currentAction = findRandomAction(R, currentState)

    else: # else, use SARSA
      Q[currentState][currentAction] = findQ(R, Q, currentState, currentAction, gamma)


      currentState = currentAction
      currentAction = findSARSAAction(R, currentState, Q)

  Q[currentState][currentAction] = findQ(R, Q, currentState, currentAction, gamma)

  return np.round(Q, 2)
# ITERATOR ============================================================================

def iterator(option):
  epoch = 0
  epsilon = 1e-6
  epsilonGreedy = 1
  difference = []
  diffCounter = 0


  Q = np.zeros((25, 25), dtype=int)
  R = np.full((25, 25), -1, dtype=int)
  R = initializeR(R)
  prevQ = Q
  nextQ = Q


  if option == 0: # Q-Learning ALgorithm
    print("================ RL Q-Learning Algorithm ================")


  elif option == 1: # SARSA Algorithm
    print("================ RL SARSA Algorithm ================")


  else: # Epsilon-Greedy Algorithm
    print("================ RL Epsilon-Greedy Algorithm ================")

  printQGridResult(R, "R Matrix")


  prevQ = Q
  nextQ = Q
  while True:

    if option == 0: # Q-Learning ALgorithm
      nextQ = qLearning(R, Q)

    elif option == 1: # SARSA Algorithm
      nextQ = SARSA(R, Q)

    else: # Epsilon-Greedy Algorithm
      nextQ = greedy(R, Q, epsilonGreedy)

    currentDiff = gridDiffQ(prevQ, nextQ)

    if epoch == 0 or epoch == 1 or epoch == 10: # print the Q matrix if it's at epoch 0, 1, or 10
      print(f"epoch: {epoch} ------------------")
      printQGridResult(nextQ, "Q Matrix")
      difference.append(np.max(currentDiff)) # append the absolute differences between the previous grid and the current grid
      prevQ = nextQ.copy()
      epsilonGreedy = decayEpsilon(epoch) # decay the epsilon for epsilon greedy algorithm
      epoch += 1

    if np.max(currentDiff) < epsilon: # if the differences(error) is less than epsilon, keep iterating
      diffCounter += 1
      difference.append(np.max(currentDiff))
      prevQ = nextQ.copy()
      epsilonGreedy = decayEpsilon(epoch)
      epoch += 1


    if diffCounter >= 10: # if the differences(error) is < epsilon 10 or more times, print the result
      print(f"Converged at epoch: {epoch} ------------------")
      printQGridResult(nextQ, "Q Matrix")
      break





    difference.append(np.max(currentDiff))
    prevQ = nextQ.copy()
    epsilonGreedy = decayEpsilon(epoch)
    epoch += 1


  plotter(epoch, difference, epsilon) # plot the error vs epoch graph
  print("\n\n")




iterator(0)
iterator(1)
iterator(2)