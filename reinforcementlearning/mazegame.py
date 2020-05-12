import numpy as np

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

MATRIX_SIZE = 4

maze = [[0, 1, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 0, 1],
		[1, 0, 2, 0]]

M = np.zeros(shape = (MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE))

for row in range(0, MATRIX_SIZE):
	for column in range(0, MATRIX_SIZE):
			if column == 0:
				M[row, column, LEFT] = -1
			elif column == MATRIX_SIZE - 1:
				M[row, column, RIGHT] = -1
			
			if column != MATRIX_SIZE - 1 and maze[row][column + 1] == 2:
				M[row, column, RIGHT] = 100
			if column != 0 and maze[row][column - 1] == 2:
				M[row, column, LEFT] = 100
			
			if row != MATRIX_SIZE - 1 and maze[row + 1][column] == 2:
				M[row, column, DOWN] = 100
			if row != 0 and maze[row - 1][column] == 2:
				M[row, column, UP] = 100
			
			if row == 0:
				M[row, column, UP] = -1
			elif row == MATRIX_SIZE - 1:
				M[row, column, DOWN] = -1

print("Values of the M Table:")
print(M)
print()

Q = np.zeros(shape = (MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE))

currentstate = (0, 0)

def available_actions(state):
    current_actions = M[state[0], state[1]]
    available_action = np.where(current_actions >= 0)[0]
    return available_action

available_action = available_actions(currentstate)
#print(available_action)

# Chooses one of the available actions at random
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_action, 1))
    return next_action


#action = sample_next_action(available_action)

def getnewstates(state, action):
	if action == LEFT:
		return (state[0], state[1] - 1)
	if action == RIGHT:
		return (state[0], state[1] + 1)
	if action == UP:
		return (state[0] - 1, state[1])
	if action == DOWN:
		return (state[0] + 1, state[1])

def update(state, action, gamma):
	#Get the max_indexes of the action-th row of Q
	newstates = getnewstates(state, action)
	max_index = np.where(Q[newstates[0], newstates[1]] == np.max(Q[newstates[0], newstates[1]]))[0]
	#If there is more than one max_index, choose a random one
	if max_index.shape[0] > 1:
		max_index = int(np.random.choice(max_index, size = 1))
	else:
		max_index = int(max_index)
	#Gets the max value for the max_index
	max_value = Q[newstates[0], newstates[1], max_index]

	Q[state[0], state[1], action] = M[state[0], state[1], action] + gamma * max_value

	if (np.max(Q) > 0):
		return(np.sum(Q / np.max(Q)*100))
	else:
		return (0)

def getrandomstate():
	value = 2
	while value == 2:
		x = np.random.randint(0, MATRIX_SIZE)
		y = np.random.randint(0, MATRIX_SIZE)
		value = maze[x][y]
	return (x, y)

scores = []
for i in range(1000):
    current_state = getrandomstate()
    available_action = available_actions(current_state)
    action = sample_next_action(available_action)
    score = update(current_state, action, 0.75)
    scores.append(score)

print("Values of the Q Table:")
print(Q)
print()

print("Values of the score")
print(scores)
"""
# Determines the available actions for a given state (which node it can jump to)
def available_actions(state):
	#gets the state-th row of M
	x = state[0]
	y = state[1]
	maze[0, 0]
	return available_action

# Chooses one of the available actions at random
def sample_next_action(available_actions_range):
	next_action = int(np.random.choice(available_action, 1))
	return next_action
"""