import numpy as np

def checkifcurrentlythree(state):
	a = []
	for i in range(0, 9):
		a.append(((state >> i) & 0b000000001) == 1)
	
	return (a[0] and a[1] and a[2]) or (a[3] and a[4] and a[5]) or (a[6] and a[7] and a[8]) or (a[0] and a[3] and a[6]) or (a[1] and a[4] and a[7]) or (a[2] and a[5] and a[8]) or (a[2] and a[5] and a[8]) or (a[0] and a[4] and a[8]) or (a[2] and a[4] and a[6])
	
def checkifthree(state, action):
	nstate = state | (1 << action)
	return checkifcurrentlythree(nstate)

STATE_NUMBER = 512

M = np.zeros(shape = (STATE_NUMBER, 9))

for i in range(0, STATE_NUMBER):
	for j in range(0, 9):
		a = (i >> j) & 0b000000001

		if a == 0b000000001:
			M[i, j] = -1
		else:
			if checkifthree(i, j):
				M[i, j] = 100


print(M)
Q = np.zeros(shape = (STATE_NUMBER, 9))

gamma = 0.75
initial_state = 1

# Determines the available actions for a given state (which node it can jump to)
def available_actions(state):
	#gets the state-th row of M
	current_state_row = M[state]
	available_action = np.where(current_state_row >= 0)[0]
	return available_action

#actions = available_actions(initial_state)
#print(actions)

# Chooses one of the available actions at random
def sample_next_action(available_actions_range):
	next_action = int(np.random.choice(available_actions_range, 1))
	return next_action


#action = sample_next_action(actions)
#print(action)

def update(state, action, gamma):
	newstate = state | (1 << action)

	#Get the max_indexes of the action-th row of Q
	max_index = np.where(Q[newstate, ] == np.max(Q[newstate, ]))[0]
	#If there is more than one max_index, choose a random one
	if max_index.shape[0] > 1:
		max_index = int(np.random.choice(max_index, size = 1))
	else:
		max_index = int(max_index)
	
	#Gets the max value for the max_index
	max_value = Q[newstate, max_index]
	Q[state, action] = M[state, action] + gamma * max_value
	if (np.max(Q) > 0):
		return(np.sum(Q / np.max(Q)*100))
	else:
		return (0)

# Updates the Q-Matrix according to the path chosen
#update(initial_state, action, gamma)

scores = []
for i in range(10000):
	current_state = np.random.randint(0, 512)
	available_action = available_actions(current_state)

	if available_action.size != 0:
		action = sample_next_action(available_action)
		score = update(current_state, action, gamma)
		scores.append(score)

current_state = 0
steps = []

print("Trained Q matrix:")
print(Q)

while not checkifcurrentlythree(current_state):
	#Get the max_indexes of the action-th row of Q
	max_index = np.where(Q[current_state, ] == np.max(Q[current_state, ]))[0]
	#If there is more than one max_index, choose a random one
	if max_index.shape[0] > 1:
		max_index = int(np.random.choice(max_index, size = 1))
	else:
		max_index = int(max_index)
	
	current_state = current_state | (1 << max_index)
	steps.append(current_state)

for i in steps:
	for j in range(0, 9):
		if(j % 3 == 0):
			print("|", end='')

		x = 'o' if ((i >> j) & 0b000000001) == 1 else ' '
		print(x, end="|")
		if((j + 1) % 3 == 0):
			print()
	print()