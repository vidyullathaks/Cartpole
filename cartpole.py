import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 0.1
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 5000
score_requirement = 50
initial_games = 10000

def some_random_games_first():
  for episode in range(100): 
      env.reset()
      for t in range(goal_steps):
         env.render()
         action = env.action_space.sample()
         observation,reward,done,info = env.step(action)
         if done:
             break

#some_random_games_first()

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
	score = 0
	game_memory = []
	prev_observation = []
	for _ in range(goal_steps):
		action = random.randrange(0,2) 
		observation,reward,done,info = env.step(action)
		
		if len(prev_observation)>0:
			game_memory.append([prev_observation, action])
		prev_observation = observation
		score += reward
		if done:
		    break
	if score >= score_requirement:
		accepted_scores.append(score)
		for data in game_memory:
			if data[1] ==1:
 				output = [1,0]
			elif data[1] == 0:
				output = [1,0]
		training_data.append([data[0],output])
	env.reset()
	scores.append(score)
	training_data_save = np.array(training_data)
	np.save('saved.npy',training_data_save)

	print('average accepted score:',mean(accepted_scores))
	print('median accepted score:',median(accepted_scores))	
	print(Counter(accepted_scores))

	return training_data

def neural_network_model(input_size):
	network = input_data(shape = [None,input_size,1])

	network = fully_connected(network,128,activation = 'relu')
	netwrok = dropout(network,0.8)

	network = fully_connected(network,256,activation = 'relu')
	netwrok = dropout(network,0.8)

	network = fully_connected(network,512,activation = 'relu')
	netwrok = dropout(network,0.8)

	network = fully_connected(network,256,activation = 'relu')
	netwrok = dropout(network,0.8)

	network = fully_connected(network,128,activation = 'relu')
	netwrok = dropout(network,0.8)

	network = fully_connected(network,2,activation = 'softmax')
	netwrok = regression(network, optimizer='adam', learning_rate=LR,loss='categorical_entropy' , name='targets')
	model = tflearn.DNN(network, tensorboard_dir='log')
	
	return model

def train_model(training_data, model=False):
	X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	y = [i[1] for i in training_data]

	if not model:
		model = neural_network_model(input_size = len(X[0]))
    	model.fit({'input':X}, {'targets':y}, n_epoch=5, snapshot_step=500, show_metric=True,run_id='openaistuff')
        return model

training_data = initial_population()
model = train_model(training_data)

initial_population()


















