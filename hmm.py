import numpy as np
from hidden_markov import hmm

states = ('s', 't')
possible_observation = ('A','B' )

start_probability = np.matrix( '0.5 0.5 ')
transition_probability = np.matrix('0.6 0.4 ;  0.3 0.7 ')
emission_probability = np.matrix( '0.3 0.7 ; 0.4 0.6 ' )

# Initialize class object
test = hmm(states,possible_observation,start_probability,transition_probability, emission_probability)

obs1 = ('A', 'B','B','A')
obs2 = ('B', 'A','B')
obs3 = ('A', 'B', 'B')

observations = []
observations.extend( [obs1,obs2,obs3] )

quantities_observations = [10, 20,10]
num_iter=1000

e,t,s = test.train_hmm(observations,num_iter,quantities_observations)

print(e)
print("\n")
print(t)
print("\n")
print (s)
# e,t,s contain new emission transition and start probabilities
