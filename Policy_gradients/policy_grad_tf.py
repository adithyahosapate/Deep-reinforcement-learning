import numpy as np
import tensorflow as tf
import gym

###Hyperparameters

H=100#Size of hidden layer
batch_size=10#no of episodes before param update
learning_rate=1e-4
gamma=0.99#discount factor for reward
decay_rate=0.99# decay factor for rms_prop
resume=False 
render=False

###############
sess=tf.Session()



input_dim1=80
input_dim2=80
D=input_dim1*input_dim2

input_image=tf.placeholder("float32,",shape=[None,D,1],name="placeholder_x")
model={}

w_1=tf.get_variable("w1", shape=[D, H],
					initializer=tf.contrib.layers.xavier_initializer())
w_2=tf.get_variable("w2", shape=[H, 1],
					initializer=tf.contrib.layers.xavier_initializer())





def forward_pass(x):
	l_1=tf.nn.relu(tf.matmul(x,w_1))#hidden_layer
	l_2=tf.matmul(l1,w_2)
	return (tf.nn.sigmoid(l_2),l_1)



optimizer=tf.train.AdamOptimizer(0.0001).minimize()



def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

grad_buffer = { k : np.zeros_like(v) for k,v in zip(model.keys(),model.values()) } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in zip(model.keys(),model.values()) }


def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # cropping 
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
	discounted_r=np.zeros_like(r)
	running_add=0
	for t in reversed(range(0,r.size)):
		if r[t]!=0:
			running_add=0
		running_add=running_add*gamma+r[t]
		discounted_r[t]=running_add
		return discounted_r	


env=gym.make("Pong-v0")
observation=env.reset()
prev_x=None
reward_sum=0
xs,hs,dlogps,drs = [],[],[],[]
episode_number=0

sess.run(tf.global_variables_initializer())


while True:
	if render: env.render()
	cur_x=prepro(observation)
	x=cur_x-prev_x if prev_x is not None else np.zeros(D)
	prev_x=cur_x
	p,h=forward_pass(h)
	action=2 if np.random.uniform < p else 3

	xs.append(x)#observation
	hs.append(h)
	yu=1 if action==2 else 0
	y = 1 if action == 2 else 0 # a "fake label"
  	dlogps.append(y - p)

  	observation, reward, done, info=env.step(action)
  	reward_sum+=reward
  	drs.append(reward)
  	if done:
  		episode_number+=1
  		epx = np.vstack(xs)
    	eph = np.vstack(hs)
    	epdlogp = np.vstack(dlogps)
    	epr = np.vstack(drs)
    	xs,hs,dlogps,drs = [],[],[],[] # reset array memory
    	discounted_epr = discount_rewards(epr)
    	# standardize the rewards to be unit normal (helps control the gradient estimator variance)
    	discounted_epr -= np.mean(discounted_epr)
    	discounted_epr /= np.std(discounted_epr)

    	epdlog*= discounted_epr
    	grad=



