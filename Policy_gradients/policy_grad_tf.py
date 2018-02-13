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

input_vector=tf.placeholder("float32",shape=[None,D],name="placeholder_x")
labels=tf.placeholder("float32",shape=[1,None,1],name="placeholder_y")
advantages=tf.placeholder("float32",shape=[1,None,1],name="placeholder_adv")


model={}

w_1=tf.get_variable("w1", shape=[D, H],
					initializer=tf.contrib.layers.xavier_initializer())
w_2=tf.get_variable("w2", shape=[H,1],
					initializer=tf.contrib.layers.xavier_initializer())






layer_1=tf.nn.relu(tf.matmul(input_vector,w_1))#hidden_layer
layer_2=tf.nn.sigmoid(tf.matmul(layer_1,w_2))

loss_function=tf.reduce_mean(tf.multiply(advantages,tf.multiply(tf.log(layer_2),labels)
								+tf.multiply(tf.log(layer_2),1-labels)))
	

optimizer=tf.train.AdamOptimizer(0.0001).minimize(-loss_function)



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
labels_list=[]
xs,hs,dlogps,drs = [],[],[],[]
advantages_list=[]
x_list=[]
episode_number=0

sess.run(tf.global_variables_initializer())

counter=0

while True:

	if episode_number%5==0:
		render=True
	else:
		render=False

	if render: env.render()
	cur_x=prepro(observation)
	x=cur_x-prev_x if prev_x is not None else np.zeros(D)
	prev_x=cur_x

	p=sess.run([layer_2],feed_dict={input_vector:np.array([x,x])})
	x_list.append(np.array(x))
	action=2 if np.random.uniform() < p[0][0] else 3

	y = 1 if action == 2 else 0 # a "fake label"
	labels_list.append([y])
	counter=counter+1
	observation, reward, done, info=env.step(action)
	reward_sum+=reward
	drs.append(reward)
	if done:
		if reward_sum<0:
			advantage=-1
		else:
			advantage=1
		
		advantages_list.extend([[advantage]]*counter)
		counter=0
		observation = env.reset()
		if episode_number%5==0:
			_, loss, _ = sess.run([layer_2, loss_function, optimizer],
                                                    {input_vector:np.array(x_list),
                                                    advantages:np.array([advantages_list]),
                                                    labels:np.array([labels_list])})	
			advantages_list=[]
			x_list=[]	
			labels_list=[]
			print(loss)
		episode_number+=1
		print("episode {}".format(episode_number))
		
		
    	



