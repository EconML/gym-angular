import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D
from keras.optimizers import Adam
import gym_angular
import math
from collections import deque
import sys, os

class DQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=5000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0001
        self.tau = .125

        #this model predicts which actions to take
        self.model = self.create_model()

        #the target model tracks the action we want our model to take
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()

        state_shape  = self.env.get_obs_space().shape
        model.add(Dense(24, input_shape=(795,), activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(24, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="categorical_crossentropy",
            optimizer=Adam(lr=self.learning_rate))
        return model

    """
    predicts action based on self.model,
    randomly selects action according to epsilon(e-greedy algorithm)
    and updates epsilon
    """
    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        print(len(self.memory))
        if(len(self.memory)==5000):
            self.memory.popleft()
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 128
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            state = state.reshape(1,795)
            target = self.target_model.predict(state)
            # print(target)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] += reward + Q_future * self.gamma
            # print('Fitting Model')
            self.model.fit(state, target, epochs=5, verbose=2)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    # def save_model(self, fn):lstm
    #     self.model.save(fn)

def main(path):
    env = gym.make('angular-v0')
    env.myInitial(path)
    gamma   = 0.9
    epsilon = .95

    num_models  = env.get_num_models()


    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    for cur_model in range(num_models-40):
        cur_model_steps = env.get_num_imgs()
        cur_state = env.reset()
        cur_state = cur_state.reshape(1,795)
        for cur_step in range(cur_model_steps):
            action = dqn_agent.act(cur_state)
            obs, reward, done, _ = env.step(action)
            obs = obs.reshape(1,795)

            # reward = reward if not done else -20
            dqn_agent.remember(cur_state, action, reward, obs, done)

            dqn_agent.replay()       # internally iterates default (prediction) model
            dqn_agent.target_train() # iterates target model

            cur_state = obs
            if done:
                break

    dqn_agent.model.save('trained_model.h5')
    # if step >= 199:
    #     print("Failed to complete in trial {}".format(trial))
    #     if step % 10 == 0:
    #         dqn_agent.save_model("trial-{}.model".format(trial))
    # else:
    #     print("Completed in {} num_models".format(trial))
    #     dqn_agent.save_model("success.model")
    #     break

if __name__ == "__main__":
    #TODO: get and parse command line args for input path, output path, etc
    if(len(sys.argv)) != 2:
        print("Usage: img_translate <file_path> <addr_output>\n"
        "       <input_path>    File path for .npy test/train data\n"
        )
        # exit(1)
    input_path = sys.argv[1]
    main(input_path)
