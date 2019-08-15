from rl.agent import Agent
from envs.GridKeyEnv import GridWorldKey

if __name__ == "__main__":

    ## Training on key1 and key2 environment, then apply experience reuse on key3 environment
    env1 = GridWorldKey(max_time=1000,n_keys=1,normlize_obs=True,use_nearby_obs=True)
    env2 = GridWorldKey(max_time=3000,n_keys=2,normlize_obs=True,use_nearby_obs=True)
    env3 = GridWorldKey(max_time=8000, n_keys=3, normlize_obs=True, use_nearby_obs=True)

    train_env_sets = [env1,env2]
    ##scale maximum possible returns to the same for balace learning
    rew_scale_factor = [1.2,1.4]

    agent = Agent('GridWorldKey', envs=train_env_sets, rew_scale=rew_scale_factor, batch_size=20, n_ways=0)

    print('----Initing')
    agent.init_scaler(10)
    print("----Learning L0 on env1 and env2")
    agent.train(n_iter=3000, L='0', mask=[1,1])

    print("----Learning L11 on env1")
    agent.add_module(L_name='11')
    agent.train(n_iter=1500, L='11', mask=[1,0])

    print("----Learning L12 on env2")
    print("\tLearning combine-weights on env2 for several iteration")
    agent.train(n_iter=500, L='11', mask=[0,1],trainWeight=True)
    print("\tLearning L12 module")
    agent.add_module(L_name='12')
    agent.train(n_iter=3500, L='12', mask=[0,1])

    print('----Learning L13 on unseen env3')
    agent.addenv(env=env3, rew_scale=1.6)
    print('\tLearning combine-weights on env2 for several iteration')
    agent.train(n_iter=500, L='12', mask=[0,0,1],trainWeight=True)
    print('\tLearning L13 module')
    agent.add_module(L_name='13')
    agent.train(n_iter=6000, L='13', mask=[0,0,1])

    agent.save_model()