import time

from gym_art.envs.art_env import ArtEnv
environment = ArtEnv(source="cifar10", brush_widths=4, max_steps=100)

for i in range(10):
    observation = environment.reset()
    done = False
    while done == False:
        action = environment.action_space.sample()
        (target_observation, canvas_observation), reward, done, info = environment.step(action)
        environment.render()
        print("Step", i, "Reward", reward)
