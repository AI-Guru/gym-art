from gym.envs.registration import register

kwargs = {
    "source": "mnist",
    "brush_widths": 2,
}
register(
    id='art-mnist-v0',
    entry_point='gym_art.envs:ArtEnv',
    kwargs=kwargs
)

kwargs = {
    "source": "cifar10",
    "brush_widths": 2,
}
register(
    id='art-cifar10-v0',
    entry_point='gym_art.envs:ArtEnv',
    kwargs=kwargs
)
