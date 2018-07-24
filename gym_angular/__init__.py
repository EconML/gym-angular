from gym.envs.registration import register

register(
    id='angular-v0',
    entry_point='angular_qa.envs:AngularEnv'
)

# register(
#     id='foo-extrahard-v0',
#     entry_point='gym_foo.envs:FooExtraHardEnv'
# )
