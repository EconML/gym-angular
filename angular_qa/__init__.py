from gym.envs.registration import register

register(
    id='angular-qa-v0',
    entry_point='angular_qa.envs:Angular_qa_env'
)

# register(
#     id='foo-extrahard-v0',
#     entry_point='gym_foo.envs:FooExtraHardEnv'
# )
