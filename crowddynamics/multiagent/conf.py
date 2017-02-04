from schematics.types import IntType, StringType


# TODO: default value as value/function

AGENT_MODELS = ['circular', 'three_circle']
BODY_TYPES = ['adult', 'male', 'female', 'child', 'eldery']
agent = (
    ('size', IntType(min_value=1, max_value=100000)),
    ('model', StringType(choices=AGENT_MODELS)),
    ('body_type', StringType(choices=BODY_TYPES)),
)
