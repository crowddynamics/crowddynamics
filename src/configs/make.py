try:
    from ruamel import yaml
except ImportError:
    import yaml

from src.structure.agent import spec_agent
from src.structure.obstacle import spec_linear


def make_yaml_config():
    d = {
        "agent": spec_agent,
        "walls": spec_linear
    }
    data = {}
    for name, spec in d.items():
        data[name] = [item[0] for item in spec]

    with open("saveable.yaml", "w") as f:
        yaml.safe_dump(data, stream=f, default_flow_style=False)


make_yaml_config()
