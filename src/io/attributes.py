from timeit import default_timer as timer

from src.struct.constant import constant_attr_names
from src.struct.result import result_attr_names
from src.struct.agent import agent_attr_names
from src.struct.wall import wall_attr_names


class Attr:
    def __init__(self, name, is_resizable=False, save_func=None):
        self.name = name
        self.is_resizable = is_resizable
        self.save_func = save_func

    def __str__(self):
        return "Attr({name}, is_resizable={is_resizable}, " \
               "save_func={save_func})"\
            .format(name=self.name, is_resizable=self.is_resizable,
                    save_func=self.save_func)


class Attrs(dict):
    def __init__(self, names_or_attrs):
        super().__init__()
        for value in names_or_attrs:
            if isinstance(value, Attr):
                self[value.name] = value
            elif isinstance(value, str):
                self[value] = Attr(value)
            else:
                raise ValueError("Value: {value} of type {type} not valid "
                                 "type for names_or_attrs."
                                 .format(value=value, type=type(value)))

    def __iter__(self):
        return iter(self.values())

    def __str__(self):
        return "[\n\t" + ",\n\t".join((str(i) for i in self)) + "\n]"


class Intervals:
    def __init__(self, interval):
        self.interval = interval
        if self.interval < 0:
            raise ValueError("Interval should be > 0")
        self.prev = timer()

    @property
    def name(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        if self.interval == 0:
            return True
        else:
            current = timer()
            diff = current - self.prev
            ret = diff >= self.interval
            if ret:
                self.prev = current
            return ret

    def __str__(self):
        return "{name}({interval})".format(name=self.name,
                                           interval=self.interval)


attrs_constant = Attrs(constant_attr_names)
attrs_result = Attrs(result_attr_names)
attrs_agent = Attrs(agent_attr_names)
attrs_wall = Attrs(wall_attr_names)

attrs_agent["position"] = Attr("position", True, Intervals(5))
attrs_agent["velocity"] = Attr("velocity", True, Intervals(5))
# attrs_agent["force"] = Attr("force", True, Intervals(5))
