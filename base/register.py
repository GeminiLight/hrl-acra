SolverLibrary = {}


class Register(dict):

    _dict = SolverLibrary

    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)

    @classmethod
    def register(cls, solver_name, target):
        assert 'solver' in target and 'env' in target
        if not callable(target['solver']) or not callable(target['env']):
            print(f'Failed to register {solver_name}!')
        cls._dict[solver_name] = target
        # if callable(target):
        #     return add_item(target.__name__, target)
        # else:
        #     return lambda x : add_item(target, x)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return list(self._dict.keys())

    def values(self):
        return list(self._dict.values())

    def items(self):
        return self._dict.items()