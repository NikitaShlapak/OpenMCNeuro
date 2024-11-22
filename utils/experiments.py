from itertools import product


class ParamGrid:
    def __init__(self, params: dict[str, list[float]]) -> None:
        self.params = params.keys()
        self.values = product(*params.values())
        print(self.params, self.values)

    @property
    def grid(self):
        for values in self.values:
            yield dict(zip(self.params, values))


if __name__ == '__main__':

    PG = ParamGrid(
        {
            'a': [1, 10, 1],
            'b': [2, 10, 2],
            'c': [3, 10, 3]
        }
    )
    for p in PG.grid:
        print(p)
