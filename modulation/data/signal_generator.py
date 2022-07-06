from functools import reduce
from typing import Union, Callable, Iterable, Sequence

from warg import Number, identity

__all__ = ["SignalGenerator"]


class SignalGenerator:
    """ """

    def __init__(self, *funcs: Union[Callable, Number], delta_time: float = 1.0):
        self.reset_internal_time()
        self.delta_time = delta_time
        self.funcs = (0, *funcs)

    def __iter__(self):
        self.reset_internal_time()
        return self

    def __next__(self):
        self.t += self.delta_time
        return self.apply(self.t)

    def apply(self, t: float) -> float:
        """

        :param t:
        :type t:
        :return:
        :rtype:
        """
        return reduce(lambda x, y: x + y(t), self.funcs)

    def __call__(self, t: Iterable[Number]) -> Sequence:
        return [self.apply(i) for i in t]

    def reset_internal_time(self):
        """ """
        self.t = 0.0

    def set_internal_time(self, t):
        """

        :param t:
        :type t:
        """
        self.t = t

    def __enter__(self):
        self.reset_internal_time()
        return True


if __name__ == "__main__":

    def asidjashdya():
        """
        counts
        """
        for _, i in zip(range(10), SignalGenerator(identity)):
            print(i)

    asidjashdya()
