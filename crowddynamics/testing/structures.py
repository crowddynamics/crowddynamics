"""Classes and function for testing"""


def function(a: (0, None) = 1,
             b: (None, 10.0) = 1.0,
             c: ('bar', 'baz') = 'foo',
             d = False):
    pass


class Class:
    def method(self,
               a: (0, None) = 1,
               b: (None, 10.0) = 1.0,
               c: ('bar', 'baz') = 'foo',
               d=False):
        pass
