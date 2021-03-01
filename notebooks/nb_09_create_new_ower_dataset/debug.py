def auto_str(cls):
    def __str__(self):
        lines = [self.__class__.__name__ + ':']

        for key, val in vars(self).items():
            lines += '{}: {}'.format(key, val).split('\n')

        return '\n    '.join(lines)

    cls.__str__ = __str__

    return cls
