
def print_tf(xx):
    if not isinstance(xx, list):
        xx = [xx]

    for x in xx:
        print('Type  :\n{}'.format(type(x)))
        print('Value :\n{}'.format(x))
