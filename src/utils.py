# my utils, supplem, aux etc


def save_to_globals(**kwargs):
    print('save to console:')
    for key, value in kwargs.items():
        globals()[key + '_'] = value
        print(f'{key} saved to globals as {key}_')


if __name__ == "__main__":

    def _test_save_to_globals():
        class Tst:
            pass
        t=Tst()
        save_to_globals(t=t)


    _test_save_to_globals()
