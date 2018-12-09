# my utils, supplem, aux etc


def save_to_console(**kwargs):
    print('save to console:')
    for key, value in kwargs.items():
        globals()[key + '_'] = value
        print(f'{key} saved to globals as {key}_')


if __name__ == "__main__":
    import pandas


    def test_save_to_globals():
        df = pandas.DataFrame([1, 2, 3])
        save_to_console(df=df)


    test_save_to_globals()
