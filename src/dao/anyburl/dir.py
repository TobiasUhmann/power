from os import makedirs, path, remove
from os.path import isfile


def assert_not_existing(anyburl_dataset_dir, files, overwrite):
    """
    Check 'AnyBURL Dataset Directory':
    - Create it if it does not already exist
    - Assert that its files do not already exist
    """

    makedirs(anyburl_dataset_dir, exist_ok=True)
    files['anyburl'] = {}

    train_txt = path.join(anyburl_dataset_dir, 'train.txt')
    files['anyburl']['train_txt'] = train_txt
    if isfile(train_txt):
        if overwrite:
            remove(train_txt)
        else:
            print("'AnyBURL Dataset Directory' / 'Train TXT' already exists"
                  ", use --overwrite to overwrite it")
            exit()

    valid_txt = path.join(anyburl_dataset_dir, 'valid.txt')
    files['anyburl']['valid_txt'] = valid_txt
    if isfile(valid_txt):
        if overwrite:
            remove(valid_txt)
        else:
            print("'AnyBURL Dataset Directory' / 'Valid TXT' already exists"
                  ", use --overwrite to overwrite it")
            exit()

    test_txt = path.join(anyburl_dataset_dir, 'test.txt')
    files['anyburl']['test_txt'] = test_txt
    if isfile(test_txt):
        if overwrite:
            remove(test_txt)
        else:
            print("'AnyBURL Dataset Directory' / 'Test TXT' already exists"
                  ", use --overwrite to overwrite it")
            exit()
