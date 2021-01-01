import os


def main():
    if not os.path.isdir('data/'):
        os.mkdir('data/')


if __name__ == '__main__':
    main()
