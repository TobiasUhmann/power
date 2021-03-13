from torch.utils.tensorboard import SummaryWriter
import numpy as np


def main():
    writer = SummaryWriter()

    for i in range(1, 100):
        writer.add_scalars('loss', {'train': 1 / i}, i)

    for i in range(1, 100):
        writer.add_scalars('loss', {'valid': 2 / i}, i)

    # r = 5
    # for i in range(100):
    #     writer.add_scalars('run_14h', {'xsinx': i * np.sin(i / r),
    #                                    'xcosx': i * np.cos(i / r),
    #                                    'tanx': np.tan(i / r)}, i)
    writer.close()


if __name__ == '__main__':
    main()
