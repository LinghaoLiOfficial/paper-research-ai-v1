import random

import torch


class STDataLoader:

    def __init__(self, x: list, y: list, batch_size: int, shuffle: bool, random_seed: int = 0):

        batch_x, batch_y = self._split_into_batches(
            x=x,
            y=y,
            batch_size=batch_size,
            shuffle=shuffle,
            random_seed=random_seed
        )

        self.x = batch_x
        self.y = batch_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length = len(self.x[0])

    def __len__(self):

        return self.length

    def __getitem__(self, item):

        out_x = [m[item] for m in self.x]
        out_y = [m[item] for m in self.y]

        return out_x, out_y

    def shuffle_self(self):

        batch_x = self.x

        new_batch_x = []
        new_batch_y = []
        for i in range(len(batch_x)):

            indices = list(range(len(batch_x[i])))
            # random.seed(random_seed)
            random.shuffle(indices)

            current_batch_x = [batch_x[i][l] for l in indices]
            current_batch_y = [batch_x[i][l] for l in indices]

            new_batch_x.append(current_batch_x)
            new_batch_y.append(current_batch_y)

        self.x = new_batch_x
        self.y = new_batch_y

    def _split_into_batches(self, x: list, y: list, batch_size: int, shuffle: bool, random_seed: int = 0):

        total_num = len(x[0])

        batch_num = int(total_num / batch_size) + 1

        batch_x = []
        batch_y = []
        for i in range(len(x)):

            current_batch_x = []
            current_batch_y = []

            for j in range(batch_num):

                if j != batch_num - 1:

                    current_x_list = x[i][j * batch_size: (j + 1) * batch_size]
                    current_y_list = y[i][j * batch_size: (j + 1) * batch_size]

                    current_batch_x.append(current_x_list)
                    current_batch_y.append(current_y_list)

            if shuffle:
                indices = list(range(len(current_batch_x)))
                # random.seed(random_seed)
                random.shuffle(indices)

                current_batch_x = [current_batch_x[l] for l in indices]
                current_batch_y = [current_batch_y[l] for l in indices]

            batch_x.append(current_batch_x)
            batch_y.append(current_batch_y)

        return batch_x, batch_y
