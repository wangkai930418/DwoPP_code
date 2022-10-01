from torch.utils.data import sampler
import random
from collections import defaultdict


class RandomSampler(sampler.Sampler):
    def __init__(self, data_source, batch_k, CL_sign=False):
        super(RandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_k = batch_k
        self.CL_sign=CL_sign

        self._id2index = defaultdict(list)
        if hasattr(data_source, 'imgs'):
            for idx, path in enumerate(data_source.imgs):
                _id = data_source.id(path)
                self._id2index[_id].append(idx)

        ############ CL ###########################
        elif hasattr(data_source, 'train_data'):
            for idx, path in enumerate(data_source.train_data):
                _id = data_source.train_label[idx]
                self._id2index[_id].append(idx)
        ####################################

        elif hasattr(data_source, 'pids'):
            for idx, _id in enumerate(data_source.pids):
                self._id2index[_id].append(idx)
        elif hasattr(data_source, 'data'):
            for idx, (_, _id, _) in enumerate(data_source.data):
                self._id2index[_id].append(idx)
        else:
            raise NotImplementedError

    def __iter__(self):
        if self.CL_sign:
            unique_ids=list(set(self.data_source.train_label))
        else:
            unique_ids = self.data_source.unique_ids
        random.shuffle(unique_ids)

        imgs = []
        for _id in unique_ids:
            imgs.extend(self._sample(self._id2index[_id], self.batch_k))
        return iter(imgs)

    def __len__(self):
        return len(self._id2index) * self.batch_k

    @staticmethod
    def _sample(population, k):
        if len(population) < k:
            population = population * k
        return random.sample(population, k)
