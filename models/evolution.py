import os
import random
import numpy as np
import torch

from engine import accuracy


def decode_cand_tuple(cand_tuple):
    depth = cand_tuple[0]
    expand_ratio = list(cand_tuple[1:depth + 1])
    d_state = list(cand_tuple[depth + 1: 2 * depth + 1])
    kernel_size = list(cand_tuple[2 * depth + 1: 3 * depth + 1])
    embed_dim = cand_tuple[-1]
    return depth, expand_ratio, d_state, kernel_size, embed_dim


class EvolutionSearcher(object):
    def __init__(self, args, device, model, choices, val_loader, test_loader, output_dir, logger):
        self.device = device
        self.model = model
        self.args = args

        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num

        self.m_prob = args.m_prob
        self.s_prob = args.s_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num

        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits

        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = output_dir

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.checkpoint_path = args.checkpoint_path

        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        self.choices = choices

        self.logger = logger

    def save_checkpoint(self):
        info = {
            'top_accuracies': self.top_accuracies,
            'memory': self.memory,
            'candidates': self.candidates,
            'vis_dict': self.vis_dict,
            'keep_top_k': self.keep_top_k,
            'epoch': self.epoch
        }
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{self.epoch}.pth")
        torch.save(info, checkpoint_path)
        self.logger.info(f'save checkpoint to {checkpoint_path}')

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return False
        info = torch.load(self.checkpoint_path)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        self.logger.info(f'load checkpoint from {self.checkpoint_path}')
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        depth, expand_ratio, d_state, kernel_size, embed_dim = decode_cand_tuple(cand)
        sampled_config = {
            'depth': depth,
            'expand_ratio': expand_ratio,
            'd_state': d_state,
            'kernel_size': kernel_size,
            'embed_dim': embed_dim
        }
        n_parameters = self.model.get_sampled_params_numel(sampled_config)
        info['params'] = n_parameters / 10. ** 6

        if info['params'] > self.parameters_limits:
            self.logger.info('parameters limit exceed')
            return False

        if info['params'] < self.min_parameters_limits:
            self.logger.info('under minimum parameters limit')
            return False

        val_stats = self.evaluate(data_loader=self.val_loader, sample_config=sampled_config)
        test_stats = self.evaluate(data_loader=self.test_loader, sample_config=sampled_config)

        info['val_acc'] = val_stats['acc1']
        info['test_acc'] = test_stats['acc1']
        info['visited'] = True

        return True

    def evaluate(self, data_loader, sample_config=None, mode="sub"):
        assert mode in ['sub', 'super']
        self.model.eval()
        if mode == "sub":
            assert sample_config is not None
            self.model.set_sample_config(config=sample_config)
        if mode == "super":
            self.model.set_super_config()
        correct_num, total_num = np.array([0., 0.]), 0
        for index, (x, label) in enumerate(data_loader):
            x, label = x.to(self.device), label.to(self.device)
            pred = self.model(x)
            correct_num += accuracy(pred, label, topk=(1, 5))
            total_num += label.size(0)
        accs = np.around(correct_num / total_num, decimals=4)
        info = dict(acc1=accs[0], acc5=accs[1])
        return info

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        self.logger.info('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                # info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random_cand(self):
        cand_tuple = list()
        dimensions = ['expand_ratio', 'd_state', 'kernel_size']
        depth = random.choice(self.choices['depth'])
        cand_tuple.append(depth)
        for dimension in dimensions:
            for i in range(depth):
                cand_tuple.append(random.choice(self.choices[dimension]))

        cand_tuple.append(random.choice(self.choices['embed_dim']))
        return tuple(cand_tuple)

    def get_random(self, num):
        self.logger.info('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            self.logger.info(f'random {len(self.candidates)}/{num}')
        self.logger.info(f'random_num = {len(self.candidates)}')

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        self.logger.info('mutation ......')
        res = []
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            depth, expand_ratio, d_state, kernel_size, embed_dim = decode_cand_tuple(cand)

            random_s = random.random()
            # depth
            if random_s < s_prob:
                new_depth = random.choice(self.choices['depth'])

                if new_depth > depth:
                    expand_ratio = (expand_ratio +
                                    [random.choice(self.choices['expand_ratio']) for _ in range(new_depth - depth)])
                    d_state = d_state + [random.choice(self.choices['d_state']) for _ in range(new_depth - depth)]
                    kernel_size = (kernel_size +
                                   [random.choice(self.choices['kernel_size']) for _ in range(new_depth - depth)])
                else:
                    expand_ratio = expand_ratio[:new_depth]
                    d_state = d_state[:new_depth]
                    kernel_size = kernel_size[:new_depth]

                depth = new_depth

            # expand_ratio
            for i in range(depth):
                random_s = random.random()
                if random_s < m_prob:
                    expand_ratio[i] = random.choice(self.choices['expand_ratio'])

            # d_state
            for i in range(depth):
                random_s = random.random()
                if random_s < m_prob:
                    d_state[i] = random.choice(self.choices['d_state'])

            # kernel_size
            for i in range(depth):
                random_s = random.random()
                if random_s < m_prob:
                    kernel_size[i] = random.choice(self.choices['kernel_size'])

            # embed_dim
            random_s = random.random()
            if random_s < s_prob:
                embed_dim = random.choice(self.choices['embed_dim'])

            result_cand = [depth] + expand_ratio + d_state + kernel_size + [embed_dim]

            return tuple(result_cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            self.logger.info(f'mutation {len(res)}/{mutation_num}')

        self.logger.info(f'mutation_num = {len(res)}')
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        self.logger.info('crossover ......')
        res = []
        max_iters = 10 * crossover_num

        def random_func():
            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            max_iters_tmp = 50
            while len(p1) != len(p2) and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])
            return tuple(random.choice([i, j]) for i, j in zip(p1, p2))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            self.logger.info(f'crossover {len(res)}/{crossover_num}')

        self.logger.info(f'crossover_num = {len(res)}')
        return res

    def search(self):
        self.model.eval()

        test_acc = self.evaluate(data_loader=self.test_loader, mode='super')
        val_acc = self.evaluate(data_loader=self.val_loader, mode='super')

        self.logger.info(
            f'population_num = {self.population_num} '
            f'select_num = {self.select_num} '
            f'mutation_num = {self.mutation_num} '
            f'crossover_num = {self.crossover_num} '
            f'random_num = {self.population_num - self.mutation_num - self.crossover_num} '
            f'max_epochs = {self.max_epochs} '
            f"super_top1_test_acc = {test_acc['acc1']:.4f} "
            f"super_top5_test_acc = {test_acc['acc5']:.4f} "
            f"super_top1_val_acc = {val_acc['acc1']:.4f} "
            f"super_top5_val_acc = {val_acc['acc5']:.4f} "
        )

        # self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            self.logger.info(f'epoch = {self.epoch}')

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['val_acc'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['val_acc'])

            self.logger.info(f'epoch = {self.epoch} : top {len(self.keep_top_k[self.population_num])} result')
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[self.population_num]):
                self.logger.info(f"No.{i + 1} {cand}  "
                                 f"Top1_val_acc = {self.vis_dict[cand]['acc1']}, "
                                 f"Top1_test_acc = {self.vis_dict[cand]['test_acc']}, "
                                 f"params = {self.vis_dict[cand]['params']}")
                tmp_accuracy.append(self.vis_dict[cand]['val_acc'])
            self.top_accuracies.append(tmp_accuracy)

            mutation = self.get_mutation(self.select_num, self.mutation_num, self.m_prob, self.s_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

            self.save_checkpoint()
