import os
import re
import random
import pickle

import numpy as np
from tqdm import tqdm
from logger import logger

dirname = os.path.dirname(os.path.abspath(__file__))

EMPTY = 0
BLACK = 1
WHITE = -1


class Model(object):

    def __init__(self, epsilon=0.7, count=10000) -> None:
        self.epsilon = epsilon
        self.count = count
        self.filename = os.path.join(dirname, "model.pkl")
        self.table: dict[tuple, np.ndarray] = self.load()
        logger.info("load model state count %s", self.table.__len__())

    def save(self):
        with open(self.filename, 'wb') as file:
            file.write(pickle.dumps(self.table))

    def load(self) -> dict[tuple, np.ndarray]:
        if not os.path.exists(self.filename):
            return {}
        with open(self.filename, 'rb') as file:
            return pickle.loads(file.read())

    def end_game(self, state: np.ndarray) -> np.ndarray | None:
        r0 = list(np.sum(state, axis=0))
        r1 = list(np.sum(state, axis=1))
        r2 = [np.trace(state)]
        r3 = [np.trace(np.flip(state, axis=1))]
        r = r0 + r1 + r2 + r3

        # 三个数分别表示 黑，平，白
        if 3 in r:
            return np.array([1, 0, 0])
        if -3 in r:
            return np.array([0, 0, 1])
        if len(np.argwhere(state == 0)) == 0:
            return np.array([0, 1, 0])
        return None

    def hash(self, state: np.ndarray):
        return tuple(state.reshape(9))

    def act(self, state: np.ndarray, turn: int):
        # wheres = np.argwhere(state == EMPTY)
        # where = random.choice(wheres)
        # return tuple(where)
        return self.exploitation(state, turn)

    def exploration(self, state: np.ndarray):
        wheres = np.argwhere(state == EMPTY)
        assert (len(wheres) > 0)
        where = random.choice(wheres)
        return tuple(where), 0.0

    def exploitation(self, state: np.ndarray, turn: int):
        wheres = np.argwhere(state == EMPTY)
        assert (len(wheres) > 0)

        results = []
        for where in wheres:
            where = tuple(where)
            s = state.copy()
            s[where] = turn

            key = self.hash(s)
            if key not in self.table:
                continue

            black, draw, white = self.table[key]
            p = (black - white) / sum(self.table[key]) * turn
            results.append((where, p))

        if not results:
            return self.exploration(state)

        result = sorted(results, key=lambda e: e[1])[-1]
        return result

    def step(self, state: np.ndarray, turn: int, chain: list):
        if random.random() < self.epsilon:
            where, confidence = self.exploration(state)
        else:
            where, confidence = self.exploitation(state, turn)

        state[where] = turn
        chain.append(self.hash(state))
        end = self.end_game(state)
        if end is None:
            return self.step(state, turn * -1, chain)
        for key in chain:
            self.table.setdefault(key, np.array([0, 0, 0]))
            self.table[key] += end
        return

    def train(self):
        state = np.zeros((3, 3), dtype=np.int8)
        turn = BLACK
        bar = tqdm(range((self.count)))
        for _ in bar:
            self.step(state.copy(), turn, [])
            bar.set_postfix(cnt=len(self.table))
        # self.save()
