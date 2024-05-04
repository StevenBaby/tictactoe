import os
import re
import random
import pickle

import numpy as np
from tqdm import tqdm

dirname = os.path.dirname(os.path.abspath(__file__))

EMPTY = 0
BLACK = 1
WHITE = -2


def end_game(state: np.ndarray) -> np.ndarray | None:
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


def hash(state: np.ndarray):
    return tuple(state.reshape(9))


class Model(object):

    def __init__(self, epsilon=0.7, count=10000) -> None:
        self.epsilon = epsilon
        self.count = count
        self.filename = os.path.join(dirname, "model.pkl")
        self.table: dict[tuple, np.ndarray] = {}
        self.table: dict[tuple, np.ndarray] = self.load()

    def save(self):
        with open(self.filename, 'wb') as file:
            file.write(pickle.dumps(self.table))

    def load(self) -> dict[tuple, np.ndarray]:
        if not os.path.exists(self.filename):
            self.train()
        with open(self.filename, 'rb') as file:
            return pickle.loads(file.read())

    def act(self, state: np.ndarray, turn: int):
        # wheres = np.argwhere(state == EMPTY)
        # where = random.choice(wheres)
        # return tuple(where)
        return self.exploitation(state, turn)

    def exploration(self, state: np.ndarray):
        wheres = np.argwhere(state == EMPTY)
        assert (len(wheres) > 0)
        where = random.choice(wheres)
        return tuple(where)

    def exploitation(self, state: np.ndarray, turn: int):
        wheres = np.argwhere(state == EMPTY)
        assert (len(wheres) > 0)

        results = []
        for where in wheres:
            where = tuple(where)
            s = state.copy()
            s[where] = turn

            key = hash(s)
            if key not in self.table:
                continue

            black, draw, white = self.table[key]
            p = (black - white) / sum(self.table[key]) * turn
            results.append((where, p))

        if not results:
            return self.exploration(state)

        result = sorted(results, key=lambda e: e[1])[-1]
        return result[0]

    def step(self, state: np.ndarray, turn: int, chain: list):
        if random.random() < self.epsilon:
            where = self.exploration(state)
        else:
            where = self.exploitation(state, turn)

        state[where] = turn
        chain.append(hash(state))
        end = end_game(state)
        if end is None:
            return self.step(state, turn * -1, chain)
        for key in chain:
            self.table.setdefault(key, np.array([0, 0, 0]))
            self.table[key] += end
        return

    def train(self):
        state = np.zeros((3, 3), dtype=np.int8)
        turn = BLACK
        for _ in tqdm(range(self.count)):
            self.step(state.copy(), turn, [])
        self.save()


class Game(object):

    def __init__(self) -> None:
        self.state = np.zeros((3, 3), dtype=np.int8)
        self.turn = BLACK
        self.model = Model(epsilon=0.7, count=100000)
        # self.model.train()
        print(len(self.model.table))

    def input_action(self):
        while True:
            index = input("please input action (1 ~ 9):")
            if not re.match('[1-9]', index):
                continue
            index = int(index) - 1
            where = (index // 3, index % 3)
            if self.state[where] != EMPTY:
                continue
            return where

    def action(self, where: tuple[int, int]):
        assert (self.state[where] == 0)
        self.state[where] = self.turn
        self.turn *= -1

    def check(self):
        r = end_game(self.state)
        if r is None:
            return False
        print(self.state)
        black, draw, white = r
        if black:
            print('black win')
        if white:
            print('white win')
        if draw:
            print('draw')
        return True

    def start(self):
        while True:
            print(self.state)
            where = self.input_action()
            self.action(where)
            if self.check():
                break
            where = self.model.act(self.state, self.turn)
            self.action(where)
            if self.check():
                break


def main():
    game = Game()
    game.start()


if __name__ == '__main__':
    main()
