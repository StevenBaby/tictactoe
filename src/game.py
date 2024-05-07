import re

import numpy as np

from model import BLACK, EMPTY, WHITE, Model


class Game(object):

    def __init__(self, ai=WHITE, epsilon=0.7, count=100000) -> None:
        self.state = np.zeros((3, 3), dtype=np.int8)
        self.turn = BLACK
        self.model = Model(epsilon=epsilon, count=count)
        self.ai = ai
        self.last = None
        self.stack = []
        # self.model.train()
        # print(len(self.model.table))

    def reset(self):
        self.last = None
        self.stack = []
        self.state = np.zeros((3, 3), dtype=np.int8)
        self.turn = BLACK

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
        self.stack.append((self.state.copy(), self.turn, where))
        self.state[where] = self.turn
        self.last = where
        self.turn *= -1

    def undo(self, count=2):
        if len(self.stack) < count:
            return
        self.state, self.turn, self.last = self.stack[-count]
        self.stack = self.stack[:-count]

    def check(self):
        r = self.model.end_game(self.state)
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
            if self.ai == self.turn:
                where, confidence = self.model.act(self.state, self.turn)
            else:
                where = self.input_action()
            self.action(where)
            if self.check():
                break


def main():
    game = Game()
    game.ai = BLACK
    game.start()


if __name__ == '__main__':
    main()
