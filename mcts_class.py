import numpy as np
import random
import time
from scipy import signal

class Node:

    def __init__(self, checkerboard, last_move, color, parent=None, children=None, value=0, n_visits=0):

        self.checkerboard = checkerboard  # 棋盘
        self.color = color  # 棋子颜色
        self.last_move = last_move #a list

        self.parent = parent
        if children is None:
            children = []
        self.children = children

        self.value = value  # 胜率
        self.n_visits = n_visits  # 访问次数

        self.good = [
            # 一颗棋子的情况，
            # [1, 3, 0, 0, 0]并不一定表示一行的形式，也可能是斜对角线的形式
            [[1, 3, 0, 0, 0], [0, 1, 3, 0, 0], [0, 0, 1, 3, 0], [0, 0, 0, 1, 3], [0, 0, 0, 3, 1],
             [-1, 3, 0, 0, 0], [0, -1, 3, 0, 0], [0, 0, -1, 3, 0], [0, 0, 0, -1, 3], [0, 0, 0, 3, -1]],
            # 二颗棋子的情况
            [[1, 1, 3, 0, 0], [0, 1, 3, 1, 0], [0, 0, 3, 1, 1],
             [-1, -1, 3, 0, 0], [0, 0, 3, -1, -1], [0, -1, 3, -1, 0]],
            # 三颗棋子的情况
            [[1, 1, 1, 3, 0], [1, 1, 3, 1, 0], [1, 3, 1, 1, 0], [0, 3, 1, 1, 1],
             # [0, 1, 1, 1, 3], [0, 1, 1, 3, 1], [0, 1, 3, 1, 1],这三种情况可以不写，因为搜索该点的右边/下面/右下角的点就会出现上面一行的情况
             [-1, -1, -1, 3, 0], [3, -1, -1, -1, 0],
             # [0, 3, -1, -1, -1],[0, -1, 3, -1, -1],[0, -1, -1, 3, -1]这种情况可以不写，感觉上面两种也有一种可以不写,但为了不老是下在同一个地方，最后要用random进行选择
             [-1, -1, 3, 0, -1], [-1, 0, 3, -1, -1], [-1, -1, 3, -1, 0], [-1, 3, -1, -1, 0]],
            # 四颗棋子情况
            [[1, 1, 1, 1, 3], [3, 1, 1, 1, 1], [1, 1, 1, 3, 1], [1, 3, 1, 1, 1], [1, 1, 3, 1, 1],
             [-1, -1, -1, -1, 3], [3, -1, -1, -1, -1], [-1, -1, 3, -1, -1], [-1, 3, -1, -1, -1], [-1, -1, -1, 3, -1]]
        ]

    def copy(self):
        copied_node = Node(self.checkerboard.copy(), self.last_move, self.color, None, None, 0, 0)

        return copied_node

    def is_terminal_node(self):
        # t = self.checkerboard.check_winner()
        return self.checkerboard.check_winner() != 0

    def get_legal_moves(self):
        """
        找以落子为中心3*3内的空位
        """
        bline = self.checkerboard.bline
        pos = self.last_move
        row, col = pos[0], pos[1]
        row_lim = [max(row - 2, 0), min(row + 3, bline)]
        col_lim = [max(col - 2, 0), min(col + 3, bline)]

        legal_moves = []
        for i in range(row_lim[0], row_lim[1]):
            for j in range(col_lim[0], col_lim[1]):
                if self.checkerboard.board[i, j] == 0:
                    legal_moves.append((i, j))
        return legal_moves

    def is_fully_expanded(self):

        n_moves = len(self.get_legal_moves())
        n_children = len(self.children)
        return n_children == n_moves

    def add_child(self, move):

        new_checkerboard = self.checkerboard.copy()

        new_color = -1 if self.color == 1 else 1
        new_pos = move
        new_checkerboard.update_board(new_pos[0], new_pos[1], new_color)
        new_node = Node(checkerboard=new_checkerboard, last_move=new_pos, color=new_color, parent=self)
        # 所以每次增加一个节点之后，新的节点对应的checkerboard的局面是更新之后的局面，这就意味着我们不能在selection的时候从根节点开始找？
        # 不应该从这个结点开始的子节点找起，应该从根节点开始找起
        self.children.append(new_node)
        return new_node


    def update_value(self, result):
        n_win = self.value * self.n_visits
        if result == self.color:
            n_win += 1
        elif result == 2: #这行应该需要加上，万一结果是平局呢。
            n_win += 2
        self.n_visits += 1
        self.value = n_win / self.n_visits


    def check_good_move(self, row, col, level, dx, dy):
        """
        Although this function does not return a value, its purpose is to update self.optimal_moves
        which is used in the choose_best_move function.
        Starting from the point (row row, col column) on the chessboard.
        This function performs a check. The check is performed from the last level to the first, with levels being 3, 2, 1, and 0.
        dx and dy represent the direction for the next move.
        The possible values for dx and dy are 0, 1, or -1.
        """
        board = self.checkerboard.board
        self.optimal_moves = []
        for s in self.good[level]:
            check = 1
            nrow, ncol = -1, -1
            for i in range(5): #(dx,dy) is checking direction
                if row + i * dx in range(0,11) and col + i * dy in range(0,11):
                    if s[i] == 3 and board[row + i * dx, col + i * dy] == 0:
                        nrow, ncol = row + i * dx, col + i * dy
                    elif s[i] != board[row + i * dx, col + i * dy]:
                        check = 0
                        break
            if check != 0 and nrow != -1:
                self.optimal_moves.append((nrow, ncol))
            if len(self.optimal_moves) > 2:
                break
        return nrow, ncol

    def choose_best_move(self):
        lrow, lcol = self.last_move
        best_row, best_col = -1,-1
        max_level = -1
        ## Since we are searching within a 5 x5 region centered around the current move
        direction = [(0,-1),(-1,1),(-1,-1),(-1,0),(1, 0), (1, 1), (1, -1), (0, 1)]
        find = False
        row_range = [max(lrow-2,0),min(lrow+3,11)]
        col_range = [max(lcol-2,0),min(lcol+3,11)]
        for i in range(row_range[0], row_range[1]):
            if find:
                break
            for j in range(col_range[0],col_range[1]):
                if find:
                    break
                for level in range(3,-1,-1): #3， 2， 1， 0
                    if find:
                        break
                    if level > max_level:
                        for d in direction:
                            nrow, ncol = self.check_good_move(i, j, level, *d)
                            print("self.optimal_moves:",self.optimal_moves)
                            if len(self.optimal_moves) > 0:
                                find = True
                                max_level = level
                                best_row, best_col = random.choice(self.optimal_moves)
        if max_level < 0:
            moves = self.get_legal_moves()
            best_move = random.choice(moves)
            if len(moves) == 0:
                board = self.checkerboard.board
                kernel = np.ones((3, 3))
                score = signal.convolve2d(abs(board), kernel, mode="same")
                top_scores = sorted(score.flatten(), reverse=True)
                for top_score in top_scores:
                    xy = np.where(score == top_score)
                    moves = list(zip(xy[0], xy[1]))
                    best_moves = [move for move in moves if board[move] == 0]
                    if best_moves:
                        break
                best_row, best_col = random.choice(best_moves)

        return best_row, best_col


class MCTS:

    def __init__(self, n_searches=50, duration=5):
        self.n_searches = n_searches
        self.duration = duration
        self.C = 1

    def calculate_UCB(self, node):
        if node.n_visits == 0:
            return np.inf
        if node.parent:
            parent_n_visits = node.parent.n_visits
        else:
            parent_n_visits = self.n_searches
        return node.value + self.C * np.sqrt(np.log(parent_n_visits) / node.n_visits)

    def selection(self, node):
        # If the node has come to and end, which is win, lose or draw, return the node for further backpropagation
        if node.is_terminal_node():
            return node
        # If the node is not fully expanded, return the node for its further expansion
        if not node.is_fully_expanded():
            return node
        # If the node is not the terminal node and has already fully expanded, then select one of its child node with
        # greatest UCT.
        UCBs = []
        children = node.children
        for child_node in children:
            UCB = self.calculate_UCB(child_node)
            UCBs.append(UCB)
        candidate = [idx for idx, UCB in enumerate(UCBs) if UCB == max(UCBs)]
        selected = random.choice(candidate)
        return self.selection(children[selected])

    def expansion(self, node):
        moves = node.get_legal_moves()
        children = node.children
        children_moves = []
        for child_node in children:
            child_move = child_node.last_move
            children_moves.append(child_move)
        for move in moves:
            if move not in children_moves:
                new_node = node.add_child(move)
                return new_node

    def simulation(self, node):

        node_copy = node.copy()
        color = node_copy.color
        best_move = node_copy.choose_best_move()
        node_copy.checkerboard.update_board(best_move[0], best_move[1], color)
        node_copy.last_move = best_move
        color = 1 if color == -1 else -1
        node_copy.color = color

        while node_copy.checkerboard.check_winner() == 0:
            board = node_copy.checkerboard.board
            # Find the densest grid by 3 × 3
            kernel = np.ones((3, 3))
            score = signal.convolve2d(abs(board), kernel, mode="same")
            top_scores = sorted(score.flatten(), reverse=True)
            for top_score in top_scores:
                xy = np.where(score == top_score)
                moves = list(zip(xy[0], xy[1]))
                best_moves = [move for move in moves if board[move] == 0]
                if best_moves:
                    break

            best_move = random.choice(best_moves)

            color = 1 if node_copy.color == -1 else -1
            node_copy.checkerboard.update_board(best_move[0], best_move[1], color)
            node_copy.last_move = best_move
            node_copy.color = color
        return node_copy.checkerboard.check_winner()

    def backprogation(self, node, result):
        while node.parent is not None:
            node.update_value(result)
            node = node.parent
        node.update_value(result)

    def search(self, root):
        start = time.time()
        while time.time() - start < self.duration:
            selected_node = self.selection(root)
            if selected_node.is_terminal_node():
                result = selected_node.checkerboard.check_winner()
                score = 2 * result if result != 2 else 0
                self.backprogation(selected_node, score)
            # If the node gets no legal moves, as we define them in a certain grid, then backpropagate with score 0,
            # then it has no influence of the score.
            elif selected_node.get_legal_moves() is None:
                score = 0
                self.backprogation(selected_node, score)
            # If the node still has legal moves, then just expand it and do simulation, finally backpropagate the score.
            else:
                new_node = self.expansion(selected_node)
                result = self.simulation(new_node)
                score = result if result != 2 else 0
                self.backprogation(new_node, score)


    def get_next_move(self, node):
        """
        AI move
        """
        self.search(node)
        children = node.children
        values = [child_node.value for child_node in children]
        # print("Winning rate:", max_values)
        # idx = values.index(max_values)
        idx = values.index(max(values))
        best_node = children[idx]
        move = best_node.last_move
        return move
# 我有一个问题，MCST在selection的时候是从当前结点开始寻找最大UCB的点还是从根节点开始呀？
# 我一方面觉得从根节点开始用递归去找很合理，一方面又觉得根节点是没有
#现在觉得：当前的位置就是根节点，每次都是，因为棋盘格局每次都更新