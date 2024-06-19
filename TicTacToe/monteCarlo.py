import copy
import random
import time
import sys
import math
from collections import namedtuple
import numpy as np

GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')

def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None

# MonteCarlo Tree Search support

class MCTS:
    class Node:
        def __init__(self, state, par=None):
            self.state = copy.deepcopy(state)
            self.parent = par
            self.children = []
            self.visitCount = 0
            self.winScore = 0

        def getChildWithMaxScore(self):
            maxScoreChild = max(self.children, key=lambda x: x.visitCount)
            return maxScoreChild

    def __init__(self, game, state):
        self.root = self.Node(state)
        self.state = state
        self.game = game
        self.exploreFactor = math.sqrt(2)

    def isTerminalState(self, utility, moves):
        return utility != 0 or len(moves) == 0

    def monteCarloPlayer(self, timelimit=4):
        """Entry point for Monte Carlo search"""
        start = time.perf_counter()
        end = start + timelimit

        """Use timer above to apply iterative deepening"""
        while time.perf_counter() < end:
             #count = 100  # use this and the next line for debugging. Just disable previous while and enable these 2 lines
            # while count >= 0:
            #     count -= 1
            # SELECT stage use selectNode()
            # print("Your code goes here -3pt")
            node = self.selectNode(self.root)

            if not self.isTerminalState(node.state.utility, node.state.moves):
                self.expandNode(node)

            # SIMULATE stage using simuplateRandomPlay()
            # print("Your code goes here -3pt")
            node = random.choice(node.children) if len(node.children) > 0 else node
            result = self.simulateRandomPlay(node)

            # BACKUP stage using backPropagation
            # print("Your code goes here -2pt")
            self.backPropagation(node, result)

        winnerNode = self.root.getChildWithMaxScore()
        assert (winnerNode is not None)
        return winnerNode.state.move
    
    """selection stage function. walks down the tree using findBestNodeWithUCT()"""
    def selectNode(self, nd):
        # node = nd
        while len(nd.children)>0:
            nd = self.findBestNodeWithUCT(nd)
        return nd

    def findBestNodeWithUCT(self, nd):
        """finds the child node with the highest UCT. Parse nd's children and use uctValue() to collect uct's for the
        children....."""
        childUCT = []
         # Compute UCT values for each child
  
        childUCT = [self.uctValue(nd.visitCount, child.winScore, child.visitCount) for child in nd.children]
        # Find the child with the maximum UCT value
        return nd.children[np.argmax(childUCT)]

    def uctValue(self, parentVisit, nodeScore, nodeVisit):
        if nodeVisit == 0:
            return 0 if self.exploreFactor == 0 else sys.maxsize
        return (nodeScore / nodeVisit) + self.exploreFactor * math.sqrt(math.log(parentVisit) / nodeVisit)

    def expandNode(self, nd):
        """generate the child nodes and append them to nd's children"""
        for a in self.game.actions(nd.state):
            childState = self.game.result(nd.state, a)
            childNode = self.Node(childState, nd)
            nd.children.append(childNode)
    
    def simulateRandomPlay(self, nd):
        # first check win possibility for the current node:
        winStatus = self.game.compute_utility(nd.state.board, nd.state.move, nd.state.board[nd.state.move])

        if nd.parent is not None:
            if(winStatus == self.game.k and nd.state.to_move == 'X') or (winStatus == -self.game.k and nd.state.to_move == 'O'):
                nd.parent.winScore = sys.maxsize
                return nd.state.to_move
            elif(winStatus == self.game.k and nd.state.to_move == 'O') or (winStatus == -self.game.k and nd.state.to_move == 'X'):
                nd.parent.winScore = -sys.maxsize
                return nd.state.to_move

        """now roll out a random play down to a terminating state. """
        tempState = copy.deepcopy(nd.state)
        while not self.game.terminal_test(tempState):
            possibleMove = self.game.actions(tempState)
            move = random.choice(possibleMove)
            tempState = self.game.result(tempState,move)

        if tempState.to_move not in tempState.board:
            return 'N' 

        final_utility = self.game.compute_utility(tempState.board, tempState.to_move, tempState.board[tempState.to_move])
        return 'X' if final_utility > 0 else 'O' if final_utility < 0 else 'N'



    def backPropagation(self, nd, winningPlayer):
        """propagate upword to update score and visit count from
        the current leaf node to the root node."""
        tempNode = nd
        # print("Your code goes here -5pt")
        while tempNode is not None:
            tempNode.visitCount += 1
            if winningPlayer != 'N':
                if tempNode.state.to_move == winningPlayer:
                    tempNode.winScore += sys.maxsize
                else:
                    tempNode.winScore -= sys.maxsize
            tempNode = tempNode.parent



