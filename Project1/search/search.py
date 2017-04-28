# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def genericSearch(problem, frontier):
    # an empty list: positions that have been explored
    explored = []
    # starting position
    state = problem.getStartState()
    # a list of actions that lead the pacman to goal state
    path = []
    frontier.push((state, path, 0))
    """
    two basic if statements:
    1. whether state has been explored?
    2. whether state is goal state?
    """
    while (not problem.isGoalState(state)) and (not frontier.isEmpty()):
        (state, path, pathCost) = frontier.pop()
        if problem.isGoalState(state):
            return path
        if not state in explored:
            explored.append(state)
            for (dest, action, cost) in problem.getSuccessors(state):
                if dest not in explored:
                    newPath = list(path)
                    newPath.append(action)
                    """
                    for A*search case, the newPathCost will be automatically updated
                    when it is pushed into the functional priority queue. Because the
                    priority queue with function will can calculate for you.
                    """
                    newPathCost = cost
                    frontier.push((dest, newPath, newPathCost))
    if problem.isGoalState(state):
        return path
    else:
        return []

def depthFirstSearch(problem):
    return genericSearch(problem, util.Stack())
    # "*** YOUR CODE HERE ***"
    # root = (problem.getStartState(), [], 0)
    # if problem.isGoalState(problem.getStartState()):
    #     return root[1]
    # # A FIFO queue with node as the only element to store node((x, y), action, cost)
    # frontier = util.Stack()
    # frontier.push(root)
    # # # An empty set to store node state (x, y)
    # explored = set()
    # while not frontier.isEmpty():
    #     parent = frontier.pop()
    #     # add parent node as explored node
    #     explored.add(parent[0])
    #     # get successor nodes of parent node
    #     for child in problem.getSuccessors(parent[0]):
    #         if not child[0] in explored:
    #             # determine if parent node is in a goal state
    #             if problem.isGoalState(child[0]):
    #                 # print actions + [child[1]]
    #                 return parent[1] + [child[1]]
    #             frontier.push((child[0], parent[1] + [child[1]], child[2]))
    # return []


def breadthFirstSearch(problem):
    return genericSearch(problem, util.Queue())

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "f(n) = g(n)"
    def priorityFunction(node):
        return problem.getCostOfActions(node[1])
    return genericSearch(problem, util.PriorityQueueWithFunction(priorityFunction))

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    """
    Best-First-Search: Priority queue keep tracking the cost, the lowest one with the highest priority
    f(n) = g(n) + h(n)"
    """
    def heuristicFunction(node):
        return problem.getCostOfActions(node[1]) + heuristic(node[0], problem)
    return genericSearch(problem, util.PriorityQueueWithFunction(heuristicFunction))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
