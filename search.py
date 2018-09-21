# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import *


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    '''Getting the Start state to start the DFS traversal'''
    start_node = problem.getStartState()

    '''Initializing :
       visited_node[]: List of visited nodes
       dfs_stack : Stack data structure imported from util.py for DFS traversal'''
    visited_nodes = []
    dfs_stack = Stack()

    '''Converting the start node in form of a tuple start_tuple:
        start_tuple[0] : Co-ordinates of the start node
        start_tuple[1] : Direction List to reach start_node (Empty as we are at the 
                         start_node already)
        start_tuple[2] : Cost to reach start_node from parent (0 as start_node has
                         no parent)'''
    start_tuple = (start_node, [], 0)

    ''' Push the initial node in the stack to start the traversal'''
    dfs_stack.push(start_tuple)

    ''' Continue the DFS till the stack is not empty'''

    while not dfs_stack.isEmpty():
        '''Pop the node in LIFO order from the stack'''
        traverse_node = dfs_stack.pop()

        ''' Check if the node is goal state'''
        if problem.isGoalState(traverse_node[0]):
            return traverse_node[1]

        '''Process further only if the traverse node is not in visited_nodes list'''
        if traverse_node[0] not in visited_nodes:

            '''Add traverse node to the visited_nodes list'''
            visited_nodes.append(traverse_node[0])

            '''Get successors for the parent and push them in the stack'''
            successor_list = problem.getSuccessors(traverse_node[0])

            if len(successor_list)>0:
                for successor in successor_list:
                    if successor[0] not in visited_nodes:
                        direction_list = traverse_node[1]+[successor[1]]
                        dfs_stack.push((successor[0], direction_list, successor[2]))


    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    '''Getting the Start state to start the BFS traversal'''
    start_node = problem.getStartState()

    '''Initializing :
       visited_node[]: List of visited nodes
       bfs_queue : Queue data structure imported from util.py for BFS traversal'''

    visited_nodes = []
    bfs_queue = Queue()

    '''Converting the start node in form of a tuple start_tuple:
       start_tuple[0] : Co-ordinates of the start node
       start_tuple[1] : Direction List to reach start_node (Empty as we are at the 
                             start_node already)
       start_tuple[2] : Cost to reach start_node from parent (0 as start_node has
                             no parent)'''

    start_tuple = (start_node, [], 0)

    ''' Push the initial node in the queue to start the traversal'''
    bfs_queue.push(start_tuple)

    ''' Continue the BFS till the queue is not empty'''
    while not bfs_queue.isEmpty():
        '''Pop the node in FIFO order from the queue'''
        traverse_node = bfs_queue.pop()

        ''' Check if the node is goal state'''
        if problem.isGoalState(traverse_node[0]):
            return traverse_node[1]

        '''Process further only if the traverse node is not in visited_nodes list'''
        if traverse_node[0] not in visited_nodes:

            '''Add traverse node to the visited_nodes list'''
            visited_nodes.append(traverse_node[0])

            '''Get successors for the parent and push them in the queue'''
            successor_list = problem.getSuccessors(traverse_node[0])

            if len(successor_list)>0:
                for successor in successor_list:
                    if successor[0] not in visited_nodes:
                        direction_list = traverse_node[1] + [successor[1]]
                        bfs_queue.push((successor[0], direction_list, successor[2]))


    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    "*** YOUR CODE HERE ***"
    '''Getting the Start state to start the UCS traversal'''
    start_node = problem.getStartState()

    '''Initializing :
       visited_node[]: List of visited nodes
       ucs_queue : Priority Queue data structure imported from util.py for UCS traversal'''

    visited_nodes = []
    ucs_queue = PriorityQueue()

    '''Converting the start node in form of a tuple start_tuple:
       start_tuple[0] : Co-ordinates of the start node
       start_tuple[1] : Direction List to reach start_node (Empty as we are at the 
                             start_node already)
       start_tuple[2] : Cost to reach start_node from parent (0 as start_node has
                             no parent)'''
    start_tuple = (start_node, [], 0)

    '''Push start tuple in the Priority Queue along with it's cost (0 in this case
       as we are at the start_node itself). Cost is used to assign priority'''
    ucs_queue.push(start_tuple, 0)

    ''' Continue the UCS till the queue is not empty'''
    while not ucs_queue.isEmpty():
        '''Pop the node with the lowest cost'''
        traverse_node = ucs_queue.pop()

        ''' Check if the node is goal state'''
        if problem.isGoalState(traverse_node[0]):
            return traverse_node[1]

        '''Process further only if the traverse node is not in visited_nodes list'''
        if traverse_node[0] not in visited_nodes:

            '''Add traverse node to the visited_nodes list'''
            visited_nodes.append(traverse_node[0])

            '''Get successors for the parent and push them in the queue'''
            successor_list = problem.getSuccessors(traverse_node[0])

            for successor in successor_list:
                if successor[0] not in visited_nodes:
                    '''While pushing the node in the priority queue add the cost of it's
                    parent as well. This will allow us to keep track of total cost of
                    the path to the node.'''
                    direction_list = traverse_node[1] + [successor[1]]
                    ucs_queue.push((successor[0], direction_list, successor[2] + traverse_node[2]), successor[2] + traverse_node[2])

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    '''Getting the Start state to start the Astar traversal'''
    start_node = problem.getStartState()

    '''Initializing :
       visited_node[]: List of visited
       astar_queue : Priority Queue data structure imported from util.py for Astar traversal'''
    visited_nodes = []
    astar_queue = PriorityQueue()

    '''Converting the start node in form of a tuple start_tuple:
       start_tuple[0] : Co-ordinates of the start node
       start_tuple[1] : Direction List to reach start_node (Empty as we are at the 
                             start_node already)
       start_tuple[2] : Cost to reach start_node from parent (0 as start_node has
                             no parent)'''
    start_tuple = (start_node, [], 0)

    '''Push start tuple in the Priority Queue along with it's cost (0 in this case
       as we are at the start_node itself). Cost + Heuristic is used to assign priority'''
    astar_queue.push(start_tuple, 0)

    ''' Continue the Astar till the queue is not empty'''
    while not astar_queue.isEmpty():
        '''Pop the node with the lowest (cost+heuristic)'''
        traverse_node = astar_queue.pop()

        ''' Check if the node is goal state'''
        if problem.isGoalState(traverse_node[0]):
            return traverse_node[1]

        '''Process if node in visited_nodes'''
        if traverse_node[0] not in visited_nodes:

            '''Add traverse node to the visited_nodes list'''
            visited_nodes.append(traverse_node[0])

            '''Get successors for the parent and push them in the queue'''
            successor_list = problem.getSuccessors(traverse_node[0])

            for successor in successor_list:
                if successor[0] not in visited_nodes:
                    '''While pushing the node in the priority queue add the cost of it's
                        parent and it's own heuristic as well. This will allow us to keep track of total cost of
                        the path to the node with admissible heuristic too.'''
                    direction_list = traverse_node[1] + [successor[1]]
                    astar_queue.push((successor[0], direction_list, successor[2] + traverse_node[2]), successor[2] + traverse_node[2]+heuristic(successor[0],problem))

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
