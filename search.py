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
    return  [s, s, w, s, w, w, s, w]

# def df_recursive(graph,node,visited,action):
#     # visited = list(visited)
#     visited.add(node)
#     if graph.isGoalState(node):
#         action.append('Stop')
#         return
#
#     for neighbor in graph.getSuccessors(node):
#         if neighbor[0] not in visited:
#             action.append(neighbor[1])
#             df_recursive(graph,neighbor[0],visited,action)


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    stack = util.Stack()
    stack.push((problem.getStartState(),[]))
    visited = set()

    while stack.isEmpty() == False:
        current_node, path = stack.pop()

        # If we found the goal, return the path (list of directions)
        if problem.isGoalState(current_node):
            return path

        # Mark current node as visited
        visited.add(current_node)

        # Explore neighbors (direction, neighbor node) pairs
        for neighbor in problem.getSuccessors(current_node):
            if neighbor[0] not in visited :
                # Add the neighbor and updated path to the stack
                stack.push((neighbor[0], path + [neighbor[1]]))

    return None  # No path found


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    queue.push((problem.getStartState(), []))
    visited = set(problem.getStartState())

    while queue.isEmpty() == False:
        current_node, path = queue.pop()

        # If we found the goal, return the path (list of directions)
        if problem.isGoalState(current_node):
            return path

        # Mark current node as visited


        # Explore neighbors (direction, neighbor node) pairs
        for neighbor in problem.getSuccessors(current_node):
            if neighbor[0] not in visited :
                visited.add(neighbor[0])
                # Add the neighbor and updated path to the stack
                queue.push((neighbor[0], path + [neighbor[1]]))

    return None  # No path found

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    queue = util.PriorityQueue()
    queue.push((problem.getStartState(), []),0)
    visited = {problem.getStartState():0}

    while queue.isEmpty() == False:
        current_node,path = queue.pop()
        cost = visited[current_node]
        # If we found the goal, return the path (list of directions)
        if problem.isGoalState(current_node):
            return path
        # Mark current node as visit
        # Explore neighbors (direction, neighbor node) pairs
        for neighbor in problem.getSuccessors(current_node):
            if neighbor[0] not in visited:
                visited[neighbor[0]] = cost + neighbor[2]
                # Add the neighbor and updated path to the stack
                queue.update((neighbor[0],path + [neighbor[1]]),cost + neighbor[2])
            else:
                if cost+neighbor[2] < visited[neighbor[0]]:
                    visited[neighbor[0]] = cost + neighbor[2]
                    queue.update((neighbor[0], path + [neighbor[1]]), cost + neighbor[2])




    return None  # No path found

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    pq = util.PriorityQueue() # Priority queue for A*
    pq.push((problem.getStartState(), [], heuristic(problem.getStartState(),problem)), heuristic(problem.getStartState(),problem))  # Push start node with f(start) = g(start) + h(start)

    visited = {problem.getStartState():0}  # To track visited nodes and their g-values (actual cost)

    while not pq.isEmpty():
        current_node, path, current_f_cost = pq.pop()  # Get node with lowest f(n)
        cost = visited[current_node]
        # Goal check when popping from the queue
        if problem.isGoalState(current_node):
            return path

        for neighbor in problem.getSuccessors(current_node):
            if neighbor[0] not in visited:
                visited[neighbor[0]] = cost + neighbor[2]
                f_cost = visited[neighbor[0]] + heuristic(neighbor[0],problem)
                # Add the neighbor and updated path to the stack
                pq.update((neighbor[0], path + [neighbor[1]],f_cost), f_cost)
            else:
                if cost + neighbor[2] < visited[neighbor[0]]:
                    visited[neighbor[0]] = cost + neighbor[2]
                    f_cost = visited[neighbor[0]] + heuristic(neighbor[0], problem)
                    pq.update((neighbor[0], path + [neighbor[1]],f_cost), f_cost)

    return None  # Return None if no path is found


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
