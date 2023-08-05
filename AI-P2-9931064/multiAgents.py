# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        from util import manhattanDistance
        RUN_PACMAN = 10000

        eval_res = successorGameState.getScore()
        food_dist = [manhattanDistance(newPos, food) for food in newFood.asList()]
        ghost_dist = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]

        for it in range(len(ghost_dist)):
            if newScaredTimes[it] == 0:
                if ghost_dist[it] > 1:
                    eval_res += 1.0 / ghost_dist[it]
                else:
                    eval_res -= RUN_PACMAN
            else:
                eval_res += ghost_dist[it]

        return (eval_res + 1.0 / (min(food_dist) if len(food_dist) > 0 else 1.0))

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        action, val = self.get_val(gameState, 0)
        return action

    def get_min(self, gameState, depth):
        import math

        index = depth % gameState.getNumAgents()
        actions = gameState.getLegalActions(index)

        if len(actions) == 0:
            return '', self.evaluationFunction(gameState)

        res_action = ''
        res_val = math.inf

        for next_action in actions:
            succ = gameState.generateSuccessor(index, next_action)
            action, val = self.get_val(succ, depth + 1)
            if val < res_val:
                res_action, res_val = (next_action, val)

        return (res_action, res_val)

    def get_max(self, gameState, depth):
        import math

        index = 0
        actions = gameState.getLegalActions(index)

        if len(actions) == 0:
            return '', self.evaluationFunction(gameState)

        res_action = ''
        res_val = -math.inf

        for next_action in actions:
            succ = gameState.generateSuccessor(index, next_action)
            action, val = self.get_val(succ, depth + 1)
            if val > res_val:
                res_action, res_val = (next_action, val)

        return (res_action, res_val)

    def get_val(self, gameState, depth):
        num_agents = gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose() or depth == self.depth * num_agents:
            return '', self.evaluationFunction(gameState)

        return self.get_max(gameState, depth) if depth % num_agents == 0 else self.get_min(gameState, depth)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        import math

        alpha, beta = -math.inf, math.inf
        action, val = self.get_val(gameState, 0, (alpha, beta))
        return action

    def get_min(self, gameState, depth, args):
        import math

        index = depth % gameState.getNumAgents()
        actions = gameState.getLegalActions(index)
        (alpha, beta) = args

        if len(actions) == 0:
            return '', self.evaluationFunction(gameState)

        res_action = ''
        res_val = math.inf

        for next_action in actions:
            succ = gameState.generateSuccessor(index, next_action)
            action, val = self.get_val(succ, depth + 1, (alpha, beta))
            if val < res_val:
                res_action, res_val = (next_action, val)

            if alpha > res_val:
                return (res_action, res_val)
            beta = min(beta, res_val)

        return (res_action, res_val)

    def get_max(self, gameState, depth, args):
        import math

        index = 0
        actions = gameState.getLegalActions(index)
        (alpha, beta) = args

        if len(actions) == 0:
            return '', self.evaluationFunction(gameState)

        res_action = ''
        res_val = -math.inf

        for next_action in actions:
            succ = gameState.generateSuccessor(index, next_action)
            action, val = self.get_val(succ, depth + 1, (alpha, beta))
            if val > res_val:
                res_action, res_val = (next_action, val)

            if beta < res_val:
                return (res_action, res_val)
            alpha = max(alpha, res_val)

        return (res_action, res_val)

    def get_val(self, gameState, depth, args):
        num_agents = gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose() or depth == self.depth * num_agents:
            return '', self.evaluationFunction(gameState)

        return self.get_max(gameState, depth, args) if depth % num_agents == 0 else self.get_min(gameState, depth, args)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        action, val = self.get_val(gameState, 0)
        return action

    def get_exp(self, gameState, depth):
        import math

        index = depth % gameState.getNumAgents()
        actions = gameState.getLegalActions(index)

        if len(actions) == 0:
            return '', self.evaluationFunction(gameState)

        res_action = ''
        res_val = 0

        for next_action in actions:
            succ = gameState.generateSuccessor(index, next_action)
            action, val = self.get_val(succ, depth + 1)
            res_val += val * (1.0 / len(actions))

        return (res_action, res_val)

    def get_max(self, gameState, depth):
        import math

        index = 0
        actions = gameState.getLegalActions(index)

        if len(actions) == 0:
            return '', self.evaluationFunction(gameState)

        res_action = ''
        res_val = -math.inf

        for next_action in actions:
            succ = gameState.generateSuccessor(index, next_action)
            action, val = self.get_val(succ, depth + 1)
            if val > res_val:
                res_action, res_val = (next_action, val)

        return (res_action, res_val)

    def get_val(self, gameState, depth):
        num_agents = gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose() or depth == self.depth * num_agents:
            return '', self.evaluationFunction(gameState)

        return self.get_max(gameState, depth) if depth % num_agents == 0 else self.get_exp(gameState, depth)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()

    from util import manhattanDistance

    eval_res = currentGameState.getScore()
    total_capsules = len(currentGameState.getCapsules())
    total_eaten_food = len(foods.asList(False))
    total_food_dist = sum([manhattanDistance(pacmanPosition, food) for food in foods.asList()])
    total_ghost_dist = sum([manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates])

    total_food_dist = (1.0 / total_food_dist if total_food_dist > 0 else 0)
    total_times_scared = sum(scaredTimers)

    eval_res = total_eaten_food + total_food_dist
    total_times_scared = max(total_times_scared, 0)

    eval_res += total_times_scared
    if total_times_scared > 0:
        eval_res -= total_capsules + total_ghost_dist
    else:
        eval_res += total_capsules + total_ghost_dist

    return eval_res

# Abbreviation
better = betterEvaluationFunction
