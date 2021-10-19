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
import random, util, math

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

    # sigmoid function returns value between 0 and 1
    # credit https://www.delftstack.com/howto/python/sigmoid-function-python/
    def sigmoid(self, x):
        if x >= 0:
            z = math.exp(-x)
            sig = 1 / (1 + z)
            return sig
        else:
            z = math.exp(x)
            sig = z / (1 + z)
            return sig

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

        "*** YOUR CODE HERE ***"
        # maximize utility
        utility = 0

        # get score (maximize)
        score = successorGameState.getScore()

        # get the closest food distance (minimize)
        # closest_food_distance = util.manhattanDistance(newPos, )
        closest_food_distance = float('inf')
        for food in newFood.asList():
            food_distance = util.manhattanDistance(newPos, food)
            if food_distance < closest_food_distance:
                closest_food_distance = food_distance
        closest_food_distance = 1/closest_food_distance

        # get how much food is left (minimize)
        num_food_left = 1/max(len(newFood.asList()), 1)

        # get how far closest monster is away (maximize)
        closest_monster_distance = float('inf')
        for monster in newGhostStates:
            monster_pos = monster.getPosition()
            distance_to_monster = util.manhattanDistance(newPos, monster_pos)
            if distance_to_monster < closest_monster_distance:
                closest_monster_distance = distance_to_monster

        utility = score + self.sigmoid(closest_food_distance) + self.sigmoid(num_food_left) + self.sigmoid(closest_monster_distance)
        # print('score: ' + str(score))
        # print('closest_food_distance: ' + str(self.sigmoid(closest_food_distance)))
        # print('num_food_left: ' + str(self.sigmoid(num_food_left)))
        # print('total_monster_distance: ' + str(self.sigmoid(closest_monster_distance)))
        return utility

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
        "*** YOUR CODE HERE ***"


        def value(gameState, agentIndex, depth):
            # if state is base case
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                # return utility
                return self.evaluationFunction(gameState), None

            nextAgentIndex = agentIndex + 1

            # if agent is pacman
                # initialize v = -inf
                # for each successor of state
                    # v = max(v, value(successor))
                # return v

            if agentIndex == 0:
                v = float('-inf')
                e = None
                for action in gameState.getLegalActions(agentIndex):
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    successorValue, _ = value(successorState, nextAgentIndex, depth)
                    if successorValue > v:
                        v = successorValue
                        e = action
                return v, e

            # if agent is a ghost
                # initialize v = inf
                # for each successor of state
                    # v = min(v, value(successor))
                # return v

            if gameState.getNumAgents() == nextAgentIndex:
                nextAgentIndex = 0
                depth = depth + 1

            v = float('inf')
            e = None
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                successorValue, _ = value(successorState, nextAgentIndex, depth)
                if successorValue < v:
                    v = successorValue
                    e = action
            return v, e

        v, e = value(gameState, 0, 0)
        return e

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def value(gameState, agentIndex, depth, alpha, beta):
            # if state is base case
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                # return utility
                return self.evaluationFunction(gameState), None

            nextAgentIndex = agentIndex + 1

            # if agent is pacman
                # initialize v = -inf
                # for each successor of state
                    # v = max(v, value(successor))
                # return v

            if agentIndex == 0:
                v = float('-inf')
                e = None
                for action in gameState.getLegalActions(agentIndex):
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    successorValue, _ = value(successorState, nextAgentIndex, depth, alpha, beta)
                    if successorValue > v:
                        v = successorValue
                        e = action
                    alpha = max(alpha, successorValue)
                    if beta < alpha:
                        break
                return v, e

            # if agent is a ghost
                # initialize v = inf
                # for each successor of state
                    # v = min(v, value(successor))
                # return v

            if gameState.getNumAgents() == nextAgentIndex:
                nextAgentIndex = 0
                depth = depth + 1

            v = float('inf')
            e = None
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                successorValue, _ = value(successorState, nextAgentIndex, depth, alpha, beta)
                if successorValue < v:
                    v = successorValue
                    e = action
                beta = min(beta, successorValue)
                if beta < alpha:
                    break
            return v, e

        v, e = value(gameState, 0, 0, float('-inf'), float('inf'))
        return e

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
        "*** YOUR CODE HERE ***"
        def value(gameState, agentIndex, depth):
            # if state is base case
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                # return utility
                return self.evaluationFunction(gameState), None

            nextAgentIndex = agentIndex + 1

            # if agent is pacman
                # initialize v = -inf
                # for each successor of state
                    # v = max(v, value(successor))
                # return v

            if agentIndex == 0:
                v = float('-inf')
                e = None
                for action in gameState.getLegalActions(agentIndex):
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    successorValue, _ = value(successorState, nextAgentIndex, depth)
                    if successorValue > v:
                        v = successorValue
                        e = action
                return v, e

            # if agent is a ghost
                # initialize v = inf
                # for each successor of state
                    # v = min(v, value(successor))
                # return v

            if gameState.getNumAgents() == nextAgentIndex:
                nextAgentIndex = 0
                depth = depth + 1

            v = 0
            p = 1/len(gameState.getLegalActions(agentIndex))
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                successorValue, _ = value(successorState, nextAgentIndex, depth)
                v += (p * successorValue)
            return v, _

        v, e = value(gameState, 0, 0)
        return e

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # sigmoid function returns value between 0 and 1
    # credit https://www.delftstack.com/howto/python/sigmoid-function-python/
    def sigmoid(x):
        if x >= 0:
            z = math.exp(-x)
            sig = 1 / (1 + z)
            return sig
        else:
            z = math.exp(x)
            sig = z / (1 + z)
            return sig

    # essential info
    pos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # evaluation function
    ef = 0

    # get score of the game
    score = currentGameState.getScore() # always maximize

    # get how close a capsule is
    closest_capsule_distance = float('inf')
    for capsule in capsules:
        capsule_distance = util.manhattanDistance(pos, capsule)
        if capsule_distance < closest_capsule_distance:
            closest_capsule_distance = capsule_distance
    closest_capsule_distance = 1/closest_capsule_distance # always minimize

    # get distance to closest food
    closest_food_distance = float('inf')
    for food in foodGrid.asList():
        food_distance = util.manhattanDistance(pos, food)
        if food_distance < closest_food_distance:
            closest_food_distance = food_distance
    closest_food_distance = 1/closest_food_distance # always minimize

    # get how many foods are left
    num_food_left = 1/max(len(foodGrid.asList()), 1) # always minimize

    # get distance from closest ghost
    closest_monster_distance = float('inf')
    closest_monster = None
    for monster in ghostStates:
        monster_pos = monster.getPosition()
        distance_to_monster = util.manhattanDistance(pos, monster_pos)
        if distance_to_monster < closest_monster_distance:
            closest_monster_distance = distance_to_monster
            closest_monster = monster

    # maximize score
    ef += score
    # minimize how many foods are left
    ef += sigmoid(num_food_left)
    # minimize distance to closest food
    ef += 20 * sigmoid(closest_food_distance)
    # minimize how close a capsule is
    ef += 100 * sigmoid(closest_capsule_distance)

    if closest_monster:
        # if capsule for closest monster is ON
        if closest_monster.scaredTimer > 0:
            # minimize distance to closest ghost
            ef += 100 * sigmoid(1/closest_monster_distance)
        else:
            # capsule is OFF
            # maximize distance to closest ghost
            ef += sigmoid(closest_monster_distance)

    # print(str(scaredTimes))
    return ef


# Abbreviation
better = betterEvaluationFunction
