# qlearningAgents.py
# ------------------

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
    """
    def __init__(self, ghostAgents=None, **args):
        "Initialize Q-values"
        ReinforcementAgent.__init__(self, **args)
        self.actions = {"North":0, "East":1, "South":2, "West":3, "Stop":4}
        self.g_dir = {"N":0, "NE":1, "E":2,"SE":3,"S":4,"SW":5, "W":6,"NW":7}       # Direccion donde se situa un fantasma
        self.legal = {"N":0, "S":1, "E":2, "W":3, "NS":4, "NE":5, "NW":6, "SE":7, "SW":8, "EW":9, "NSE":10, "NSW":11, "NEW":12, "SEW":13, "NSEW":14 }
        self.table_file = open("qtable.txt", "r+")
        self.q_table = self.readQtable()
        #self.memoria = []               # Evitar bucles
        self.last_action = "Stop"  # REVISAR
        # self.epsilon = 0.7
        # self.alpha = 0.4
        # self.discount = 0.9

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()
        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item)+" ")
            self.table_file.write("\n")

    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")

    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()

    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        """
        my_state = self.extractAttributes(state)
        return self.actions[my_state[0]] + 5* self.g_dir[my_state[1]] + 40 * self.legal[my_state[2]] +  600 * my_state[3] + 1200 * my_state[4]

    def getQValue(self, state, action):

        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        position = self.computePosition(state)
        action_column = self.actions[action]
        return self.q_table[position][action_column]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = state.getLegalActions()
        if len(legalActions)==0:
          return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = state.getLegalActions()
        if len(legalActions)==0:
          return None

        legalActions.remove("Stop")
        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value
        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """

        # Pick Action
        legalActions = state.getLegalActions()

        action = None

        if len(legalActions) == 0:
             return action

        legalActions.remove("Stop")
        flip = util.flipCoin(self.epsilon)

        if flip:
            return random.choice(legalActions)
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        Good Terminal state -> reward 1
        Bad Terminal state -> reward -1
        Otherwise -> reward 0

        Q-Learning update:

        if terminal_state:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        else:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))

        """
        reward = self.getReward(reward, state, action)
        position = self.computePosition(state)
        action_column = self.actions[action]

        if len(nextState.getLegalActions())==0:
            self.q_table[position][action_column] = (1-self.alpha) * self.getQValue(state, action) + self.alpha * (reward + 0)
            self.last_action = "Stop"
        else:
            max_action = self.getPolicy(nextState)
            self.q_table[position][action_column] = (1-self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.discount * self.getQValue(nextState, max_action))
            self.last_action = action

    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)

    def getReward(self, reward, state, action):
        if (self.last_action == "East" and action =="West") or (self.last_action == "West" and action =="East"):
            reward -= 1000

        if (self.last_action == "North" and action == "South") or (self.last_action == "South" and action == "North"):
            reward -= 1000
        # if state.getPacmanPosition() in self.memoria:
        #     reward-=1000
        #     print("castigo")
        # self.memoria.append(state.getPacmanPosition())
        # #print(self.memoria)
        # if len(self.memoria) != 0 and len(self.memoria)%5 == 0:
        #     self.memoria.pop(0)
        return reward

    def extractAttributes(self, state):
        """Método encargado de devolver los atributos seleccionados para calcular
        la fila correspondiente al estado recibido"""
        pacman_direction = ""
        pacman_pos = state.getPacmanPosition()

        # extrayendo la posicion del objeto más cercano
        ghosts_positions = state.getGhostPositions()
        ghosts_distances = state.data.ghostDistances
        min_distance = 9999
        for i in ghosts_distances:
            if i is not None:
                if i <= min_distance:
                    min_distance = i
        g_pos = ghosts_positions[ghosts_distances.index(min_distance)]
        food_dist, food_pos = state.getDistanceNearestFood()

        if food_dist is not None and food_dist < min_distance:
            dir_x, dir_y = food_pos[0] - pacman_pos[0], food_pos[1] - pacman_pos[1]
        else:
            dir_x, dir_y = g_pos[0] - pacman_pos[0], g_pos[1] - pacman_pos[1]

        if dir_y < 0:
            pacman_direction += "S"

        elif dir_y > 0:
            pacman_direction += "N"

        if dir_x < 0:
            pacman_direction += "W"

        elif dir_x > 0:
            pacman_direction += "E"
        # print(pacman_direction)

        # acciones legales de pacman
        pacman_actions = state.getLegalPacmanActions() # Acciones legales del Pacman
        pacman_actions.remove("Stop")
        legal_actions = ""
        for action in pacman_actions:
            legal_actions += action[0]
        # print(legal_actions)

        gap = self.checkBoundaries(state, pacman_direction)
        gapVertical = self.checkVerticalBoundaries(state, pacman_direction)

        return (self.last_action,pacman_direction, legal_actions, gap, gapVertical)

    def checkBoundaries(self, state, direction):
        pacman_pos = state.getPacmanPosition()
        pos_x, pos_y = pacman_pos[0], pacman_pos[1]
        walls = state.getWalls()
        width = walls.width-1
        height = walls.height-1
        found = 0
        if direction[0] == "N":
            if walls[pos_x][pos_y+1] and pos_y+1 != height:  # Muro norte distinto de los bordes
                pos_x-=1
                counter = 0
                while(not found and pos_x != 0):             # Recorrer muro
                    if not walls[pos_x][pos_y+1]:
                        found = 1
                    pos_x-=1

        elif direction[0] == "S":
            if walls[pos_x][pos_y-1] and pos_y-1 != 2:     # Muro sur distinto de los bordes
                pos_x-=1
                while(not found and pos_x != 0):
                    if not walls[pos_x][pos_y-1]:
                        found = 1
                    pos_x-=1

        return found

        # print(walls[pos_x][pos_y+1])
        # print(walls)
        # print(walls[0])
        # print(walls[width])

    def checkVerticalBoundaries(self, state, direction):
        pacman_pos = state.getPacmanPosition()
        pos_x, pos_y = pacman_pos[0], pacman_pos[1]
        walls = state.getWalls()
        width = walls.width-1
        height = walls.height-1
        found = 0
        if (len(direction) == 1 and direction[0] == "E") or (len(direction) == 2 and direction[1] == "E"):
            if walls[pos_x+1][pos_y] and pos_x+1 != width:  # Muro este distinto de los bordes
                pos_y-=1
                counter = 0
                while(not found and pos_y != 0):             # Recorrer muro
                    if not walls[pos_x+1][pos_y]:
                        found = 1
                    pos_y-=1

        elif (len(direction) == 1 and direction[0] == "W") or (len(direction) == 2 and direction[1] == "W"):
            if walls[pos_x+1][pos_y] and pos_x-1 != 0:     # Muro oeste distinto de los bordes
                pos_y-=1
                while(not found and pos_y != 0):
                    if not walls[pos_x-1][pos_y]:
                        found = 1
                    pos_y-=1
        #{"N":0, "NE":1, "E":2,"SE":3,"S":4,"SW":5, "W":6,"NW":7}
        return found

class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        feats = self.featExtractor.getFeatures(state, action)
        for f in feats:
          self.weights[f] = self.weights[f] + self.alpha * feats[f]*((reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action))

        # util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
