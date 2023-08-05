# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        mdp = self.mdp

        states = mdp.getStates()
        for _it in range(self.iterations):
            values = util.Counter()
            for state in states:
                q_vals = util.Counter()
                if not mdp.isTerminal(state):
                    for action in mdp.getPossibleActions(state):
                        q_vals[action] = self.computeQValueFromValues(state, action)
                    values[state] = max(q_vals.values())

            self.values = values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        mdp = self.mdp

        ret = 0.0
        state_prob = mdp.getTransitionStatesAndProbs(state, action)
        for next_state, prob in state_prob:
            ret += prob * (mdp.getReward(state, action, next_state) \
                           + self.discount * self.values[next_state])

        return ret

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        mdp = self.mdp

        if mdp.isTerminal(state):
            return None

        q_vals = util.Counter()
        possible_actions = mdp.getPossibleActions(state)
        for action in possible_actions:
            q_vals[action] = self.computeQValueFromValues(state, action)

        return q_vals.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        mdp = self.mdp

        states = mdp.getStates()
        for it in range(self.iterations):
            state = states[it % len(states)]
            q_vals = util.Counter()
            if not mdp.isTerminal(state):
                for action in mdp.getPossibleActions(state):
                    q_vals[action] = self.computeQValueFromValues(state, action)

                self.values[state] = max(q_vals.values())

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        mdp = self.mdp

        states = mdp.getStates()
        pq = util.PriorityQueue()
        adj_list = dict((state, set()) for state in states)

        for state in states:
            if mdp.isTerminal(state):
                continue
            q_vals = util.Counter()
            possible_actions = mdp.getPossibleActions(state)
            for action in possible_actions:
                state_prob = mdp.getTransitionStatesAndProbs(state, action)
                for next_state, prob in state_prob:
                    adj_list[next_state].add(state)
                q_vals[action] = self.getQValue(state, action)
            weight = (-1) * abs(self.values[state] - max(q_vals.values()))
            pq.update(state, weight)

        for _it in range(self.iterations):
            if pq.isEmpty():
                break
            state = pq.pop()
            if mdp.isTerminal(state):
                continue

            q_vals = util.Counter()
            possible_actions = mdp.getPossibleActions(state)
            for action in possible_actions:
                q_vals[action] = self.getQValue(state, action)
            self.values[state] = max(q_vals.values())

            for adj in adj_list[state]:
                q_vals = util.Counter()
                possible_actions = mdp.getPossibleActions(adj)
                for action in possible_actions:
                    q_vals[action] = self.getQValue(adj, action)
                weight = abs(self.values[adj] - max(q_vals.values()))
                if weight > self.theta:
                    pq.update(adj, -weight)

