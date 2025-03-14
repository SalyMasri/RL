#!/usr/bin/env python3
import random
import numpy as np

from agent import Fish
from communicator import Communicator
from shared import SettingLoader


class FishesModelling:
    def init_fishes(self, n):
        fishes = {}
        for i in range(n):
            fishes["fish" + str(i)] = Fish()
        self.fishes = fishes


class PlayerController(SettingLoader, Communicator):
    def __init__(self):
        SettingLoader.__init__(self)
        Communicator.__init__(self)
        self.space_subdivisions = 10
        self.actions = None
        self.action_list = None
        self.states = None
        self.init_state = None
        self.ind2state = None
        self.state2ind = None
        self.alpha = 0
        self.gamma = 0
        self.episode_max = 300

    def init_states(self):
        ind2state = {}
        state2ind = {}
        count = 0
        for row in range(self.space_subdivisions):
            for col in range(self.space_subdivisions):
                ind2state[(col, row)] = count
                state2ind[count] = [col, row]
                count += 1
        self.ind2state = ind2state
        self.state2ind = state2ind

    def init_actions(self):
        self.actions = {
            "left":  (-1, 0),
            "right": ( 1, 0),
            "down":  ( 0,-1),
            "up":    ( 0, 1)
        }
        self.action_list = list(self.actions.keys())

    def allowed_movements(self):
        """Pre-compute which discrete actions are valid from each state index."""
        self.allowed_moves = {}
        for (x, y) in self.ind2state.keys():
            state_idx = self.ind2state[(x, y)]
            valid_list = []
            # if we can go right
            if x < self.space_subdivisions - 1:
                valid_list.append(1)  # index=1 => "right"
            # if we can go left
            if x > 0:
                valid_list.append(0)  # index=0 => "left"
            # if we can go up
            if y < self.space_subdivisions - 1:
                valid_list.append(3)  # index=3 => "up"
            # if we can go down
            if y > 0:
                valid_list.append(2)  # index=2 => "down"
            self.allowed_moves[state_idx] = valid_list

    def player_loop(self):
        pass


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        while True:
            msg = self.receiver()
            if msg["game_over"]:
                return


def epsilon_greedy(
    Q,
    state,
    all_actions,
    current_total_steps=0,
    epsilon_initial=1.0,
    epsilon_final=0.2,
    anneal_timesteps=10000,
    eps_type="constant"
):
    """
    Epsilon-greedy selection. all_actions is a list of valid action indices
    for the current state. We either pick random among them (with probability epsilon)
    or pick the best Q-value among them (with probability 1-epsilon).
    """
    if eps_type == 'constant':
        epsilon = epsilon_final
        if np.random.rand() < epsilon:
            # random
            action = np.random.choice(all_actions)
        else:
            # greedy wrt Q
            q_vals = [Q[state,a] for a in all_actions]
            best_idx = np.nanargmax(q_vals)
            action = all_actions[best_idx]

    elif eps_type == 'linear':
        # linear annealing from epsilon_initial to epsilon_final
        fraction = min(float(current_total_steps)/float(anneal_timesteps), 1.0)
        epsilon = epsilon_initial + fraction*(epsilon_final - epsilon_initial)
        if np.random.rand() < epsilon:
            action = np.random.choice(all_actions)
        else:
            q_vals = [Q[state,a] for a in all_actions]
            best_idx = np.nanargmax(q_vals)
            action = all_actions[best_idx]
    else:
        raise ValueError("eps_type unknown: " + str(eps_type))

    return action


class PlayerControllerRL(PlayerController, FishesModelling):
    def __init__(self):
        super().__init__()

    def player_loop(self):
        # Read hyperparams from (student_3_2_1.py) or (student_3_2_2.py), etc.
        self.init_actions()
        self.init_states()
        self.allowed_movements()
        self.alpha = self.settings.alpha
        self.gamma = self.settings.gamma
        self.epsilon_initial = self.settings.epsilon_initial
        self.epsilon_final = self.settings.epsilon_final
        self.annealing_timesteps = self.settings.annealing_timesteps
        self.threshold = self.settings.threshold
        self.episode_max = self.settings.episode_max

        # Do Q-learning
        Q = self.q_learning()

        # Build final policy from Q
        policy = self.get_policy(Q)

        # Send final policy
        msg = {"policy": policy, "exploration": False}
        self.sender(msg)

        # Wait for final acknowledgment
        _ = self.receiver()
        print("Q-learning returning")
        return

    def q_learning(self):
        ns = len(self.state2ind)
        na = len(self.actions)

        # Initialize Q to zeros (common approach).
        Q = np.zeros((ns, na), dtype=np.float32)

        # Mark invalid actions as NaN so they won't be selected by argmax():
        for s in range(ns):
            valid_acts = self.allowed_moves[s]
            for a in range(na):
                if a not in valid_acts:
                    Q[s,a] = np.nan

        Q_old = Q.copy()
        diff = np.inf

        # Possibly override your episode count to something bigger if needed:
        # self.episode_max = 800

        # Also set a step cutoff per episode to avoid infinite loops:
        max_steps_per_episode = 300

        # We fetch the diver's start from your environment settings
        init_pos_tuple = self.settings.init_pos_diver
        init_pos = self.ind2state[(init_pos_tuple[0], init_pos_tuple[1])]

        current_total_steps = 0
        episode = 0

        while (episode < self.episode_max) and (diff > self.threshold):
            s_current = init_pos
            end_episode = False
            steps_this_ep = 0
            # We must re-initialize R_total each episode to avoid error:
            R_total = 0

            while (not end_episode) and (steps_this_ep < self.episode_max):
                # pick an action via epsilon_greedy
                valid_acts = self.allowed_moves[s_current]
                action = epsilon_greedy(
                    Q=Q,
                    state=s_current,
                    all_actions=valid_acts,
                    current_total_steps=current_total_steps,
                    epsilon_initial=self.epsilon_initial,
                    epsilon_final=self.epsilon_final,
                    anneal_timesteps=self.annealing_timesteps,
                    eps_type="linear"  # or "constant"
                )

                # Send to environment
                action_str = self.action_list[action]
                msg_out = {"action": action_str, "exploration": True}
                self.sender(msg_out)

                # Receive next state info
                msg_in = self.receiver()
                R = msg_in["reward"]
                end_episode = msg_in["end_episode"]
                s_next_tuple = msg_in["state"]
                s_next = self.ind2state[s_next_tuple]

                # Q-learning update
                old_val = Q[s_current, action]
                best_future = np.nanmax(Q[s_next,:])  # best Q-value in next state
                td_target = R + self.gamma * best_future
                Q[s_current, action] = old_val + self.alpha * (td_target - old_val)

                # Move on
                s_current = s_next
                R_total += R
                current_total_steps += 1
                steps_this_ep += 1

            # Check difference for convergence
            diff = np.nanmean(np.abs(Q - Q_old))
            Q_old[:] = Q
            episode += 1

            # Print only every 10 episodes to avoid spam
            if episode % 10 == 0:
                print(f"Ep={episode}, Steps={steps_this_ep}, R={R_total}, Diff={diff:.3e}, TotSteps={current_total_steps}")

        return Q

    def get_policy(self, Q):
        best_actions = np.nanargmax(Q, axis=1)
        policy = {}
        for s_id, coords in self.state2ind.items():
            a_id = best_actions[s_id]
            policy[(coords[0], coords[1])] = self.action_list[a_id]
        return policy


class PlayerControllerRandom(PlayerController):
    def __init__(self):
        super().__init__()

    def player_loop(self):
        # Random agent, no Q-learning
        self.init_actions()
        self.init_states()
        self.allowed_movements()
        self.episode_max = self.settings.episode_max

        N = self.random_agent()

        # Build a "policy"
        policy = self.get_policy(N)
        msg = {"policy": policy, "exploration": False}
        self.sender(msg)
        _ = self.receiver()
        print("Random Agent returning")

    def random_agent(self):
        ns = len(self.state2ind)
        na = len(self.actions)
        N = np.zeros((ns, na), dtype=np.float32)

        init_pos_tuple = self.settings.init_pos_diver
        init_pos = self.ind2state[(init_pos_tuple[0], init_pos_tuple[1])]

        max_steps_per_episode = 200
        episode = 0
        while episode < self.episode_max:
            end_episode = False
            s_current = init_pos
            steps_this_ep = 0
            R_total = 0

            while (not end_episode) and (steps_this_ep < max_steps_per_episode):
                valid_acts = self.allowed_moves[s_current]
                action = np.random.choice(valid_acts)
                N[s_current, action] += 1

                msg_out = {"action": self.action_list[action], "exploration": True}
                self.sender(msg_out)

                msg_in = self.receiver()
                R = msg_in["reward"]
                end_episode = msg_in["end_episode"]
                s_next = self.ind2state[msg_in["state"]]

                s_current = s_next
                R_total += R
                steps_this_ep += 1

            print(f"[Random] Ep={episode}, Steps={steps_this_ep}, R={R_total}")
            episode += 1

        return N

    def get_policy(self, Q):
        """For each state, pick argmax ignoring NaNs."""
        best_actions = []
        for row in Q:
            try:
                best_actions.append(np.nanargmax(row))
            except:
                best_actions.append(np.random.choice([0,1,2,3]))

        policy = {}
        for s_id, coords in self.state2ind.items():
            a_id = best_actions[s_id]
            policy[(coords[0], coords[1])] = self.action_list[a_id]
        return policy


class ScheduleLinear:
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t)/float(self.schedule_timesteps), 1.0)
        return self.initial_p + fraction*(self.final_p - self.initial_p)