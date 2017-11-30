# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Tutorial sample #6: Discrete movement, rewards, and learning

# The "Cliff Walking" example using Q-learning.
# From pages 148-150 of:
# Richard S. Sutton and Andrews G. Barto
# Reinforcement Learning, An Introduction
# MIT Press, 1998

import numpy as np
from PIL import Image  # save images

import MalmoPython
import json
import logging
import os
import random
import sys
import time
import socket
import uuid


class TabQAgent:
    """Tabular Q-learning agent for discrete state/action spaces."""
    def __init__(self, agent_host, dqn_port, task):
        self.action_table = {
            'w': 'move 1',
            's': 'move -1',
            'a': 'strafe -1',
            'd': 'strafe 1',
            'j': 'turn -1',  # left arrow
            'i': 'look -1',  # up arrow
            'l': 'turn 1',  # right arrow
            'k': 'look 1',  # down arrow
            'q': 'q',
            ' ': 'jumpmove 1',
            'e': 'attack 1',
            'u': 'use 1',
            '1': 'hotbar.1 ',
            '2': 'hotbar.2 ',
            '3': 'hotbar.3 ',
            '4': 'hotbar.4 ',
            '5': 'hotbar.5 ',
            '6': 'hotbar.6 ',
            '7': 'hotbar.7 ',
            '8': 'hotbar.8 ',
            '9': 'hotbar.9 '
        }
        self.int_to_action_table = {
            1: 'move 1',
            2: 'turn -1',  # left arrow
            3: 'turn 1',  # right arrow
            4: 'look',  # up arrow
            # 5: 'look 1',  # down arrow
            5: 'attack 1',
            # 6: 'use 1',
            # 7: 'swap',
            9: 'newgame',
            0: 'nop'
        }
        self.perStepReward = -1
        self.maxSteps = 60
        self.sleepTime = 0.05 / 2
        self.curStepReward = self.perStepReward
        self.pitch = 0
        self.numSteps = 0
        self.turned = 0
        self.sock = None
        self.connection = None
        self.sock = None
        self.connected = False
        self.port = dqn_port
        self.isTerminal = False
        self.doorBroken = False
        self.room = 0

        self.epsilon = 0.01  # chance of taking a random action instead of the best

        self.logger = logging.getLogger(__name__)
        if False:  # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.root = None
        self.prev_s = None
        self.prev_a = None
        self.task = task
        self.agent_host = agent_host

    def newGame(self):
        self.room = 0
        if self.task in ['full']:
            self.room = random.randint(1, 2)
        if self.task in ['room1'] or self.room == 1:
            x = 13.5 - random.randint(0, 17)
            z = 0.5 + random.randint(0, 8)
            y = 46
            if (3.5 <= x <= 8.5) and (3.5 <= z <= 7.5):
                z = 8.5
        elif self.task in ['room2'] or self.room == 2:
            legitimate_starting_point = False
            while not legitimate_starting_point:
                x = random.randint(-2, 8) + 0.5
                y = 46
                z = random.randint(11, 20) + 0.5
                if not(-1 <= x <= 2 and 14 <= z <= 17):  # stairs
                    if not(5 <= x <= 6 and 18 <= z <= 19):  # table
                        if not(x == -2 and 18 <= z <= 19):  # bed
                            if not(x == 9 and 16 <= z <= 20):   # bookshelf
                                legitimate_starting_point = True

        if self.room == 0:
            self.room += 1    # for H-DRLN we need room >= 1 so if we run a single room just pass room #1 as argument.

        self.agent_host.sendCommand('chat /tp Cristina ' + str(x) + ' ' + str(y) + ' ' + str(z))

        while self.pitch != 0:
            self.agent_host.sendCommand('look ' + str(- self.pitch))
            if self.pitch > 0:
                self.pitch -= 1
            else:
                self.pitch += 1

        turn_val = random.randint(1, 2) * 2 - 3
        self.turned = (self.turned + turn_val + {True: 4, False: 0}[turn_val < 0]) % 4
        self.agent_host.sendCommand('turn ' + str(turn_val))
        turn_val = random.randint(1, 2) * 2 - 3
        self.turned = (self.turned + turn_val + {True: 4, False: 0}[turn_val < 0]) % 4
        self.agent_host.sendCommand('turn ' + str(turn_val))

        if self.doorBroken:
            self.agent_host.sendCommand('chat /setblock 0 46 10 dark_oak_door 1')
            self.agent_host.sendCommand('chat /setblock 0 47 10 dark_oak_door 8')
            self.doorBroken = False  # TODO: OPEN/CLOSE door

    def finishedMission(self, grid):
        if self.task in ['room1']:
            if grid[16] == u'lapis_block' and self.doorBroken:
                return True
        elif self.task in ['room2', 'full']:    # need to make sure 'full' is only at final task we have
            if grid[13] in [u'stained_hardened_clay', u'carpet']:
                return True
        return False

    def act(self, world_state, current_r):
        """take 1 action in response to the current world state"""
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text)  # most recent observation
        self.logger.debug(obs)

        action = self.receiveCommand()

        if action == 'turn 1':
            self.turned = (self.turned + 1) % 4
        elif action == 'turn -1':
            self.turned = (self.turned - 1 + 4) % 4

        self.logger.info("Taking q action: %s" % action)
        # try to send the selected action, only update prev_s if this succeeds
        try:
            if 'newgame' in action or self.isTerminal:
                self.newGame()
            else:
                if u'LineOfSight' in obs:
                    try:
                        msg = world_state.observations[-1].text
                        observations = json.loads(msg)
                        grid = observations.get(u'floor3x3', 0)

                        # if obs[u'LineOfSight'][u'type'] not in [u'sandstone'] and action in ['attack 1'] or \
                        if action in ['nop']:
                            action = 'jump 0'
                        elif obs[u'LineOfSight'][u'type'] in [u'dark_oak_door'] and action == 'attack 1' \
                                and grid[16] == u'lapis_block' and not self.doorBroken:  # was 'use 1'
                            action = 'attack 1'
                            self.doorBroken = True
                            self.room += 1
                        elif action in ['attack 1', 'use 1']:
                            action = 'jump 0'
                        # elif (grid[13] == u'stone_stairs' and grid[1] == u'stone_stairs' or grid[25] == u'stone_stairs' and obs[u'LineOfSight'][u'type'] == u'stone_stairs') and action in ['move 1']:
                        elif grid[25] == u'stone_stairs' and self.turned == 0 and action in ['move 1']:
                            action = 'jumpmove 1'
                    except Exception as e:  # if no grid or no observation - this is rare.. tmp fix is send null action
                        action = 'jump 0'
                elif action in ['attack 1', 'use 1', 'nop']:
                    action = 'jump 0'

                if 'swap' == action:    # TODO: swap to secondary item etc...
                    # self.agent_host.sendCommand(action + "1")
                    # self.agent_host.sendCommand(action + "0")
                    self.agent_host.sendCommand('jump 0')
                else:
                    if 'look' == action:
                        if self.pitch >= 1:
                            action = 'look -1'
                            self.pitch -= 1
                        else:
                            action = 'look 1'
                            self.pitch += 1
                    self.logger.info("Taking q action: %s" % action)
                    self.agent_host.sendCommand(action)
                    if action == 'attack 1':
                        time.sleep(5 * self.sleepTime)

        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)
        self.numSteps += 1
        return current_r

    def run(self):
        """run the agent on the world"""
        total_reward = 0
        self.doorBroken = False
        self.prev_s = None
        self.prev_a = None

        is_first_action = True
        self.newGame()
        # main loop:
        world_state = self.agent_host.getWorldState()
        current_r = 0
        while world_state.is_mission_running:

            current_r = 0
            print('Num steps: ' + str(self.numSteps) + ', Max steps: ' + str(self.maxSteps))
            if self.numSteps >= self.maxSteps:
                self.isTerminal = True

            if is_first_action:
                # wait until have received a valid observation
                while True:
                    time.sleep(self.sleepTime)
                    world_state = self.agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations) > 0 \
                            and not (world_state.observations[-1].text == "{}") and len(world_state.video_frames) > 0:
                        total_reward += self.act(world_state, current_r)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_r == 0:
                    time.sleep(self.sleepTime)
                    world_state = self.agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                # allow time to stabilise after action
                while True:
                    time.sleep(self.sleepTime)
                    world_state = self.agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations) > 0 \
                            and not (world_state.observations[-1].text == "{}") and len(world_state.video_frames) > 0:
                        msg = world_state.observations[-1].text
                        observations = json.loads(msg)
                        grid = observations.get(u'floor3x3', 0)
                        self.curStepReward = self.perStepReward

                        if self.finishedMission(grid):
                            self.isTerminal = True
                            self.curStepReward = 0
                        if self.isTerminal:
                            self.numSteps = 0
                        self.sendState(world_state.video_frames[-1].pixels, self.curStepReward, self.isTerminal)
                        self.isTerminal = False
                        total_reward += self.act(world_state, current_r)
                        break
                    if not world_state.is_mission_running:
                        break

        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTableFromTerminatingState(current_r)

        return current_r

    def sendState(self, state, reward, terminal):
        print('sending')
        if self.sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind(('localhost', self.port))
        if not self.connected:
            self.sock.listen(1)
            self.connection, addr = self.sock.accept()
            print('Connected by ' + str(addr))
            self.connected = True
        sendstr = ''
        for pixel in state:
            sendstr += chr(pixel)

        self.connection.sendall(sendstr)

        self.connection.send(str(reward))
        self.connection.send('a')
        if terminal is True:
            print('TERMINAL!')
            terminal = 1
        else:
            terminal = 0
        self.connection.send(str(terminal))
        self.connection.send('a')

        self.connection.send(str(self.room))
        self.connection.send('a')

    def receiveCommand(self):
        print('receiving')
        if self.sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind(('localhost', self.port))
        if not self.connected:
            self.sock.listen(1)
            self.connection, addr = self.sock.accept()
            print('Connected by ' + str(addr))
            self.connected = True
        action = int(self.connection.recv(1))
        print(action)
        return self.int_to_action_table[action]


def main():
    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

    domains = ['full', 'room1', 'room2']
    task = domains[0]
    iterator = 1
    dqn_port = 14001
    print('Command line is: <Malmo Port - Not Required> <Dqn Port - Required> <Task - Not Required>')
    if sys.argv[len(sys.argv) - iterator] in domains:
        task = sys.argv[len(sys.argv) - 1]
        iterator += 1
    print('Running task: ' + task)
    try:
        dqn_port = int(sys.argv[len(sys.argv) - iterator])
        iterator += 1
    except:
        print('Dqn port is required!')
        exit(0)
    try:
        malmo_port = int(sys.argv[len(sys.argv) - iterator])
        iterator += 1
    except Exception as e:
        print('Default Malmo port used - 10000')
        malmo_port = 10000

    agent_host = MalmoPython.AgentHost()
    agent = TabQAgent(agent_host, dqn_port, task)

    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print('ERROR: ' + str(e))
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    max_retries = 3

    if agent_host.receivedArgument("test"):
        num_repeats = 1
    else:
        num_repeats = 150

    cumulative_rewards = []

    my_client_pool = MalmoPython.ClientPool()
    # Add the default client - port 10000 on the local machine:
    my_client = MalmoPython.ClientInfo("127.0.0.1", int(malmo_port))
    my_client_pool.add(my_client)
    experimentID = uuid.uuid4()

    while True:
        # add 20% holes for interest
        '''for x in range(-3, 7):
            for z in range(0, 9):
                my_mission.drawBlock(x, 45, z, "grass")
                if random.random() < 0.1 and not (x == 7 and z == 1):
                    my_mission.drawBlock(x, 45, z, "lava")'''

        #print 'Repeat %d of %d' % (i + 1, num_repeats)
        agent.turned = 0
        agent.pitch = 0

        my_mission_record = MalmoPython.MissionRecordSpec()

        # -- set up the mission -- #
        mission_file = './single_room.xml'
        with open(mission_file, 'r') as f:
            print('Loading mission from ' + mission_file)
            mission_xml = f.read()
            mission_xml = mission_xml.replace('X_START', str(15.5 - random.randint(0, 7)))
            mission_xml = mission_xml.replace('Z_START', str(-3.5 + random.randint(0, 7)))
            mission_xml = mission_xml.replace('YAW_START', str(0))  # must be 0 to start with 'turned = 0'
            my_mission = MalmoPython.MissionSpec(mission_xml, True)
            my_mission.forceWorldReset()
        for retry in range(max_retries):
            try:
                agent_host.startMission(my_mission, my_client_pool, my_mission_record, 0, str(experimentID))
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print('Error starting mission: ' + str(e))
                    exit(1)
                else:
                    time.sleep(2.5)

        print('Waiting for the mission to start')
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            sys.stdout.write(".")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print('Error: ' + error.text)
        print('Mission running')
        agent.newGame()
        agent_host.sendCommand('chat /difficulty peaceful')
        # -- run the agent in the world -- #
        #time.sleep(5)
        #exit(0)
        try:
            last_r = agent.run()
        except Exception as e:
            print(e.message)
        # print 'Cumulative reward: %d' % cumulative_reward
        # cumulative_rewards += [ cumulative_reward ]

        # Check if standing on finishing block (need to expand this to a function)
        '''reward = perStepReward
        if len(world_state.observations) > 0:
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            grid = observations.get(u'floor3x3', 0)
            if grid[4] == u'lapis_block':
                reward = 0
        sendState(world_state.video_frames[-1].pixels, reward, True)'''
        # -- clean up -- #

        while world_state.is_mission_running:
            if world_state.number_of_rewards_since_last_state > 0:
                for r in world_state.rewards:
                    print('Got reward: ' + str(r.getValue()))
                    # total_reward += r.getValue()
            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                ob = json.loads(msg)
                # print ob[u'LineOfSight']["type"]
                break
            world_state = agent_host.getWorldState()

        '''agent_host.sendCommand("hotbar.1 1") #Press the hotbar key
        agent_host.sendCommand("hotbar.1 0") #Release hotbar key - agent should now be holding diamond_pickaxe
        '''
        agent.connected = False
        time.sleep(20)  # (let the Mod reset)

    print('Done.')
    print('Cumulative rewards for all ' + str(num_repeats) + ' runs:')
    print(cumulative_rewards)

if __name__ == "__main__":
    main()
