from malmo.agent import Agent as BaseAgent
import random


class Agent(BaseAgent):
    def __init__(self, params):
        super(Agent, self).__init__(params)
        self.experiment_id = 'simple_room'

    def _restart_world(self):
        mission_file = './malmo/domains/basic.xml'
        with open(mission_file, 'r') as f:
            print('Loading mission from ' + mission_file)
            mission_xml = f.read()
            self._load_mission_from_xml(mission_xml)
            self._wait_for_mission_to_begin()
