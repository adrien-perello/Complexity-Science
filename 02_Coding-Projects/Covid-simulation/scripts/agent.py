"""Agent class for a networked SIR Agent-Based Model."""

from enum import Enum
from random import choice, sample, random
from mesa import Agent


class InfectionState(Enum):
    """Agent status, following the SIR model.

    Args:
        Enum: Parent class
    """

    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2


class PandemicAgent(Agent):
    """An agent of the SIR Disease model (child classes of Mesa’s generic Agent classes).

    Args:
        Agent: (Mesa's) agent parent class
    """

    def __init__(self, unique_id, model, state):
        """Instantiate an ABM agent.

        Args:
            unique_id (int): unique id of each agent
            model (mesa model): mesa model the agent belongs to
            state (InfectionState): Agent status
        """
        super().__init__(unique_id, model)
        self.state = state
        self.infection_time = 0
        self.recovery_time = 0
        if self.state is InfectionState(1):  # time passed since infection
            self.infection_time = -choice(range(self.model.infectious_period))
        elif self.state is InfectionState(2):  # time passed since recovery
            self.recovery_time = -choice(range(self.model.immunization_period))

    def get_neighbors(self, include_self=False):
        """Get agent neighbors."""
        neighbors_nodes = self.model.grid.get_neighbors(
            self.pos, include_center=include_self
        )
        return self.model.grid.get_cell_list_contents(neighbors_nodes)

    def contact(self, daily_contact, transmission_probability):
        """How agents interact.

        Each agent is a node in a (static) graph. If the agent is infected, it exposes
        a fixed number of its neighbors.

        Args:
            daily_contact (int): maximum number of neigbhors to interact with
            transmission_probability (float): probability to infect in case of contact
        """
        if self.state is InfectionState(1):
            neighbors = self.get_neighbors()
            if len(neighbors) > 0:
                try:
                    targets = sample(
                        neighbors, k=daily_contact
                    )  # randomly choose k neighbors
                except ValueError:  # if not enough neighbors
                    targets = neighbors  # select max available
                for agent in targets:
                    agent.exposed(transmission_probability)

    def exposed(self, transmission_probability):
        """Handle the consequences of being exposed.

        Args:
            transmission_probability (float):  probability to be infected
        """
        if self.state is InfectionState(0) and random() < transmission_probability:
            self.state = InfectionState(1)
            self.infection_time = self.model.schedule.time

    def health_status(self, infectious_period, immunization_period):
        """Handle the evolution of the health status.

        Args:
            infectious_period (int/float): amount of time an agent remains infectious
            immunization_period (int/float): amount of time an agent remains immunized
        """
        if self.state is InfectionState(1):
            if (self.model.schedule.time - self.infection_time) >= infectious_period:
                self.state = InfectionState(2)
                self.infection_time = 0
                self.recovery_time = self.model.schedule.time
        elif self.state is InfectionState(2):
            if (self.model.schedule.time - self.recovery_time) >= immunization_period:
                self.state = InfectionState(0)
                self.infection_time = 0
                self.recovery_time = 0

    def step(self):
        self.contact(self.model.daily_contact, self.model.transmission_probability)
        self.health_status(self.model.infectious_period, self.model.immunization_period)


if __name__ == "__main__":
    pass
