"""Model class for a networked SIR Agent-Based Model."""

import numpy as np

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

from scripts.helpers_networkx import generate_random_graph, network_metrics
from scripts.agent import PandemicAgent, InfectionState


def count_state(model, state):
    """Count number of agent in a specific infectionState.

    Args:
        model (Mesa.model): mesa model for which to count agents
        state (infectionState): infectionState (S, I, R)

    Returns:
        int: number of agent in a specific infectionState
    """
    return sum([1 for a in model.grid.get_all_cell_contents() if a.state is state])


def count_susceptible(model):
    """Count number of susceptible agents.

    Args:
        model (Mesa.model): mesa model for which to count agents

    Returns:
        int: number of susceptible agents.
    """
    return count_state(model, InfectionState.SUSCEPTIBLE)


def count_infected(model):
    """Count number of infected agents.

    Args:
        model (Mesa.model): mesa model for which to count agents

    Returns:
        int: number of infected agents.
    """
    return count_state(model, InfectionState.INFECTED)


def count_immunized(model):
    """Count number of immunized agents.

    Args:
        model (Mesa.model): mesa model for which to count agents

    Returns:
        int: number of immunized agents.
    """
    return count_state(model, InfectionState.RECOVERED)


class NetworkedSIR(Model):
    """Model class for a networked SIR Agent-Based Model."""

    def __init__(
        self,
        num_agents,
        infectious_period,
        immunization_period,
        transmission_probability,
        daily_contact,
        ini_infection_distri,
        network_type,
        expected_avg_degree,
    ):
        """Instantiate the SIR model with agent in a random network.

        Args:
            num_agents (int): number of agents in the model
            infectious_period (int/float): amount of time an agent remains infectious
            immunization_period (int/float): amount of time an agent remains immunized
            transmission_probability (float): probability to infect in case of contact
            daily_contact (int): maximum number of contacts an agent has at each step
            ini_infection_distri (list[floats]): probability for initial InfectionState
                                                of each agent. It should have the same
                                                length/order as InfectionState
            network_type (string): type of randm graph ('random', 'small world', 'scale free')
            expected_avg_degree (int/float): expected average node degree of the network
        """

        self.num_agents = num_agents
        self.infectious_period = infectious_period
        self.immunization_period = immunization_period
        self.transmission_probability = transmission_probability
        self.daily_contact = daily_contact

        self.graph = generate_random_graph(
            network_type, num_agents, expected_avg_degree
        )
        self.grid = NetworkGrid(self.graph)
        self.schedule = RandomActivation(self)
        self.running = True

        # Create agents
        for i, node in enumerate(self.graph.nodes()):
            state = np.random.choice(
                [s.value for s in InfectionState], p=ini_infection_distri
            )
            agent = PandemicAgent(i, self, InfectionState(state))
            self.schedule.add(agent)
            self.grid.place_agent(agent, node)  # Add the agent to a random node

        self.datacollector = DataCollector(
            {
                "Infected": count_infected,
                "Susceptible": count_susceptible,
                "Recovered": count_immunized,
                "network_metrics": lambda m: network_metrics(m.graph),
                "nb components": np.round(nb_components, 2),
                "avg degree": np.round(avg_degree, 2),
                "avg distance": np.round(avg_dist, 2),
                "degree distribution": np.array(nx.degree_histogram(nx_graph)),
            }
        )

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()

    def run_model_n_steps(self, nb_steps):
        """Advance the model by n steps."""
        for _ in range(nb_steps):
            self.step()


if __name__ == "__main__":
    pass
