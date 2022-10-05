from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from .model import NetworkedSIR, InfectionState


def network_portrayal(graph):
    # The model ensures there is always 1 agent per node

    def node_color(agent):
        return {
            InfectionState.Infected: "#FF0000",
            InfectionState.Susceptible: "#008000",
        }.get(agent.state, "#808080")

    def get_agents(source, target):
        return graph.nodes[source]["agent"][0], graph.nodes[target]["agent"][0]

    portrayal = dict()
    portrayal["nodes"] = [
        {
            "size": 6,
            "color": node_color(agents[0]),
            "tooltip": "id: {}<br>state: {}".format(
                agents[0].unique_id, agents[0].state.name
            ),
        }
        for (_, agents) in graph.nodes.data("agent")
    ]

    portrayal["edges"] = [
        {
            "source": source,
            "target": target,
            "color": "#000000",
            # "width": edge_width(*get_agents(source, target)),
        }
        for (source, target) in graph.edges
    ]

    return portrayal


grid = NetworkModule(network_portrayal, 500, 500, library="d3")  # library="sigma", 'd3'
chart = ChartModule(
    [
        {"Label": "Infected", "Color": "#FF0000"},
        {"Label": "Susceptible", "Color": "#008000"},
        {"Label": "Recovered", "Color": "#808080"},
    ],
    data_collector_name="datacollector",
)

model_params = {
    "num_agents": UserSettableParameter(
        "slider",
        "Number of agents",
        200,
        50,
        500,
        10,
        description="Choose how many agents to include in the model",
    ),
    "infectious_period": UserSettableParameter(
        "slider",
        "Infectious period",
        15,
        1,
        60,
        1,
        description="Choose how long the infectious period lasts",
    ),
    "immunization_period": UserSettableParameter(
        "slider",
        "Immunization period",
        30,
        1,
        60,
        1,
        description="Choose how long the immunization period lasts",
    ),
    "transmission_probability": UserSettableParameter(
        "slider",
        "Transmission probability",
        0.2,
        0,
        1,
        0.1,
        description="Choose the probability of the virus to be transmitted when contact happens",
    ),
    "daily_contact": UserSettableParameter(
        "slider",
        "Daily contacts",
        1,
        1,
        10,
        1,
        description="Choose how many daily contacts happen between agents",
    ),
    "ini_infection_distri": UserSettableParameter(
        "choice",
        "Initial probability distribution",
        value=(0.8, 0.05, 0.15),
        choices=[(0.8, 0.05, 0.15)],
        description="Choose the initial probability distribution for each state (susceptible, infected, recovered)",
    ),
    "network_type": UserSettableParameter(
        "choice",
        "Network type",
        value="scale free",
        choices=["random", "small world", "scale free"],
        description="Choose the type of random network",
    ),
    "expected_avg_degree": UserSettableParameter(
        "slider",
        "Average node degree",
        5,
        1,
        30,
        1,
        description="Choose the (expected) average node degree",
    ),
}

server = ModularServer(NetworkedSIR, [grid, chart], "SIR Model", model_params)
server.port = 8522
