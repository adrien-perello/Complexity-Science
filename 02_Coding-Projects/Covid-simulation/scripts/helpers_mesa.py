import pandas as pd

from inspect import signature
from mesa.batchrunner import BatchRunner, BatchRunnerMP
from mesa.model import Model


def unnest_records(df, column):
    """[summary]

    Args:
        df ([type]): [description]
        column ([type]): [description]

    Returns:
        [type]: [description]
    """
    df_expanded = pd.DataFrame.from_records(df[column])
    df_final = pd.concat([df, df_expanded], axis=1)
    return df_final.drop(column, axis=1)


def unnest_columns(df, columns):
    """[summary]

    Args:
        df ([type]): [description]
        columns ([type]): [description]

    Returns:
        [type]: [description]
    """
    return (
        pd.concat(
            [pd.DataFrame(df[x].tolist(), index=df.index) for x in columns],
            axis=1,
            keys=columns,
        )
        .stack()
        .rename_axis(index={None: "Step"})
    )


def get_model_vars_dataframe_custom(batch_run):  # (self, vars_dict, extra_cols=None)
    """[summary]

    Args:
        batch_run ([type]): [description]

    Returns:
        [type]: [description]
    """
    extra_cols = ["Run"]  # + (extra_cols or [])
    index_cols = []
    if batch_run.parameters_list:
        index_cols = list(batch_run.parameters_list[0].keys())
    if batch_run.fixed_parameters:
        fixed_param_cols = list(batch_run.fixed_parameters.keys())
        index_cols += fixed_param_cols
    index_cols += extra_cols
    records = []
    for param_key, values in batch_run.model_vars.items():  # self.vars_dict
        record = dict(zip(index_cols, param_key))
        record.update(values)
        records.append(record)
    df = pd.DataFrame(records)
    return df


def batchrunner_to_dataframe(batch_run):
    """[summary]

    Args:
        batch_run ([type]): [description]

    Returns:
        [type]: [description]
    """
    if isinstance(batch_run, BatchRunnerMP) or (
        isinstance(batch_run, BatchRunner) and not batch_run.parameters_list
    ):
        run_data = get_model_vars_dataframe_custom(batch_run)
    else:  # BatchRunner instance with both fixed and variable parameters
        run_data = batch_run.get_model_vars_dataframe()
        run_data["Run"] = run_data["Run"] % batch_run.iterations  # convert 'Run' number
        #           from unique id to iteration number (for use as coordinate in xarray)

    fixed_params_idx = list(batch_run.fixed_parameters.keys())
    run_data.drop(fixed_params_idx, axis=1, inplace=True)  # drop fixed parameters
    indexes = list(run_data.drop("datacollectors", axis=1).columns)  # save columns
    #                         they will become indexes (i.e. all but datacollectors)
    run_data = unnest_records(run_data, "datacollectors")  # unnest data contained in
    #                                                 datacollectors into new columns
    variables = list(set(run_data.columns) - set(indexes))  # new columns = variable
    run_data.set_index(indexes, inplace=True, drop=True)  # reset index
    return unnest_columns(run_data, variables)  # unnest values inside variables


def get_model_fixed_attributes(mesa_object):
    """set fixed params as attributes rather than coords

    Args:
        mesa_object ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if type(mesa_object) in (BatchRunnerMP, BatchRunner):
        return mesa_object.fixed_parameters
    elif Model in type(mesa_object).__bases__:  # if base class if mesa.model.Model
        init_signature = signature(mesa_object.__init__)
        params = list(init_signature.parameters.keys())
        kwargs = dict()
        for key in params:
            try:
                kwargs[key] = mesa_object.__getattribute__(key)
            except AttributeError as attr_err:
                kwargs[key] = str(attr_err)
        return kwargs
    else:
        raise ValueError(
            "Invalid argument. Input should be of type"
            "- mesa.batchrunner.BatchRunner,"
            "- mesa.batchrunner.BatchRunnerMP,"
            "- or mesa.model.Model"
        )


def mesa_to_dataframe(mesa_object):
    """[summary]

    Args:
        mesa_object ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if type(mesa_object) in (BatchRunnerMP, BatchRunner):
        return batchrunner_to_dataframe(mesa_object)
    elif Model in type(mesa_object).__bases__:  # if base class if mesa.model.Model
        df = mesa_object.datacollector.get_model_vars_dataframe()
        df.index.names = ["Step"]
        return df
    else:
        raise ValueError(
            "Invalid argument. Input should be of type"
            "- mesa.batchrunner.BatchRunner,"
            "- mesa.batchrunner.BatchRunnerMP,"
            "- or mesa.model.Model"
        )


def mesa_to_xarray(mesa_object):
    """[summary]

    Args:
        mesa_object ([type]): [description]

    Returns:
        [type]: [description]
    """
    df = mesa_to_dataframe(mesa_object)  # generate cleaned dataframe
    da = df.to_xarray()  # convert df to xarray
    da.attrs = get_model_fixed_attributes(mesa_object)
    return da
