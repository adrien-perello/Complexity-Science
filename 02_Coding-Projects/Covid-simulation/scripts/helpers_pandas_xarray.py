import pandas as pd


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
