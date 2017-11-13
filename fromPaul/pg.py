from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def dprint(msg):
    print('DIL ({}): {}'.format(datetime.now(), msg))


def dil(df_long):
    n_long = 20
    import pdb;
    pdb.set_trace()
    dprint('proba filter...')
    df_long = df_long[df_long.proba > 0.5] \
        .sort_values('proba') \
        .tail(n_long) \
        .copy()

    dprint('calculate weights...')
    df_long['weight'] = 1.0
    df_long['weight'] = df_long.weight/df_long.weight.sum()

    dprint('setting up LP...')
    import pdb;
    pdb.set_trace()

    df_long.loc[df_long.beta_historical_12m >= 3,
                'beta_historical_12m'] = 3
    df_long.loc[df_long.beta_historical_12m <= -0.5,
                'beta_historical_12m'] = -0.5

    # exposure
    delta_exp = 0.2
    A_exp = np.vstack([np.ones(len(df_long)),
                      -1.0 * np.ones(len(df_long))])
    b_exp = np.asarray([1.0 + delta_exp, -1.0 + delta_exp])

    # beta
    delta_beta = 0.2
    betas = df_long.beta_historical_12m.values * np.ones(len(df_long))
    A_beta = np.vstack([betas, -1.0 * betas])
    b_beta = np.asarray([1.0 + delta_beta, -1.0 + delta_beta])

    # volume constraints
    equity = 10E6
    volume_weeks = 4
    volume_max = 0.1
    A_volume = np.eye(len(df_long)) * equity
    b_volume = (df_long.volume_avg_historical_1m.values *
                df_long.price.values * volume_weeks * volume_max)

    import pdb; pdb.set_trace()

    # assemble primative matrix of constraints
    A_p = np.vstack([A_exp, A_beta, A_volume])
    b_p = np.hstack([b_exp, b_beta, b_volume])

    # "Z" constraints
    A_z = np.vstack([
        np.hstack([np.eye(len(df_long)), -1.0 * np.eye(len(df_long))]),
        np.hstack([-1.0 * np.eye(len(df_long)), -1.0 * np.eye(len(df_long))])
    ])
    b_z = np.hstack([df_long.weight.values, -1.0 * df_long.weight.values])

    # bounds
    bounds_w = np.zeros((len(df_long), 2))
    bounds_w[:, 1] = 0.05
    bounds_z = np.zeros((len(df_long), 2))
    bounds_z[:, 1] = None

    A_ub = np.zeros((A_p.shape[0] + A_z.shape[0],
                     A_z.shape[1]))
    A_ub[:A_p.shape[0], :A_p.shape[1]] = A_p
    A_ub[-A_z.shape[0]:, -A_z.shape[1]:] = A_z
    b_ub = np.hstack([b_p, b_z])

    import pdb;
    pdb.set_trace()
    # restated objective function
    c = np.hstack([np.zeros(len(df_long)), np.ones(len(df_long))])
    bounds = np.vstack([bounds_w, bounds_z])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    if not res.success:
        print(res)
        return

    df_long['weight_opt'] = res.x[:len(df_long)]
    cols = ['proba', 'weight', 'weight_opt', 'beta_historical_12m',
            'position', 'volume_avg_historical_1m']
    print(df_long.assign(position=equity * df_long.weight_opt / df_long.price)[cols])

    print('--- IDEAL ---')
    print('exposure: {}\nbeta: {}'.format(
        df_long.weight.sum(),
        (df_long.weight * df_long.beta_historical_12m).sum())
    )

    print('--- OPTIMAL ---')
    print('exposure: {}\nbeta: {}'.format(
        df_long.weight_opt.sum(),
        (df_long.weight_opt * df_long.beta_historical_12m).sum())
    )


def load():
    print('loading...')
    return (pd.read_pickle('long.pkl'),)


def main():
    dil(*load())


if __name__ == '__main__':
    main()
