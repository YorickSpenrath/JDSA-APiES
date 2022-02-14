import itertools

from ...data_handlers import ConsumerDataHandler, AbstractGCSSettings
from ...execution_modules.single import run_experiment
from ...experiments.consumers.combine import combine


def _run(param, reverse=False):
    assert param in ['k', 'rho']

    combine(param)

    tau_values = list(range(3, 11, 2)) + list(range(2, 11, 2))
    if reverse:
        tau_values = tau_values[::-1]

    if param == 'k':
        cp_values = [1, 100, 250, 500, 750] + \
                    list(range(1000, 10001, 2000)) + \
                    [None] + \
                    list(range(1000, 10001, 1000)) + \
                    list(range(1000, 10001, 500))
        it = itertools.product(tau_values, cp_values)
        cpt = AbstractGCSSettings.CLUSTER_COUNT
    elif param == 'rho':
        cp_values1 = [2 ** i for i in [0] + list(range(1, 18, 2))] + [None]
        cp_values2 = [2 ** i for i in list(range(2, 18, 2))]
        cpt = AbstractGCSSettings.CLUSTER_SIZE
        it1 = itertools.product(tau_values, cp_values1)
        it2 = itertools.product(tau_values, cp_values2)
        it = itertools.chain(it1, it2)
    else:
        raise NotImplementedError

    for tau, cp in it:
        x = ConsumerDataHandler(name=f'RNN_native_tau={tau}_{param}={cp}',
                                clustering_method=AbstractGCSSettings.NATIVE,
                                clustering_parameter_type=cpt,
                                clustering_parameter=cp,
                                model_type='TAX',
                                tau=tau)
        if x.is_done():
            continue

        run_experiment(x)
        combine(param)


def run_k_experiments(reverse=False):
    _run('k', reverse)


def run_rho_experiments(reverse=False):
    _run('rho', reverse)
