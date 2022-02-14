import bitbooster.strings as bbs


def bitbooster_from_dataframe(df, n, metric, weighted):
    from bitbooster.preprocessing import binarize, discretize
    data = binarize(discretize(df, n_bits=n), n)
    if metric == bbs.EUCLIDEAN:
        if weighted:
            from bitbooster.euclidean.weighted import WeightedEuclideanBitBooster as Cl
        else:
            from bitbooster.euclidean.bitbooster import EuclideanBinaryObject as Cl
        return Cl(data=data, num_bits=n)
    else:
        raise NotImplementedError(f'Not implemented for metric={metric}, weighted={weighted}')


def vanilla_clusterable_from_dataframe(original_data, metric, weighted, normalize_data=None):
    if normalize_data is None:
        if metric in [bbs.EUCLIDEAN, bbs.MANHATTAN]:
            normalize_data = True
        elif metric == bbs.GOWER:
            normalize_data = False
        else:
            raise NotImplementedError(f'normalize_data cannot be determined for metric {metric}')

    if normalize_data:
        from bitbooster.preprocessing.normalizer import normalize
        new_data = normalize(original_data)
    else:
        new_data = original_data

    if metric == bbs.EUCLIDEAN:
        if weighted:
            from bitbooster.euclidean.weighted import VanillaWeightedEuclidean as Cl
        else:
            from bitbooster.euclidean.vanilla import EuclideanVanillaObject as Cl
    elif metric == bbs.GOWER:
        if weighted:
            raise NotImplementedError('Weighted Vanilla Gower is not implemented')
        else:
            from bitbooster.gower.vanilla import GowerVanillaObject as Cl
    else:
        raise NotImplementedError(f'Not implemented for metric={metric}, weighted={weighted}')

    return Cl(data=new_data)


def discrete_clusterable_from_dataframe(original_data, n, metric, weighted):
    if n <= 3:
        return bitbooster_from_dataframe(original_data, n, metric, weighted)
    else:
        from bitbooster.preprocessing import binarize, discretize
        discretized_data = discretize(original_data, n)
        return vanilla_clusterable_from_dataframe(discretized_data, metric, weighted, False)
