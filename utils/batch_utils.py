import numpy
import progressbar


__all__ = ['iter_batches', 'batch_wrapper']


def iter_batches(dataset_size, batch_size, edge_complete=False, shuffle=True):
    indices = numpy.arange(dataset_size)

    if shuffle:
        numpy.random.shuffle(indices)

    j = 0
    for j in xrange(0, dataset_size - batch_size, batch_size):
        yield indices[j: j + batch_size]
    if (dataset_size - (j + batch_size)) == batch_size:
        yield indices[j + batch_size:]
    else:
        if edge_complete:
            batch = indices[j + batch_size:]

            yield numpy.concatenate([
                batch,
                indices[numpy.arange(batch_size - len(batch)) % dataset_size]
            ])


def batch_wrapper(function, batch_size, aggregate=None, cut=False, shuffle=True):
    def _function(*args, **kargs):
        args = map(numpy.asarray, args)
        dataset_size = args[0].shape[0]
        indices = iter_batches(dataset_size, batch_size,
                               edge_complete=True, shuffle=shuffle)
        output_values = []

        _verbose = kargs.pop('_verbose', False)

        if _verbose:
            bar = progressbar.ProgressBar(0, dataset_size)
            for batch_indices in indices:
                batch_output = function(
                    *[arg[batch_indices] for arg in args],
                    **kargs
                )
                output_values.append(batch_output)
                bar.update(min(dataset_size, bar.value + batch_size))
        else:
            for batch_indices in indices:
                batch_output = function(
                    *[arg[batch_indices] for arg in args],
                    **kargs
                )
                output_values.append(batch_output)

        if len(function.outputs) == 1 and function.unpack_single:
            output_values = [[x] for x in output_values]

        output_values_by_index = zip(*output_values)

        output_values_joined = []

        for output_value in output_values_by_index:
            if len(output_value[0].shape) == 0:
                output = numpy.stack(output_value)
            else:
                output = numpy.concatenate(output_value)

            if cut:
                output = output[: dataset_size]

            if aggregate is not None:
                output = aggregate(output)

            output_values_joined.append(output)

        if len(function.outputs) == 1 and function.unpack_single:
            return output_values_joined[0]
        else:
            return output_values_joined

    return _function
