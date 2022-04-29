import numpy as np

from edugrad.ops import Operation, tensor_op


@tensor_op
class sigmoid(Operation):
    @staticmethod
    def forward(ctx, a):
        value = 1 / (1 + np.exp(-a))
        ctx.append(value)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        value = ctx[-1]
        return [value * (1 - value) * grad_output]


@tensor_op
class log(Operation):
    @staticmethod
    def forward(ctx, a):
        ctx.append(a)
        return np.log(a)

    @staticmethod
    def backward(ctx, grad_output):
        in_value = ctx[-1]
        return [1 / in_value * grad_output]


@tensor_op
class multiply(Operation):
    """Element-wise multiplication. """

    @staticmethod
    def forward(ctx, a, b):
        ctx.append(a)
        ctx.append(b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx[-2:]
        return b * grad_output, a * grad_output


@tensor_op
class sum_along_columns(Operation):
    @staticmethod
    def forward(ctx, a):
        ctx.append(a)
        return np.sum(a, axis=1)

    @staticmethod
    def backward(ctx, grad_output):
        return [np.ones(ctx[-1].shape) * grad_output[:, np.newaxis]]


@tensor_op
class lookup_rows(Operation):
    """Given a matrix of size [m, n] and an array of integer indices
    [i0, i1, ..., in], return the relevant rows of the matrix.
    """

    @staticmethod
    def forward(ctx, matrix, indices):
        ctx.append(matrix.shape)
        ctx.append(indices)
        return matrix[indices]

    @staticmethod
    def backward(ctx, grad_output):
        shape, indices = ctx
        grads = np.zeros(shape)
        # this is some numpy magic: `indices` may have repeats of a given token index,
        # but if we just do grads[indices] += grad_output, it won't add up the rows
        # from grad_output for each occurance of the same index; this method accumulates
        # all of those sums, which is what's needed for the gradients
        np.add.at(grads, indices, grad_output)
        return [grads, np.zeros(indices.shape)]