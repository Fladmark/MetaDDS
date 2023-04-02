import torch

def torch_scan(fn, init, xs, *args, **kwargs):
    """
    A PyTorch implementation of JAX's lax.scan function.

    Args:
        fn: A function to apply at each step. It should take the carry and an element from xs as input and return an updated carry and output.
        init: The initial carry value.
        xs: A sequence of inputs to iterate over.
        *args, **kwargs: Additional arguments and keyword arguments to pass to the function fn.

    Returns:
        A tuple containing the final carry and a tensor of outputs.
    """

    carry = init
    outputs = []

    for x in xs:
        carry, output = fn(carry, x, *args, **kwargs)
        outputs.append(output)

    outputs = torch.stack(outputs, dim=0)
    return carry, outputs