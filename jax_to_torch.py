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
def jvp_torch(grad_fn, x, eps):
    x = x.requires_grad_(True)
    f_x = grad_fn(x)
    jvp = torch.autograd.grad(outputs=f_x, inputs=x, grad_outputs=eps, create_graph=True, only_inputs=True, retain_graph=True)[0]
    return f_x, jvp

def get_div_fn(fn, shape, exact=False):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""
    _, dim = shape
    args = {
        "ode": True
    }
    if exact:
        def div_fn_1(f):
            def out(x_, t_):
                x_.requires_grad_(True)
                f_xt = f(x_[None], t_, args)
                f_xt = f_xt.squeeze()
                jac = torch.autograd.functional.jacobian(f, (x_, t_, args))
                full_jac = jac[:dim, :dim]
                func_jac = full_jac[:dim, :dim]
                return torch.trace(func_jac)
            return out

        def div_fn(x, t):
            div_1 = lambda data: div_fn_1(fn)(data, t)
            div_fn_ = torch.tensor([div_1(data) for data in x])
            return div_fn_

    else:
        eps = torch.normal(mean=0.0, std=1.0, size=shape)
        def div_fn(x, t):
            def grad_fn(data):
                with torch.enable_grad():
                    data.requires_grad_(True)
                    f_data = fn(data, t, args)[:, :dim]
                grads = torch.autograd.grad(outputs=f_data, inputs=data, grad_outputs=torch.ones_like(f_data), create_graph=True, only_inputs=True)[0]
                return grads

            #grad_fn_eps = torch.jvp(grad_fn, (x[:, :dim],), (eps,))[1]
            x_dim = x[:, :dim]
            f_x, grad_fn_eps = jvp_torch(grad_fn, x_dim, eps)

            return torch.sum(grad_fn_eps[:, :dim] * eps, axis=-1)

    return div_fn