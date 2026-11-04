OK_FORMAT = True

test = {   'name': 'p3',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> _x = torch.tensor([-1000.0, -1.0, 0.0, 1.0, 1000.0], dtype=torch.float64, requires_grad=True)\n'
                                               '>>> _s = softplus(_x)\n'
                                               ">>> assert torch.isfinite(_s).all(), 'con x = ±1000 la softplus debe ser finita: usa max(x, 0) + log(1 + e^(−|x|))'\n"
                                               '>>> assert torch.allclose(_s.detach(), torch.tensor([0.0, np.log1p(np.exp(-1)), np.log(2), 1 + np.log1p(np.exp(-1)), 1000.0], dtype=torch.float64)), '
                                               "'softplus(−1000, −1, 0, 1, 1000) debe ser (0, 0.3133, 0.6931, 1.3133, 1000)'\n"
                                               '>>> _s.sum().backward()\n'
                                               ">>> assert torch.allclose(_x.grad, torch.sigmoid(_x.detach())), 'la derivada de la softplus es σ(x)'\n"
                                               '>>> _xg = torch.randn(7, dtype=torch.float64, generator=torch.Generator().manual_seed(3), requires_grad=True)\n'
                                               ">>> assert torch.autograd.gradcheck(softplus, (_xg,), raise_exception=False), 'gradcheck falla: revisa backward'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
