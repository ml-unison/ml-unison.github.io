OK_FORMAT = True

test = {   'name': 'p2',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> _c = torch.tensor([3.0, -1.0], dtype=torch.float64)\n'
                                               '>>> _t0 = torch.zeros(2, dtype=torch.float64)\n'
                                               '>>> _t, _h = descenso_gradiente(lambda t: 0.5 * ((t - _c) ** 2).sum(), _t0, tasa=0.1, pasos=50)\n'
                                               ">>> assert len(_h) == 50 and isinstance(_h[0], float), 'historia es una lista de float, uno por paso'\n"
                                               ">>> assert abs(_h[0] - 5.0) < 1e-12, 'historia[0] es la pérdida en theta0: ½(3² + 1²) = 5'\n"
                                               ">>> assert torch.allclose(_t, _c * (1 - 0.9 ** 50)), 'θ no coincide con el descenso por gradiente: ¿limpias .grad antes de cada backward()?'\n"
                                               ">>> assert not _t.requires_grad, 'devuelve θ sin historia (usa detach)'\n"
                                               ">>> assert torch.equal(_t0, torch.zeros(2, dtype=torch.float64)), 'theta0 no debe modificarse'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
