OK_FORMAT = True

test = {   'name': 'p4',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> assert torch.allclose(gradiente_numerico(lambda v: (v ** 3).sum(), torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)), torch.tensor([3.0, 12.0, '
                                               "3.0], dtype=torch.float64), atol=1e-08), 'la derivada de Σ v³ es 3v²'\n"
                                               '>>> _A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)\n'
                                               '>>> _A0 = _A.clone()\n'
                                               '>>> _g = gradiente_numerico(lambda A: (A ** 2).sum() + A[0, 1] * A[1, 0], _A)\n'
                                               ">>> assert _g.shape == (2, 2), 'el gradiente tiene la forma de x'\n"
                                               ">>> assert torch.allclose(_g, torch.tensor([[2.0, 7.0], [8.0, 8.0]], dtype=torch.float64), atol=1e-08), 'debe funcionar con matrices'\n"
                                               ">>> assert torch.equal(_A, _A0), 'x debe quedar exactamente como estaba'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
