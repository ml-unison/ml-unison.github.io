OK_FORMAT = True

test = {   'name': 'p5',
    'points': 3,
    'suites': [   {   'cases': [   {   'code': '>>> _ = torch.manual_seed(0)\n'
                                               '>>> _m = MLP([64, 32, 10])\n'
                                               '>>> _lin = [c for c in _m.modules() if isinstance(c, nn.Linear)]\n'
                                               ">>> assert len(_lin) == 2, 'MLP([64, 32, 10]) tiene dos capas lineales'\n"
                                               ">>> assert tuple(_m(torch.zeros(5, 64)).shape) == (5, 10), 'la salida debe ser de M × 10'\n"
                                               ">>> assert sum((p.numel() for p in _m.parameters())) == 64 * 32 + 32 + 32 * 10 + 10, 'los parámetros no son los de dos capas lineales'\n"
                                               ">>> assert all((torch.all(c.bias == 0) for c in _lin)), 'los sesgos empiezan en cero'\n"
                                               ">>> _m = MLP([2000, 1000, 10], 'relu')\n"
                                               '>>> _s = [c for c in _m.modules() if isinstance(c, nn.Linear)][0].weight.std().item()\n'
                                               ">>> assert abs(_s / np.sqrt(2 / 2000) - 1) < 0.05, f'con ReLU, la desviación estándar de W[1] debe ser √(2/n[0]) = 0.0316 (obtuviste {_s:.4f})'\n"
                                               ">>> _m = MLP([2000, 1000, 10], 'tanh')\n"
                                               '>>> _s = [c for c in _m.modules() if isinstance(c, nn.Linear)][0].weight.std().item()\n'
                                               ">>> assert abs(_s / np.sqrt(2 / 3000) - 1) < 0.05, f'con tanh, la desviación estándar de W[1] debe ser √(2/(n[0] + n[1])) = 0.0258 (obtuviste "
                                               "{_s:.4f})'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
