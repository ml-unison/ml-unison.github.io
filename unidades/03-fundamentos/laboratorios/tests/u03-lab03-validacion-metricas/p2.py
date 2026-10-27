OK_FORMAT = True

test = {   'name': 'p2',
    'points': 1,
    'suites': [   {   'cases': [   {   'code': '>>> _P = pliegues(10, 3, semilla=0)\n'
                                               ">>> assert len(_P) == 3, 'debe haber k pliegues'\n"
                                               ">>> assert sorted(np.concatenate(_P).tolist()) == list(range(10)), 'los pliegues deben contener cada índice exactamente una vez'\n"
                                               ">>> assert [len(p) for p in _P] == [4, 3, 3], 'con n = 10 y k = 3 los tamaños deben ser 4, 3, 3'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> assert all((np.array_equal(a, b) for a, b in zip(pliegues(50, 5, 1), pliegues(50, 5, 1)))), 'con la misma semilla deben salir los mismos "
                                               "pliegues'\n"
                                               ">>> assert not np.array_equal(pliegues(50, 5, 1)[0], pliegues(50, 5, 2)[0]), 'la semilla debe cambiar el barajado'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
