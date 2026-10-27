OK_FORMAT = True

test = {   'name': 'p5',
    'points': 4,
    'suites': [   {   'cases': [   {   'code': ">>> _cli = TestClient(crear_app(modelo, '2011.1'))\n"
                                               ">>> _ej = {'hora': 8, 'laboral': 1, 'clima': 1, 'temp': 22.0, 'humedad': 55.0, 'viento': 10.0}\n"
                                               ">>> _r = _cli.post('/predecir', json=_ej)\n"
                                               ">>> assert _r.status_code == 200, 'una entrada válida debe responder 200'\n"
                                               ">>> assert set(_r.json()) == {'rentas', 'version'} and _r.json()['version'] == '2011.1', 'la respuesta trae rentas y version'\n"
                                               ">>> assert abs(_r.json()['rentas'] - modelo.predict(pd.DataFrame([_ej])[VARIABLES])[0]) < 1e-06, 'la predicción debe ser la del modelo'\n"
                                               ">>> assert _cli.get('/salud').json() == {'estado': 'ok', 'version': '2011.1'}\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> _cli = TestClient(crear_app(modelo, '2011.1'))\n"
                                               ">>> _ej = {'hora': 8, 'laboral': 1, 'clima': 1, 'temp': 22.0, 'humedad': 55.0, 'viento': 10.0}\n"
                                               ">>> assert _cli.post('/predecir', json={**_ej, 'hora': 24}).status_code == 422, 'hora = 24 está fuera de rango: 422'\n"
                                               ">>> assert _cli.post('/predecir', json={**_ej, 'humedad': -1}).status_code == 422, 'humedad negativa: 422'\n"
                                               ">>> assert _cli.post('/predecir', json={k: v for k, v in _ej.items() if k != 'temp'}).status_code == 422, 'falta temp: 422'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
