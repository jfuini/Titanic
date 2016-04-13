# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 12:48:46 2016

@author: John Fuini
"""

from sknn.mlp import Classifier, Layer

nn = Classifier(
    layers=[
        Layer("Rectifier", units=100),
        Layer("Linear")],
    learning_rate=0.02,
    n_iter=10)
nn.fit(X_train, y_train)

y_valid = nn.predict(X_valid)

score = nn.score(X_test, y_test)