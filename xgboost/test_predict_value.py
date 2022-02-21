import numpy

import predict_value


def test_predict_value():
    result = predict_value.predict_value("885", "3.45", "null", "null", "12.75", "0", "4", "16.75")
    assert type(result) is numpy.ndarray
    assert len(result) == 1
