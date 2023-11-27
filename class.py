from typing import Iterable, Optional, Union

import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
from scipy import linalg
from cirq import protocols
from cirq.testing import gate_features
import random

from cirq import protocols, value
from cirq.ops import common_gates, raw_types

def R(fi, hi, i=0, j=1):
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    x01_for_ms = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]])
    y01_for_ms = np.array([[0, complex(0, -1), 0],
                           [complex(0, 1), 0, 0],
                           [0, 0, 0]])
    x12_for_ms = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0]])
    y12_for_ms = np.array([[0, 0, 0],
                           [0, 0, complex(0, -1)],
                           [0, complex(0, 1), 0]])
    x02_for_ms = np.array([[0, 0, 1],
                           [0, 0, 0],
                           [1, 0, 0]])
    y02_for_ms = np.array([[0, 0, complex(0, -1)],
                           [0, 0, 0],
                           [complex(0, 1), 0, 0]])
    if (i, j) == (0, 1):
        x_for_ms = x01_for_ms
        y_for_ms = y01_for_ms
    elif (i, j) == (1, 2):
        x_for_ms = x12_for_ms
        y_for_ms = y12_for_ms
    else:
        x_for_ms = x02_for_ms
        y_for_ms = y02_for_ms
    m = np.cos(fi) * x_for_ms + np.sin(fi) * y_for_ms

    return linalg.expm(complex(0, -1) * m * hi / 2)


@value.value_equality
class AsymmetricDepolarizingChannell(raw_types.Gate):
    """A channel that depolarizes asymmetrically along different directions."""

    def __init__(self, p_x: float, p_y: float, p_z: float) -> None:
        r"""The asymmetric depolarizing channel.

        This channel applies one of four disjoint possibilities: nothing (the
        identity channel) or one of the three pauli gates. The disjoint
        probabilities of the three gates are p_x, p_y, and p_z and the
        identity is done with probability 1 - p_x - p_y - p_z. The supplied
        probabilities must be valid probabilities and the sum p_x + p_y + p_z
        must be a valid probability or else this constructor will raise a
        ValueError.

        This channel evolves a density matrix via
            \rho -> (1 -p_x + p_y + p_z) \rho
                    + p_x X \rho X + p_y Y \rho Y + p_z Z \rho Z

        Args:
            p_x: The probability that a Pauli X and no other gate occurs.
            p_y: The probability that a Pauli Y and no other gate occurs.
            p_z: The probability that a Pauli Z and no other gate occurs.

        Raises:
            ValueError: if the args or the sum of args are not probabilities.
        """

        def validate_probability(p, p_str):
            if p < -100:
                raise ValueError('{} was less than 0.'.format(p_str))
            elif p > 1000:
                raise ValueError('{} was greater than 1.'.format(p_str))
            return p

        self._p_x = validate_probability(p_x, 'p_x')
        self._p_y = validate_probability(p_y, 'p_y')
        self._p_z = validate_probability(p_z, 'p_z')
        self._p_i = 1 - validate_probability(p_x + p_y + p_z, 'p_x + p_y + p_z')

    '''
    def _qid_shape_(self):
        return (3,)
    '''
    def _channel_(self) -> Iterable[np.ndarray]:
        return (
            np.sqrt(self._p_i) * np.eye(3),
            np.sqrt(self._p_x) * np.array([[0,1],[1,0]]),
            np.sqrt(self._p_y) * np.array([[0, -1j], [1j, 0]]),
            np.sqrt(self._p_z) * np.array([[1, 0], [0, -1]]),
        )

    def _value_equality_values_(self):
        return self._p_x, self._p_y, self._p_z

    def __repr__(self) -> str:
        return 'cirq.asymmetric_depolarize(p_x={!r},p_y={!r},p_z={!r})'.format(
            self._p_x, self._p_y, self._p_z
        )

    def __str__(self) -> str:
        return 'asymmetric_depolarize(p_x={!r},p_y={!r},p_z={!r})'.format(
            self._p_x, self._p_y, self._p_z
        )

    def _circuit_diagram_info_(
        self, args: protocols.CircuitDiagramInfoArgs
    ) -> str:
        return 'A({!r},{!r},{!r})'.format(self._p_x, self._p_y, self._p_z)



def asymmetric_depolarizel(
    p_x: float, p_y: float, p_z: float
) -> AsymmetricDepolarizingChannell:
    r"""Returns a AsymmetricDepolarizingChannel with given parameter.

    This channel evolves a density matrix via
        \rho -> (1 -p_x + p_y + p_z) \rho
                + p_x X \rho X + p_y Y \rho Y + p_z Z \rho Z

    Args:
        p_x: The probability that a Pauli X and no other gate occurs.
        p_y: The probability that a Pauli Y and no other gate occurs.
        p_z: The probability that a Pauli Z and no other gate occurs.

    Raises:
        ValueError: if the args or the sum of the args are not probabilities.
    """
    return AsymmetricDepolarizingChannell(p_x, p_y, p_z)



@value.value_equality
class DepolarizingChannell(raw_types.Gate):
    """A channel that depolarizes a qubit."""

    def __init__(self, p) -> None:
        r"""The symmetric depolarizing channel.

        This channel applies one of four disjoint possibilities: nothing (the
        identity channel) or one of the three pauli gates. The disjoint
        probabilities of the three gates are all the same, p / 3, and the
        identity is done with probability 1 - p. The supplied probability
        must be a valid probability or else this constructor will raise a
        ValueError.

        This channel evolves a density matrix via
            \rho -> (1 - p) \rho
                    + (p / 3) X \rho X + (p / 3) Y \rho Y + (p / 3) Z \rho Z

        Args:
            p: The probability that one of the Pauli gates is applied. Each of
                the Pauli gates is applied independently with probability p / 3.

        Raises:
            ValueError: if p is not a valid probability.
        """

        self._p = p
        self._delegate = AsymmetricDepolarizingChannell(p / 3, p / 3, p / 3)


    def _channel_(self) -> Iterable[np.ndarray]:
        return self._delegate._channel_()
    '''
    def _qid_shape_(self):
        return (3,)
    '''
    def _value_equality_values_(self):
        return self._p

    def __repr__(self) -> str:
        return 'cirq.depolarize(p={!r})'.format(self._p)

    def __str__(self) -> str:
        return 'depolarize(p={!r})'.format(self._p)

    def _circuit_diagram_info_(
        self, args: protocols.CircuitDiagramInfoArgs
    ) -> str:
        return 'D({!r})'.format(self._p)



def depolarizel(p: float) -> DepolarizingChannell:
    r"""Returns a DepolarizingChannel with given probability of error.

    This channel applies one of four disjoint possibilities: nothing (the
    identity channel) or one of the three pauli gates. The disjoint
    probabilities of the three gates are all the same, p / 3, and the
    identity is done with probability 1 - p. The supplied probability
    must be a valid probability or else this constructor will raise a
    ValueError.

    This channel evolves a density matrix via
        \rho -> (1 - p) \rho
                + (p / 3) X \rho X + (p / 3) Y \rho Y + (p / 3) Z \rho Z

    Args:
        p: The probability that one of the Pauli gates is applied. Each of
            the Pauli gates is applied independently with probability p / 3.

    Raises:
        ValueError: if p is not a valid probability.
    """
    return DepolarizingChannell(p)

#dpg = DepolarizingChannel()
moment = depolarizel(0.5)

sim = cirq.Simulator()
circuit1 = cirq.Circuit()
qutrits1 = []
for j in range(5):
    qutrits1.append(cirq.LineQubit(j))

circuit1.append(moment(qutrits1[1]), strategy=InsertStrategy.INLINE)

res1 = sim.simulate(circuit1)
print(circuit1)
print(res1)
