import cirq

msg, R, S = cirq.LineQubit.range(3)
circuit = cirq.Circuit(cirq.H(R), cirq.CNOT(R, S))

full_wavefunction = cirq.final_wavefunction(circuit, qubit_order=[msg, R, S])
qubit_mixture = cirq.wavefunction_partial_trace_as_mixture(full_wavefunction, keep_indices=[1])
for probability, case in qubit_mixture:
    print(f'{probability:%} {cirq.dirac_notation(case)}')
