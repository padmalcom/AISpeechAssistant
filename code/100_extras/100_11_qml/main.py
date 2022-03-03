import cirq

q1 = cirq.NamedQubit('a')
q2 = cirq.NamedQubit('b')

# Lege den Schaltkreis an
circuit = cirq.Circuit(
  cirq.H(q1),
  cirq.CNOT(control=q1, target=q2),
  cirq.measure(q1, q2)
)
print("Circuit:")
print(circuit)

# Starte die Simulation
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=20)
print("Results:")
print(result)
