# The following script takes the classical Monte Carlo approach to simulate hail
# events (MC_hail.py) and modifies to use quantum Monte Carlo (QMC) methods.
# Specifically, we will you quantum circuit Born machines (QCBM) to learn the
# Probability Mass Function (PMF) of events and hail size, and also
# quantum amplitude estimation (QAE) to estimate the expected loss.
# Note that Qiskit 2.0.2 and qiskit-aer 0.17.0 are required to run this script.
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import nbinom
from scipy.special import rel_entr
from scipy.optimize import minimize
from collections import Counter


from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit.circuit import ParameterVector

import os
import datetime
import glob

from library import set_plot_style

# Set plot style
set_plot_style()

# ------------------------------------------------------------------------------
#    Variables
# Define any useful variables here.
# ------------------------------------------------------------------------------
n_years = int(1e6)  # Number of years to simulate
n_weeks = 48  # Number of weeks in a year
random_seed = 42
start_year = 1980
end_year = 2024
time_span = end_year - start_year  # Time span in years
qc_depth = 4  # Depth of the quantum circuit for QCBM
n_shots = 2048  # Number of shots for QCBM simulation


# ------------------------------------------------------------------------------
#    Functions
# Section of useful functions
# ------------------------------------------------------------------------------
def damage_function(size):
    # Damage function depending on hail size. Size is assumed to be the diameter
    # of the hail in inches.
    if size < 2:
        return 0
    elif size < 3:
        return 0.1
    elif size < 5:
        return 0.4
    else:
        return 0.9


def calc_neg_binomial_params(data):
    """Calculate negative binomial parameters using method of moments.

    A negative binomial distribution is good for data that is overdispersed (variance
    much larger than mean). The parameters for the negative binomial distribution,
    r and p, are calculated after checking that the data is oversdispersed. If it
    is not, we fallback to a Poisson assumption. Instaed of having to call a Poisson
    distribution later on, the limits for the negative binomial distribution are
    set to follow Poisson by letting r=infinity and p close to 1.

    Parameters:
    ----------
    r : float
        Number of scuccesses until the experiment is stopped.
    p : float
        Probability of success in each trial.
    """
    mu = data.mean()
    var = data.var()  # Fix: use actual sample variance

    if var <= mu:
        # fallback to Poisson assumption
        return np.inf, mu / (mu + 1e-9)  # p close to 1

    r = mu**2 / (var - mu)
    p = r / (r + mu)
    return r, p


def create_qcbm_circuit(num_qubits, depth):
    """Create a quantum circuit for the Quantum Circuit Born Machine (QCBM).

    The structure uses ring entanglement where each quibit is entangled using a
    CNOT gate with its successive neighbor, e.g., q_1 -> q_2 -> q_3 -> q_1. Each
    quibit is operated on by a Ry rotation gate and teh entire structures is
    repeated for depth number of times.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Depth of the circuit, i.e., number of layers of gates.

    Returns
    -------
    qc : QuantumCircuit
        The quantum circuit for the QCBM that can be used for measurement and
        optimization.
    params : dict or list
        Contains parameter objects from qiskit.circuit.Parameter.
    """
    # Each qubit gets 1 rotation per layer → total parameters = num_qubits * depth
    num_params = num_qubits * 7  # 3 Rx, 3 Rz per qubit (grouped into 3 rotation layers)
    params = ParameterVector("theta", length=num_params)
    qc = QuantumCircuit(num_qubits)

    param_idx = 0

    # Rotation Layer 1: Rx -> Rz
    for q in range(num_qubits):
        qc.rx(params[param_idx], q)
        param_idx += 1
        qc.rz(params[param_idx], q)
        param_idx += 1

    # Entangling Layer 1: CNOT ladder (q → q+1)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)

    # Rotation Layer 2: Rz -> Rx -> Rz
    for q in range(num_qubits):
        qc.rz(params[param_idx], q)
        param_idx += 1
        qc.rx(params[param_idx], q)
        param_idx += 1
        qc.rz(params[param_idx], q)
        param_idx += 1

    # Entangling Layer 2: CNOT ladder (q+1 → q)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)

    # Rotation Layer 3: Rz -> Rx
    for q in range(num_qubits):
        qc.rz(params[param_idx], q)
        param_idx += 1
        qc.rx(params[param_idx], q)
        param_idx += 1

    qc.measure_all()
    return qc, params


def kl_divergence(p_target, p_model):
    """Compute KL divergence between two PMFs."""
    # Smooth with epsilon to avoid log(0)
    epsilon = 1e-9
    p_target = np.array(p_target) + epsilon
    p_model = np.array(p_model) + epsilon
    p_target /= p_target.sum()
    p_model /= p_model.sum()
    return np.sum(rel_entr(p_target, p_model))


def get_qcbm_probs(qc, params, param_values, num_labels, shots):
    """Function to create circuit, sumulate, and measure.

    The QCBM circuit is assigned the parameter values and then simulated for a
    number of shots. Each shot, the bitstring measured is collected under counts.
    These bitstrings correspond to specific values encoded and the bitstrings are
    transformed to these known labels and collected as freqs.

    Parameters
    ----------
    qc : QuantumCircuit object
        The designed QCBM that is to be simulated.
    params : dict or list
        Contains the Ry rotation gates.
    param_values : np.array
        Contains the Ry rotation angle values in radiasn.
    num_labels : int
        Number of labels that are relevant and are to be extracted.

    Returns
    -------
    pmf : np.array
        Probability mass function (PMF) of the labels after simulation.
    """
    bound = qc.assign_parameters(param_values)
    simulator = Aer.get_backend("qasm_simulator")
    compiled = transpile(bound, simulator)
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings to integers and filter to valid label range
    freq = np.zeros(num_labels)
    for bitstring, count in counts.items():
        label = int(bitstring, 2)
        if label < num_labels:
            freq[label] += count

    pmf = freq / freq.sum()  # Normalize to get PMF

    return pmf


def qcbm_objective(param_values, qc, params, target_pmf, shots):
    """Objective function for QCBM optimization."""
    num_labels = len(target_pmf)
    model_pmf = get_qcbm_probs(qc, params, param_values, num_labels, shots)
    return kl_divergence(target_pmf, model_pmf)


# ------------------------------------------------------------------------------
#    Ingest and Process Data
# Ingest data from concatenated csv file (see hail_data_processing.py) and
# perform any operations like converting to datetime, etc.
# ------------------------------------------------------------------------------
df = pd.read_csv("concat_hail_data.csv")
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Convert the BEGIN_DATE and END_DATE columns to datetime objects
df["BEGIN_DATE"] = pd.to_datetime(df["BEGIN_DATE"])
df["END_DATE"] = pd.to_datetime(df["END_DATE"])

# Create a column of only the year and find hail events per year
df["YEAR"] = df["BEGIN_DATE"].dt.year
events_per_year = df.groupby("YEAR").size()

# Calculate the negative binomial parameters
r, p = calc_neg_binomial_params(events_per_year)

# ------------------------------------------------------------------------------
#    Probability Mass Function (PMF)
# Create PMFs for hail event and hail size using the historical data
# The PMF section is going to be used for QCBM techniques.
# ------------------------------------------------------------------------------
# Collect the weekly counts over all years
weekly_counts = df.groupby("MONTH_WEEK_INDEX").size()
all_weeks = np.arange(1, 49)
weekly_counts = weekly_counts.reindex(all_weeks, fill_value=0)
avg_weekly_counts = weekly_counts / time_span
event_pmf = weekly_counts / weekly_counts.sum()

# Get unique hail sizes and their counts
size_counts = df["MAGNITUDE"].value_counts().sort_index()
hail_sizes = size_counts.index.to_numpy()
hail_freqs = size_counts.to_numpy()
hail_probs = hail_freqs / hail_freqs.sum()

# Create a mapping from hail size to integer label (bitstring encoding)
hail_size_to_label = {size: i for i, size in enumerate(hail_sizes)}
label_to_hail_size = {i: size for i, size in enumerate(hail_sizes)}

# Create labeled data
labeled_data = df["MAGNITUDE"].map(hail_size_to_label).dropna().astype(int)

# Calculate number of quibits needed for encoding hail sizes
num_qubits = int(np.ceil(np.log2(len(hail_sizes))))

# Create quantum circuit and paramter vector for QCBM
qc, params = create_qcbm_circuit(num_qubits, qc_depth)

# Generate random parameter values for now (will optimize later)
np.random.seed(random_seed)
param_values = 2 * np.pi * np.random.rand(len(params))  # between 0 and 2π

# Bind parameters to circuit
bound_circuit = qc.assign_parameters(param_values)

# Simulate
simulator = Aer.get_backend("qasm_simulator")
compiled = transpile(bound_circuit, simulator)
result = simulator.run(compiled, shots=n_shots).result()
counts = result.get_counts()

# Show histogram of bitstrings
plot_histogram(counts)
plt.tight_layout()
plt.savefig("qcbm_histogram.pdf", dpi=800, bbox_inches="tight")
# plt.show()

# Optimize the QCMB parameters to match the target PMF
init_params = 2 * np.pi * np.random.rand(len(params))

# Optimize
result = minimize(
    qcbm_objective,
    init_params,
    args=(qc, params, hail_probs, n_shots),
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

trained_params = result.x

final_probs = get_qcbm_probs(qc, params, trained_params, len(hail_probs), n_shots)
print("Empirical Hail PMF:", np.round(hail_probs, 3))
print("Trained QCBM PMF:  ", np.round(final_probs, 3))

plt.figure(figsize=(12, 5))
x = np.arange(len(hail_probs))
plt.bar(x - 0.2, hail_probs, width=0.4, label="Empirical Hail PMF")
plt.bar(x + 0.2, final_probs, width=0.4, label="Trained QCBM PMF")
plt.xlabel("Hail Size Label (Integer Encoding)")
plt.ylabel("Probability")
plt.title("Comparison of Empirical and QCBM-Generated Hail Size PMF")
plt.xticks(x)
plt.legend()
plt.tight_layout()
plt.savefig("qcbm_pmf_comparison.pdf", dpi=800, bbox_inches="tight")
plt.show()
