"""Reading and writing the binary messages exchanged with Java (EnvServer.java, PythonPolicy.java).

Everything is big-endian. A sparse vector is: int nnz, nnz x int32 indices, nnz x float32 values.
A decision (Features.Encoded in Java) is: obs (sparse), int A, then A action rows (sparse).
"""
import struct

import numpy as np

MAGIC = 0x41434745          # "ACGE", must match EnvServer.MAGIC


def read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise EOFError("Java process closed the pipe (see its error output above)")
    return data


def read_int(f):
    return struct.unpack(">i", read_exact(f, 4))[0]


def read_double(f):
    return struct.unpack(">d", read_exact(f, 8))[0]


def read_sparse(f, size):
    """A sparse vector from Java -> dense float32 array of length `size`."""
    nnz = read_int(f)
    idx = np.frombuffer(read_exact(f, 4 * nnz), ">i4")
    val = np.frombuffer(read_exact(f, 4 * nnz), ">f4")
    out = np.zeros(size, np.float32)
    out[idx] = val
    return out


def read_decision(f, obs_size, act_size):
    """-> obs [obs_size], acts [A, act_size] (one row per legal action, in Java's order)."""
    obs = read_sparse(f, obs_size)
    n = read_int(f)
    acts = np.stack([read_sparse(f, act_size) for _ in range(n)]) if n else np.zeros((0, act_size), np.float32)
    return obs, acts


def write_ints(f, *values):
    f.write(struct.pack(">" + "i" * len(values), *values))
