import shlex
import subprocess

import pandas as pd
import ray

from .input import convert_input
from .utils import MALLET_COMMAND_MAP, MALLET_JAVA_BASE


@ray.remote(num_cpus=2)
def remote_convert_input(
    data: ray.ObjectRef,
    chunk_start: int,
    chunk_end: int,
    temp_dir: str,
    train_mallet_file: str,
    train_id2word_file: str,
    mem_gb=4,
):
    """Convert sparse.csc_matrix to Mallet format and save it to a binary file, also save the id2word dictionary. Ray version."""
    # get a random dir to save the mallet files
    temp_prefix = f"{temp_dir}/infer_{chunk_start}_{chunk_end}"
    _data = data[:, chunk_start:chunk_end]
    mallet_path, _ = convert_input(
        data=_data,
        output_prefix=temp_prefix,
        mem_gb=mem_gb,
        train_mallet_file=train_mallet_file,
        train_id2word_file=train_id2word_file,
    )
    return mallet_path


def _infer(
    mallet_path,
    inferencer_path,
    output_path,
    topic_threshold,
    num_iterations,
    random_seed,
    mem_gb,
):
    """Infer topics for new documents."""
    _mallet_cmd = "infer-topics"
    _mallet_cmd_base = MALLET_JAVA_BASE.format(mem_gb=mem_gb, mallet_cmd=MALLET_COMMAND_MAP[_mallet_cmd])
    cmd = (
        f"{_mallet_cmd_base} "
        f"--inferencer {inferencer_path} "
        f"--input {mallet_path} "
        f"--output-doc-topics {output_path} "
        f"--doc-topics-threshold {topic_threshold} "
        f"--num-iterations {num_iterations} "
        f"--random-seed {random_seed} "
    )
    try:
        subprocess.check_output(args=shlex.split(cmd), shell=False, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}") from e
    return


@ray.remote(num_cpus=1)
def remote_infer(
    mallet_path,
    inferencer_path,
    temp_prefix,
    topic_threshold,
    num_iterations,
    random_seed,
    mem_gb,
):
    """Infer topics for new documents, ray version."""
    _infer(
        mallet_path=mallet_path,
        inferencer_path=inferencer_path,
        output_path=f"{temp_prefix}_doctopics.txt",
        topic_threshold=topic_threshold,
        num_iterations=num_iterations,
        random_seed=random_seed,
        mem_gb=mem_gb,
    )
    # load doc_topic table
    doctopics_path = f"{temp_prefix}_doctopics.txt"
    doc_topic = pd.read_csv(doctopics_path, header=None, sep="\t", comment="#").iloc[:, 2:].to_numpy()
    return doc_topic
