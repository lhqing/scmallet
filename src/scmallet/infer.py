import shlex
import subprocess

import pandas as pd
import ray

from .input import convert_input
from .utils import MALLET_COMMAND_MAP, MALLET_JAVA_BASE


@ray.remote(num_cpus=2)
def _remote_convert_input(
    data: ray.ObjectRef,
    chunk_start: int,
    chunk_end: int,
    temp_dir: str,
    train_mallet_file: str,
    train_id2word_file: str,
    mem_gb=4,
):
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
def _remote_infer(
    mallet_path,
    inferencer_path,
    temp_prefix,
    topic_threshold,
    num_iterations,
    random_seed,
    mem_gb,
):
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


# class TopicInferrerMixin:
#     """Infer topics for new data in parallel using trained models."""
#     def parallel_infer(
#         self,
#         data,
#         topic_threshold=0.0,
#         num_iterations=300,
#         random_seed=555,
#         mem_gb=16,
#     ):
#         """
#         Infer topics for new data in parallel.

#         Parameters
#         ----------
#         data: sparse.csr_matrix
#             Binary matrix containing cell/document as columns and regions/words as rows.
#         model_dirs: list of str
#             List of paths to the model directories.
#         output_prefix: str
#             Prefix to save the output files.
#         topic_threshold : float, optional
#             Threshold of the probability above which we consider a topic. Default: 0.0.
#         num_iterations : int, optional
#             Number of training iterations. Default: 300.
#         random_seed: int, optional
#             Random seed to ensure consistent results. Default: 555.
#         mem_gb: int, optional
#             Memory to use in GB. Default: 16.
#         """
#         with tempfile.TemporaryDirectory(prefix="bolero_") as parent_temp_dir:
#             model_dict = {}
#             for model_dir in model_dirs:
#                 model_temp_dir = tempfile.mkdtemp(dir=parent_temp_dir, prefix=model_dir.name)
#                 model_files = {
#                     "inferencer": {},
#                     "train_mallet": pathlib.Path(f"{model_temp_dir}/train_corpus.mallet"),
#                     "train_id2word": model_dir / "train_corpus.id2word",
#                 }
#                 actual_train_mallet = model_dir / "train_corpus.mallet"
#                 assert actual_train_mallet.exists(), f"train_corpus.mallet file does not exist in {model_dir}"
#                 # copy the mallet file to temp path and make it read only
#                 subprocess.run(["cp", actual_train_mallet, model_files["train_mallet"]], check=True)
#                 model_files["train_mallet"].chmod(0o444)

#                 assert model_files["train_id2word"].exists(), f"train_corpus.id2word file does not exist in {model_dir}"

#                 inferencer_paths = list(pathlib.Path(model_dir).rglob("model*/*inferencer.mallet"))
#                 for inferencer_path in inferencer_paths:
#                     topic_model_name = inferencer_path.parent.name
#                     model_dir_name = model_dir.name
#                     # inferencer name will be unique
#                     model_files["inferencer"][f"{model_dir_name}_{topic_model_name}"] = str(inferencer_path)
#                 model_dict[model_temp_dir] = model_files

#             data, cell_names, _ = prepare_binary_matrix(data)
#             data_remote = ray.put(data)

#             # get the number of cpu available and adjust the chunk size
#             n_cpu = int(ray.available_resources()["CPU"])
#             chunk_size = max(100, (data.shape[1] + n_cpu) // int(n_cpu / 2))

#             # convert input for each model, this is required as the train_mallet and train_id2word files are different for each model
#             futures = {}
#             for model_temp_dir, model_files in model_dict.items():
#                 # split the data in chunks and prepare inputs
#                 _futures = [
#                     _remote_convert_input.remote(
#                         data=data_remote,
#                         temp_dir=model_temp_dir,
#                         chunk_start=chunk_start,
#                         chunk_end=min(chunk_start + chunk_size, data.shape[1]),
#                         train_mallet_file=model_files["train_mallet"],
#                         train_id2word_file=model_files["train_id2word"],
#                         mem_gb=mem_gb,
#                     )
#                     for chunk_start in range(0, data.shape[1], chunk_size)
#                 ]
#                 futures[model_temp_dir] = _futures
#             # the mallet paths for each model
#             mallet_paths_dict = {}
#             for model_temp_dir, _futures in futures.items():
#                 mallet_paths = ray.get(_futures)
#                 mallet_paths_dict[model_temp_dir] = mallet_paths

#             # run the inference in parallel for each inferencer on each chunk
#             inferencer_future_dict = {}
#             for model_temp_dir, model_files in model_dict.items():
#                 mallet_paths = mallet_paths_dict[model_temp_dir]
#                 for inferencer_name, inferencer_path in model_files["inferencer"].items():
#                     temp_dir = tempfile.mkdtemp(dir=parent_temp_dir, prefix=inferencer_name)
#                     futures = [
#                         _remote_infer.remote(
#                             mallet_path=mallet_path,
#                             inferencer_path=inferencer_path,
#                             temp_prefix=f"{temp_dir}/{pathlib.Path(mallet_path).stem}",
#                             topic_threshold=topic_threshold,
#                             num_iterations=num_iterations,
#                             random_seed=random_seed,
#                             mem_gb=mem_gb,
#                         )
#                         for mallet_path in mallet_paths
#                     ]
#                     inferencer_future_dict[inferencer_name] = futures

#             # get the results
#             doc_topic_dict = {}
#             for name, futures in inferencer_future_dict.items():
#                 doc_topic = np.concatenate(ray.get(futures), axis=0)
#                 doc_topic = pd.DataFrame(
#                     doc_topic,
#                     index=cell_names,
#                     columns=[f"topic{i}" for i in range(doc_topic.shape[1])],
#                 )
#                 doc_topic_dict[name] = doc_topic
#         return doc_topic_dict
