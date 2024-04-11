import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

import anndata
import joblib
import numpy as np
import pandas as pd
import ray
from gensim import utils
from gensim.utils import revdict
from scipy.sparse import csc_matrix

from .binarize import binarize_topics
from .infer import remote_convert_input, remote_infer
from .input import convert_input, prepare_binary_matrix
from .utils import MALLET_COMMAND_MAP, MALLET_JAVA_BASE


@ray.remote
def _do_nothing(num_topics, renew):
    return num_topics, renew


class Mallet:
    """
    A wrapper for the Mallet LDA model.

    Attributes
    ----------
        output_dir (Path): The directory where the output files will be stored.
        train_prefix (str): The prefix for the training files.
        train_mallet_file (Optional[Path]): The path to the training mallet file.
        train_id2word_file (Optional[Path]): The path to the training id2word file.
        trained (bool): A flag indicating whether the model has been trained.
        _id2word (Optional[dict]): The id2word mapping after training.
        trained_num_topics (Set[int]): The number of topics trained.
        train_cell_names (Optional[any]): The cell names used in training.
        train_region_names (Optional[any]): The region names used in training.
    """

    def __init__(self, output_dir: Union[str, Path]) -> None:
        """
        Initializes the Mallet wrapper.

        Args:
            output_dir (Union[str, Path]): The directory where the output files will be stored.
        """
        self.output_dir = Path(output_dir).resolve().absolute()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Train files
        self.train_prefix = str(self.output_dir / "train")
        self.train_mallet_file: Optional[Path] = Path(f"{self.train_prefix}_corpus.mallet")
        self.train_id2word_file: Optional[Path] = Path(f"{self.train_prefix}_corpus.id2word")
        if not self.train_mallet_file.exists() and not self.train_id2word_file.exists():
            self.trained = False
            self.train_mallet_file = None
            self.train_id2word_file = None
        else:
            self.trained = True

        # Model parameters after training
        self._id2word: Optional[dict] = None
        self.trained_num_topics: set[int] = set()
        self.train_cell_names = None
        self.train_region_names = None
        if self.trained:
            self._post_fit()

    @property
    def id2word(self) -> Optional[dict]:
        """
        Get the id2word dictionary.

        Returns
        -------
            Optional[dict]: The id2word dictionary if it exists, None otherwise.
        """
        if self._id2word is None:
            try:
                self._id2word = joblib.load(self.train_id2word_file)
            except TypeError as e:
                raise ValueError("No id2word dictionary found. Please train the model first.") from e
        return self._id2word

    @property
    def num_terms(self) -> int:
        """
        Get the number of terms in the id2word dictionary.

        Returns
        -------
            int: The number of terms in the id2word dictionary.
        """
        return 1 + max(self.id2word.keys())

    def fit(
        self,
        num_topics: list[int],
        data: Optional[csc_matrix] = None,
        cpu_per_task: int = 8,
        mem_gb: int = 16,
        **train_kwargs,
    ) -> None:
        """
        Train Mallet LDA with multiple number of topics.

        Args:
            num_topics (List[int]): List of number of topics to train.
            data (Optional[csc_matrix]): Binary matrix containing cells/documents as columns and regions/words as rows.
            cpu_per_task (int, optional): Number of CPU to use per task. Default: 8.
            mem_gb (int, optional): Memory to use in GB. Default: 16.
            **train_kwargs: Additional keyword arguments for the training. See :meth:`train`.
        """
        train_kwargs["n_cpu"] = cpu_per_task
        train_kwargs["mem_gb"] = mem_gb

        if data is not None:
            data, self.train_cell_names, self.train_region_names = prepare_binary_matrix(data)
            self.train_mallet_file, self.train_id2word_file = convert_input(
                data=data, output_prefix=self.train_prefix, mem_gb=4
            )
        else:
            assert self.trained, "No data provided and no trained model found"

        tasks = []
        if isinstance(num_topics, int):
            num_topics = [num_topics]
        for num_topic in num_topics:
            Path(self.output_dir / f"topic{num_topic}").mkdir(exist_ok=True, parents=True)
            train_kwargs["num_topics"] = num_topic
            task = self.train(**train_kwargs)
            tasks.append(task)

        while tasks:
            done, tasks = ray.wait(tasks)
            for task in done:
                num_topics, renew = ray.get(task)
                # save topic feather files
                if renew:
                    self.get_cell_topics(num_topics, renew=renew)
                    self.get_region_topics(num_topics, renew=renew)

        self.trained = True
        self._post_fit()
        return

    def _get_topic_path(self, num_topics: int, name: str, temp: bool) -> Path:
        """
        Get the path for a specific topic file.

        Parameters
        ----------
        num_topics : int
            The number of topics.
        name : str
            The name of the file.
        temp : bool
            Whether the file is temporary or not.

        Returns
        -------
        Path
            The path to the topic file.
        """
        temp_suffix = "_temp" if temp else ""
        return Path(f"{self.output_dir}/topic{num_topics}/topic{num_topics}_{name}{temp_suffix}")

    def _get_topic_state_path(self, num_topics: int, temp: bool = False) -> Path:
        """
        Get the path for the state file of a specific number of topics.

        Parameters
        ----------
        num_topics : int
            The number of topics.
        temp : bool, optional
            Whether the file is temporary or not. Default is False.

        Returns
        -------
        Path
            The path to the state file.
        """
        return self._get_topic_path(num_topics, "state.mallet.gz", temp)

    def _get_topic_doctopics_path(self, num_topics: int, temp: bool = False) -> Path:
        """
        Get the path for the doctopics file of a specific number of topics.

        Parameters
        ----------
        num_topics : int
            The number of topics.
        temp : bool, optional
            Whether the file is temporary or not. Default is False.

        Returns
        -------
        Path
            The path to the doctopics file.
        """
        return self._get_topic_path(num_topics, "doctopics.txt", temp)

    def _get_topic_inferencer_path(self, num_topics: int, temp: bool = False) -> Path:
        """
        Get the path for the inferencer file of a specific number of topics.

        Parameters
        ----------
        num_topics : int
            The number of topics.
        temp : bool, optional
            Whether the file is temporary or not. Default is False.

        Returns
        -------
        Path
            The path to the inferencer file.
        """
        return self._get_topic_path(num_topics, "inferencer.mallet", temp)

    def _get_topic_topickeys_path(self, num_topics: int, temp: bool = False) -> Path:
        """
        Get the path for the topickeys file of a specific number of topics.

        Parameters
        ----------
        num_topics : int
            The number of topics.
        temp : bool, optional
            Whether the file is temporary or not. Default is False.

        Returns
        -------
        Path
            The path to the topickeys file.
        """
        return self._get_topic_path(num_topics, "topickeys.txt", temp)

    def _get_train_flag_path(self, num_topics: int) -> Path:
        """
        Get the path for the train flag file of a specific number of topics.

        Parameters
        ----------
        num_topics : int
            The number of topics.

        Returns
        -------
        Path
            The path to the train flag file.
        """
        return self._get_topic_path(num_topics, "train_flag.txt", temp=False)

    def _get_train_cell_topics_path(self, num_topics: int, temp: bool = False, binary: bool = False) -> Path:
        """
        Get the path for the cell topics file of a specific number of topics.

        Parameters
        ----------
        num_topics : int
            The number of topics.
        temp : bool, optional
            Whether the file is temporary or not. Default is False.
        binary : bool, optional
            Whether the file is binary or not. Default is False.

        Returns
        -------
        Path
            The path to the cell topics file.
        """
        binary_str = "_binarized" if binary else ""
        return self._get_topic_path(num_topics, f"cell_topics{binary_str}.feather", temp=temp)

    def _get_train_region_topics_path(self, num_topics: int, temp: bool = False, binary: bool = False) -> Path:
        """
        Get the path for the region topics file of a specific number of topics.

        Parameters
        ----------
        num_topics : int
            The number of topics.
        temp : bool, optional
            Whether the file is temporary or not. Default is False.
        binary : bool, optional
            Whether the file is binary or not. Default is False.

        Returns
        -------
        Path
            The path to the region topics file.
        """
        binary_str = "_binarized" if binary else ""
        return self._get_topic_path(num_topics, f"region_topics{binary_str}.feather", temp=temp)

    def _temp_to_final(self, path: str) -> Optional[Path]:
        """
        Convert a temporary file path to its final path.

        Parameters
        ----------
        path : str
            The path of the file.

        Returns
        -------
        Optional[Path]
            The final path of the file, or None if the file does not exist.
        """
        if not Path(path).exists():
            return None
        path = str(path)
        if path.endswith("_temp"):
            final_path = Path(path[:-5])
            if final_path.exists():
                final_path.unlink()
            Path(path).rename(final_path)
            return Path(final_path)
        return Path(path)

    def train(
        self,
        num_topics: int,
        alpha: float = 50,
        beta: float = 0.1,
        optimize_interval: int = 0,
        optimize_burn_in: int = 200,
        topic_threshold: float = 0.0,
        iterations: int = 300,
        random_seed: int = 555,
        n_cpu: int = 8,
        mem_gb: int = 16,
    ) -> ray.ObjectRef:
        """
        Train Mallet LDA.

        Parameters
        ----------
        num_topics : int
            The number of topics to use in the model.
        alpha : float, optional
            alpha value for mallet train-topics. Default: 50.
        beta : float, optional
            beta value for mallet train-topics. Default: 0.1.
        optimize_interval : int, optional
            Optimize hyperparameters every `optimize_interval` iterations (sometimes leads to Java exception 0 to switch off hyperparameter optimization). Default: 0.
        optimize_burn_in : int, optional
            Number of iterations to run before starting hyperparameter optimization. Default: 200.
        topic_threshold : float, optional
            Threshold of the probability above which we consider a topic. Default: 0.0.
        iterations : int, optional
            Number of training iterations. Default: 300.
        random_seed : int, optional
            Random seed to ensure consistent results, if 0 - use system clock. Default: 555.
        n_cpu : int, optional
            Number of threads that will be used for training. Default: 8.
        mem_gb : int, optional
            Memory to use in GB. Default: 16.

        Returns
        -------
        ray.ObjectRef
            Ray object reference to the training task.
        """
        flag_path = self._get_train_flag_path(num_topics)

        # record number of iterations trained in total
        if flag_path.exists():
            with open(flag_path) as fin:
                cur_cycle = int(fin.read())
                if cur_cycle >= iterations:
                    return _do_nothing.remote(num_topics, False)
        else:
            cur_cycle = 0
        optimize_burn_in = max(0, optimize_burn_in - cur_cycle)

        previous_state_path = self._get_topic_state_path(num_topics)
        state_path = self._get_topic_state_path(num_topics, temp=True)
        doctopics_path = self._get_topic_doctopics_path(num_topics, temp=True)
        inferencer_path = self._get_topic_inferencer_path(num_topics, temp=True)
        topickeys_path = self._get_topic_topickeys_path(num_topics, temp=True)

        _mallet_cmd = "train-topics"
        _mallet_cmd_base = MALLET_JAVA_BASE.format(mem_gb=mem_gb, mallet_cmd=MALLET_COMMAND_MAP[_mallet_cmd])
        if Path(previous_state_path).exists():
            input_state_option = f"--input-state {previous_state_path}"
        else:
            input_state_option = ""
        cmd = (
            f"{_mallet_cmd_base} "
            f"--input {self.train_mallet_file} "
            f"{input_state_option} "
            f"--num-topics {num_topics} "
            f"--alpha {alpha} "
            f"--beta {beta} "
            f"--optimize-interval {optimize_interval} "
            f"--optimize-burn-in {optimize_burn_in} "
            f"--num-threads {int(n_cpu*2)} "
            f"--num-iterations {iterations} "
            f"--output-state {state_path} "
            f"--output-doc-topics {doctopics_path} "
            f"--output-topic-keys {topickeys_path} "
            f"--inferencer-filename {inferencer_path} "
            f"--doc-topics-threshold {topic_threshold} "
            f"--random-seed {random_seed}"
        )

        @ray.remote(num_cpus=n_cpu)
        def _train_worker(
            cmd: str,
            flag_path: Path,
            iterations: int,
            num_topics: int,
            temp_paths: list[Path],
        ) -> ray.ObjectRef:
            print("Running command:", cmd)
            try:
                subprocess.check_output(args=shlex.split(cmd), shell=False, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}") from e

            # record number of iterations trained in total
            if flag_path.exists():
                with open(flag_path) as fin:
                    cur_cycle = int(fin.read())
                cur_cycle += iterations
            else:
                cur_cycle = iterations
            with open(flag_path, "w") as fout:
                fout.write(str(cur_cycle))

            # temp to final
            for path in temp_paths:
                path.rename(path.with_name(path.name[:-5]))

            renew = True  # renew the cell and region topics
            return num_topics, renew

        _temp_paths = [state_path, doctopics_path, inferencer_path, topickeys_path]
        task = _train_worker.remote(
            cmd=cmd,
            flag_path=flag_path,
            iterations=iterations,
            num_topics=num_topics,
            temp_paths=_temp_paths,
        )
        return task

    def _load_word_topics(self, num_topics: int) -> np.ndarray:
        """
        Load words X topics matrix from :meth:`gensim.models.wrappers.LDAMallet.LDAMallet.fstate` file.

        Parameters
        ----------
        num_topics : int
            The number of topics.

        Returns
        -------
        np.ndarray
            Words X topics matrix, shape `vocabulary_size` x `num_topics`.
        """
        state_path = self._get_topic_state_path(num_topics)
        word_topics = np.zeros((num_topics, self.num_terms), dtype=np.float64)
        if hasattr(self.id2word, "token2id"):
            word2id = self.id2word.token2id
        else:
            word2id = revdict(self.id2word)

        with utils.open(state_path, "rb") as fin:
            _ = next(fin)  # header
            self.alpha = np.fromiter(next(fin).split()[2:], dtype=float)
            assert len(self.alpha) == num_topics, "Mismatch between MALLET vs. requested topics"
            _ = next(fin)  # beta
            for _, line in enumerate(fin):
                line = utils.to_unicode(line)
                *_, token, topic = line.split(" ")
                if token not in word2id:
                    continue
                tokenid = word2id[token]
                word_topics[int(topic), tokenid] += 1.0
        return word_topics

    def _save_binarized_topics(self, df: pd.DataFrame, path: Path) -> None:
        """
        Binarize topic probabilities and save to file.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing topic probabilities.
        path : Path
            The path to save the binarized topics.
        """
        temp_path = path.with_name(path.name + ".temp")
        binarized = binarize_topics(df, nbins=100)
        binarized.reset_index().to_feather(temp_path)
        temp_path.rename(path)
        return

    def get_cell_topics(self, num_topics: int, renew: bool = False) -> pd.DataFrame:
        """
        Get cell-by-topic dataframe.

        Parameters
        ----------
        num_topics : int
            Number of topics.
        renew : bool, optional
            Whether to renew the dataframe if it already exists. Default is False.

        Returns
        -------
        pd.DataFrame
            Cell by topic dataframe.
        """
        if not renew:
            final_path = self._get_train_cell_topics_path(num_topics)
            if Path(final_path).exists():
                return pd.read_feather(final_path).set_index("cell")

        temp_path = self._get_train_cell_topics_path(num_topics, temp=True)
        doctopics_path = self._get_topic_doctopics_path(num_topics)
        cell_by_topic = pd.read_csv(doctopics_path, header=None, sep="\t").iloc[:, 2:]
        cell_by_topic.index = self.train_cell_names
        cell_by_topic.index.name = "cell"
        cell_by_topic.columns = [f"topic{i}" for i in range(num_topics)]
        cell_by_topic.reset_index().to_feather(temp_path)
        self._temp_to_final(temp_path)

        binarized_path = self._get_train_cell_topics_path(num_topics, binary=True)
        self._save_binarized_topics(cell_by_topic, path=binarized_path)
        return cell_by_topic

    def get_region_topics(self, num_topics: int, renew: bool = False) -> pd.DataFrame:
        """
        Get region-by-topic dataframe and normalize by region's topic sum.

        Parameters
        ----------
        num_topics : int
            Number of topics.
        renew : bool, optional
            Whether to renew the dataframe if it already exists. Default is False.

        Returns
        -------
        pd.DataFrame
            Region by topic dataframe.
        """
        if not renew:
            final_path = self._get_train_region_topics_path(num_topics)
            if Path(final_path).exists():
                return pd.read_feather(final_path).set_index("region")

        topic_word = self._load_word_topics(num_topics)  # shape: num_topics x num_regions
        # this will panalize the regions occuring in multiple topics
        norm_topic_word = topic_word / topic_word.sum(axis=1)[:, None]

        region_by_topic = pd.DataFrame(norm_topic_word.T, index=self.train_region_names)
        region_by_topic.index.name = "region"
        region_by_topic.columns = [f"topic{i}" for i in range(num_topics)]
        temp_path = self._get_train_region_topics_path(num_topics, temp=True)
        region_by_topic.reset_index().to_feather(temp_path)
        self._temp_to_final(temp_path)

        binarized_path = self._get_train_region_topics_path(num_topics, binary=True)
        self._save_binarized_topics(region_by_topic, path=binarized_path)
        return region_by_topic

    def _post_fit(self):
        """
        Perform post-processing after fitting the model.

        Returns
        -------
            None
        """
        self.trained = True
        # scan output dir and get trained topics
        for topic_dir in self.output_dir.glob("topic*"):
            num_topics = int(topic_dir.name[5:])
            self.trained_num_topics.add(num_topics)
        return

    def parallel_infer(
        self,
        data: csc_matrix,
        use_num_topics: Optional[set[int]] = None,
        topic_threshold: float = 0.0,
        num_iterations: int = 300,
        random_seed: int = 555,
        mem_gb: int = 16,
    ) -> dict[int, pd.DataFrame]:
        """
        Infer topics for new data in parallel.

        Parameters
        ----------
            data (csc_matrix): Binary matrix containing cell/document as columns and regions/words as rows.
            use_num_topics (Optional[Set[int]]): List of number of topics to use in the model.
            topic_threshold (float): Threshold of the probability above which we consider a topic. Default: 0.0.
            num_iterations (int): Number of training iterations. Default: 300.
            random_seed (int): Random seed to ensure consistent results. Default: 555.
            mem_gb (int): Memory to use in GB. Default: 16.

        Returns
        -------
            Dict[int, pd.DataFrame]: Dictionary of inferred topics for each number of topics.
        """
        if not self.trained:
            raise ValueError(f"No trained model found at {self.output_dir}")

        with tempfile.TemporaryDirectory(prefix="bolero_") as parent_temp_dir:
            actual_train_mallet = Path(self.train_mallet_file)
            temp_mallet_path = Path(f"{parent_temp_dir}/{actual_train_mallet.name}")
            # copy the mallet file to temp path and make it read only
            subprocess.run(["cp", actual_train_mallet, temp_mallet_path], check=True)
            # make temp_mallet_path read only
            temp_mallet_path.chmod(0o444)

            if use_num_topics is not None:
                use_num_topics = set(use_num_topics) & self.trained_num_topics
            else:
                use_num_topics = self.trained_num_topics

            model_dict = {}
            for num_topics in use_num_topics:
                model_temp_dir = tempfile.mkdtemp(dir=parent_temp_dir, prefix=f"topic{num_topics}_")
                model_dict[model_temp_dir] = self._get_topic_inferencer_path(num_topics)

            data, cell_names, _ = prepare_binary_matrix(data)
            data_remote = ray.put(data)

            # get the number of cpu available and adjust the chunk size
            n_cpu = int(ray.available_resources()["CPU"] / 1.21)  # each job take 1.2 cpu
            chunk_size = max(100, (data.shape[1] + n_cpu) // n_cpu)

            # convert input for each model, this is required as the train_mallet and train_id2word files are different for each model
            futures = {}
            _futures = [
                remote_convert_input.remote(
                    data=data_remote,
                    temp_dir=parent_temp_dir,
                    chunk_start=chunk_start,
                    chunk_end=min(chunk_start + chunk_size, data.shape[1]),
                    train_mallet_file=temp_mallet_path,
                    train_id2word_file=self.train_id2word_file,
                    mem_gb=mem_gb,
                )
                for chunk_start in range(0, data.shape[1], chunk_size)
            ]
            mallet_paths = ray.get(_futures)

            # run the inference in parallel for each inferencer on each chunk
            total_futures = {}
            for model_temp_dir, inferencer_path in model_dict.items():
                futures = [
                    remote_infer.remote(
                        mallet_path=mallet_path,
                        inferencer_path=inferencer_path,
                        temp_prefix=f"{model_temp_dir}/{Path(mallet_path).stem}",
                        topic_threshold=topic_threshold,
                        num_iterations=num_iterations,
                        random_seed=random_seed,
                        mem_gb=mem_gb,
                    )
                    for mallet_path in mallet_paths
                ]
                total_futures[model_temp_dir] = futures

            # get the results
            results = {}
            for model_temp_dir, futures in total_futures.items():
                num_topics = int(Path(model_temp_dir).name.split("_")[0][5:])
                doc_topic = np.concatenate(ray.get(futures), axis=0)
                doc_topic = pd.DataFrame(
                    doc_topic,
                    index=cell_names,
                    columns=[f"topic{i}" for i in range(doc_topic.shape[1])],
                )
                results[num_topics] = doc_topic
        return results

    def infer_adata(
        self,
        adata: anndata.AnnData,
        use_num_topics: Optional[list[int]] = None,
        binarize: bool = False,
        **infer_kwargs: dict,
    ) -> None:
        """
        Infer topics for AnnData object.

        Parameters
        ----------
            adata (anndata.AnnData): AnnData object containing the new data.
            use_num_topics (Optional[List[int]]): List of number of topics to use in the model.
            binarize (bool): Whether to binarize the inferred topics. Default: False.
            infer_kwargs (Dict): Additional keyword arguments for parallel_infer.

        Returns
        -------
            None
        """
        assert isinstance(adata, anndata.AnnData), "adata must be an AnnData object"

        results = self.parallel_infer(adata, use_num_topics, **infer_kwargs)
        if binarize:
            binary_results = {}
            for num_topics, doc_topic in results.items():
                binary_results[num_topics] = binarize_topics(doc_topic, nbins=100)
            results = binary_results

        for num_topics, doc_topic in results.items():
            adata.obsm[f"{self.output_dir.name}.topic{num_topics}"] = doc_topic.values
        return
