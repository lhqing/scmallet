import shlex
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import ray
from gensim import utils
from gensim.utils import revdict

from .binarize import binarize_topics
from .input import convert_input, prepare_binary_matrix
from .utils import MALLET_COMMAND_MAP, MALLET_JAVA_BASE


@ray.remote
def _do_nothing(num_topics):
    return num_topics


class Mallet:
    """Mallet LDA model wrapper."""

    def __init__(self, output_dir) -> None:
        self.output_dir = Path(output_dir).resolve().absolute()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Train files
        self.train_prefix = str(self.output_dir / "train")
        self.train_mallet_file = Path(f"{self.train_prefix}_corpus.mallet")
        self.train_id2word_file = Path(f"{self.train_prefix}_corpus.id2word")
        if not self.train_mallet_file.exists() and not self.train_id2word_file.exists():
            self.trained = False
            self.train_mallet_file = None
            self.train_id2word_file = None
        else:
            self.trained = True

        # Model parameters after training
        self._id2word = None
        self.trained_num_topics = set()
        self.train_cell_names = None
        self.train_region_names = None
        if self.trained:
            self._post_fit()

    @property
    def id2word(self):
        """Get the id2word dictionary."""
        if self._id2word is None:
            try:
                self._id2word = joblib.load(self.train_id2word_file)
            except TypeError as e:
                raise ValueError("No id2word dictionary found. Please train the model first.") from e
        return self._id2word

    @property
    def num_terms(self):
        """Get the number of terms in the id2word dictionary."""
        return 1 + max(self.id2word.keys())

    def fit(self, num_topics, data=None, cpu_per_task=8, mem_gb=16, **train_kwargs):
        """
        Train Mallet LDA with multiple number of topics.

        Parameters
        ----------
        num_topics: List[int]
            List of number of topics to train.
        data: csc_matrix
            Binary matrix containing cells/documents as columns and regions/words as rows.
        cpu_per_task: int, optional
            Number of CPU to use per task. Default: 8.
        mem_gb: int, optional
            Memory to use in GB. Default: 16.
        train_kwargs: Dict
            Additional keyword arguments for :meth:`train`.
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
                num_topics = ray.get(task)
                # save topic feather files
                self.get_cell_topics(num_topics, renew=True)
                self.get_region_topics(num_topics, renew=True)

        self.trained = True
        self._post_fit()
        return

    def _get_topic_path(self, num_topics, name, temp):
        temp_suffix = "_temp" if temp else ""
        return Path(f"{self.output_dir}/topic{num_topics}/topic{num_topics}_{name}{temp_suffix}")

    def _get_topic_state_path(self, num_topics, temp=False):
        return self._get_topic_path(num_topics, "state.mallet.gz", temp)

    def _get_topic_doctopics_path(self, num_topics, temp=False):
        return self._get_topic_path(num_topics, "doctopics.txt", temp)

    def _get_topic_inferencer_path(self, num_topics, temp=False):
        return self._get_topic_path(num_topics, "inferencer.mallet", temp)

    def _get_topic_topickeys_path(self, num_topics, temp=False):
        return self._get_topic_path(num_topics, "topickeys.txt", temp)

    def _get_train_flag_path(self, num_topics):
        return self._get_topic_path(num_topics, "train_flag.txt", temp=False)

    def _get_train_cell_topics_path(self, num_topics, temp=False):
        return self._get_topic_path(num_topics, "cell_topics.feather", temp=temp)

    def _get_train_region_topics_path(self, num_topics, temp=False):
        return self._get_topic_path(num_topics, "region_topics.feather", temp=temp)

    def _temp_to_final(self, path):
        if not Path(path).exists():
            return
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
        num_topics,
        alpha=50,
        beta=0.1,
        optimize_interval=0,
        optimize_burn_in=200,
        topic_threshold=0.0,
        iterations=300,
        random_seed=555,
        n_cpu=8,
        mem_gb=16,
    ):
        """
        Train Mallet LDA.

        Parameters
        ----------
        output_prefix: str
            Prefix to save the output files.
        num_topics: int
            The number of topics to use in the model.
        input_state: str, optional
            Path to the state file to use as input. Default: None.
        alpha: float, optional
            alpha value for mallet train-topics. Default: 50.
        eta: float, optional
            beta value for mallet train-topics. Default: 0.1.
        optimize_interval : int, optional
            Optimize hyperparameters every `optimize_interval` iterations (sometimes leads to Java exception 0 to switch off hyperparameter optimization). Default: 0.
        topic_threshold : float, optional
            Threshold of the probability above which we consider a topic. Default: 0.0.
        iterations : int, optional
            Number of training iterations. Default: 150.
        random_seed: int, optional
            Random seed to ensure consistent results, if 0 - use system clock. Default: 555.
        n_cpu : int, optional
            Number of threads that will be used for training. Default: 1.
        mem_gb: int, optional
            Memory to use in GB. Default: 16.

        Returns
        -------
        ray.Task
            Ray task to train the model.
        """
        flag_path = self._get_train_flag_path(num_topics)

        # record number of iterations trained in total
        if flag_path.exists():
            with open(flag_path) as fin:
                cur_cycle = int(fin.read())
                if cur_cycle >= iterations:
                    return _do_nothing.remote(num_topics)
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
        def _train_worker(cmd, flag_path, iterations, num_topics, temp_paths):
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
            return num_topics

        _temp_paths = [state_path, doctopics_path, inferencer_path, topickeys_path]
        task = _train_worker.remote(
            cmd=cmd,
            flag_path=flag_path,
            iterations=iterations,
            num_topics=num_topics,
            temp_paths=_temp_paths,
        )
        return task

    def _load_word_topics(self, num_topics):
        """
        Load words X topics matrix from :meth:`gensim.models.wrappers.LDAMallet.LDAMallet.fstate` file.

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

    def _save_binarized_topics(self, df, path):
        """Binarize topic probabilities and save to file."""
        temp_path = path.with_name(path.name + ".temp")
        binarized = binarize_topics(df, nbins=100)
        binarized.reset_index().to_feather(temp_path)
        self._temp_to_final(temp_path)
        return

    def get_cell_topics(self, num_topics: int, renew=False) -> pd.DataFrame:
        """
        Get cell-by-topic dataframe.

        Parameters
        ----------
        num_topics : int
            Number of topics.

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
        path = self._temp_to_final(temp_path)

        binarized_path = path.with_name(path.name[:-7] + "_binarized.feather")
        self._save_binarized_topics(cell_by_topic, path=binarized_path)
        return cell_by_topic

    def get_region_topics(self, num_topics: int, renew=False) -> pd.DataFrame:
        """
        Get region-by-topic dataframe and normalize by region's topic sum.

        Parameters
        ----------
        num_topics : int
            Number of topics.

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
        path = self._temp_to_final(temp_path)

        binarized_path = path.with_name(path.name[:-7] + "_binarized.feather")
        self._save_binarized_topics(region_by_topic, path=binarized_path)
        return region_by_topic

    def _post_fit(self):
        self.trained = True

        # scan output dir and get trained topics
        for topic_dir in self.output_dir.glob("topic*"):
            num_topics = int(topic_dir.name[5:])
            self.trained_num_topics.add(num_topics)
