import pathlib
import subprocess
from itertools import chain
from pathlib import Path
from typing import Union

import anndata
import joblib
import numpy as np
import pandas as pd
from gensim import corpora, matutils, utils
from gensim.utils import check_output
from scipy.sparse import csc_matrix, issparse

from .utils import MALLET_COMMAND_MAP, MALLET_JAVA_BASE


def convert_input(
    data: csc_matrix,
    output_prefix: str,
    train_mallet_file: Union[str, Path] = None,
    train_id2word_file: Union[str, Path] = None,
    mem_gb: int = 4,
):
    """
    Convert sparse.csc_matrix to Mallet format and save it to a binary file, also save the id2word dictionary.

    Parameters
    ----------
    data : sparse.csc_matrix
        binary matrix containing cells/documents as columns and regions/words as rows.
    output_prefix : str
        Prefix to save the output files.
    train_mallet_file : str or Path, optional
        If preparing the corpus for inference, provide the path to the Mallet file used to train the model. Default: None.
    train_id2word_file : str or Path, optional
        If preparing the corpus for inference, provide the path to the id2word dictionary used to train the model. Default: None.
    mem_gb : int, optional
        Memory to use in GB. Default: 4.

    Returns
    -------
    mallet_path : str
        Path to the corpus in Mallet format.
    id2word_path : str
        Path to the id2word dictionary. If the model is trained, this will be the same as the train_id2word_file.
    """
    if not issparse(data):
        raise ValueError("data should be a sparse matrix")
    if not isinstance(data, csc_matrix):
        data = data.tocsc()

    corpus = matutils.Sparse2Corpus(data)

    if train_mallet_file is not None or train_id2word_file is not None:
        trained = True
        assert train_id2word_file is not None, "train_id2word_file and train_mallet_file have to be provided together"
        assert train_mallet_file is not None, "train_id2word_file and train_mallet_file have to be provided together"
        train_mallet_path = pathlib.Path(train_mallet_file)
        # this will change the permissions of the file to read-only, because the mallet java command will try
        # to write to this file everytime it is used to keep the mallet dictionary up-to-date
        # however, this create a problem when running the mallet command in parallel
        # each parallel process will try to write to the same file and the mallet command will fail
        # also in our usecase the mallet file does not need to be updated if the model is already trained
        # also, chmod will not work in gcsfuse mounted directories, the current solution is to copy the file to a temp file
        # get current permissions of the file
        _cur_permissions = train_mallet_path.stat().st_mode
        train_mallet_path.chmod(0o444)
        id2word_path = pathlib.Path(train_id2word_file)
        id2word = joblib.load(id2word_path)
    else:
        _cur_permissions = None
        trained = False
        names_dict = {x: str(x) for x in range(data.shape[0])}
        id2word = corpora.Dictionary.from_corpus(corpus, names_dict)
        id2word_path = pathlib.Path(f"{output_prefix}_corpus.id2word")
        temp_id2word_path = pathlib.Path(f"{output_prefix}_corpus.id2word.temp")

    txt_path = pathlib.Path(f"{output_prefix}_corpus.txt")
    mallet_path = pathlib.Path(f"{output_prefix}_corpus.mallet")
    temp_mallet_path = pathlib.Path(f"{output_prefix}_corpus.mallet.temp")

    if trained:
        if mallet_path.exists():
            if _cur_permissions is not None:
                train_mallet_path.chmod(_cur_permissions)
            return mallet_path, id2word_path
    else:
        if mallet_path.exists() and id2word_path.exists():
            if _cur_permissions is not None:
                train_mallet_path.chmod(_cur_permissions)
            return mallet_path, id2word_path

    with utils.open(txt_path, "wb") as fout:
        for docno, doc in enumerate(corpus):
            tokens = chain.from_iterable([id2word[tokenid]] * int(cnt) for tokenid, cnt in doc)
            fout.write(utils.to_utf8(f"{docno} 0 {' '.join(tokens)}\n"))

    # save mallet binary file
    _mallet_cmd = "import-file"
    _mallet_cmd_base = MALLET_JAVA_BASE.format(mem_gb=mem_gb, mallet_cmd=MALLET_COMMAND_MAP[_mallet_cmd])

    cmd = (
        f"{_mallet_cmd_base} "
        "--preserve-case "
        "--keep-sequence "
        '--remove-stopwords --token-regex "\\S+" '
        f"--input {txt_path} "
        f"--output {temp_mallet_path} "
    )
    if trained:
        cmd += f"--use-pipe-from {train_mallet_path} "
    try:
        check_output(args=cmd, shell=True, stderr=subprocess.STDOUT, encoding="utf8")
    except subprocess.CalledProcessError as e:
        # Here Java will raise an error about the train-mallet-file being read-only and permission denied
        # This is expected because we modified the permissions of the file to read-only above
        # This is the last step of preparing the input file, the input mallet file is already created
        # so we can ignore this error
        # This is a ugly way to handle this error, but it is the only way to avoid the error (chatgpt said that)
        if "java.io.FileNotFoundException" in e.output:
            # print('there there, it is ok')
            # The java line that raises this erore is:
            # https://github.com/mimno/Mallet/blob/master/src/cc/mallet/classify/tui/Csv2Vectors.java#L336
            pass
        else:
            raise RuntimeError(f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}") from e
    txt_path.unlink()

    if not trained:
        # dump id2word to a file
        joblib.dump(id2word, temp_id2word_path)
        temp_id2word_path.rename(id2word_path)
        # otherwise the id2word file will be the same as the train_id2word_file

    temp_mallet_path.rename(mallet_path)
    if _cur_permissions is not None:
        train_mallet_path.chmod(_cur_permissions)
    return mallet_path, id2word_path


def prepare_binary_matrix(data):
    """
    Prepare the binary matrix for Mallet input.

    Parameters
    ----------
    data : anndata.AnnData or pd.DataFrame or pathlib.Path or str
        Annotated data matrix containing cells as rows and regions as columns.

    Returns
    -------
    binary_matrix : sparse.csc_matrix
        binary matrix containing cells as columns and regions as rows.
    cell_names : pd.Index
        Names of the cells.
    region_names : pd.Index
        Names of the regions.
    """
    # binary_matrix is a matrix containing cells as columns and regions as rows, following the cisTopic format
    if isinstance(data, anndata.AnnData):
        binary_matrix = data.X.T  # cells as columns, regions as rows
        cell_names = data.obs_names
        region_names = data.var_names
    elif isinstance(data, pd.DataFrame):
        binary_matrix = data.values.T
        cell_names = data.index
        region_names = data.columns
    elif isinstance(data, (pathlib.Path, str)):
        _data = str(data)
        if _data.endswith("h5ad"):
            return prepare_binary_matrix(anndata.read_h5ad(_data))
        else:
            raise ValueError(
                "data has to be an anndata.AnnData or a pd.DataFrame or a pathlib.Path or a str path to a h5ad file."
            )
    else:
        raise ValueError(
            "data has to be an anndata.AnnData or a pd.DataFrame or a pathlib.Path or a str path to a h5ad file."
        )

    if isinstance(binary_matrix, np.ndarray):
        binary_matrix = csc_matrix(binary_matrix)
    elif issparse(binary_matrix):
        binary_matrix = binary_matrix.tocsc()
    else:
        raise ValueError("binary_matrix has to be a numpy.ndarray or a sparse.csc_matrix")
    return binary_matrix, cell_names, region_names
