import pathlib
import subprocess


def get_mallet_path():
    """Get the path to the Mallet executable."""
    try:
        path = subprocess.check_output(["which", "mallet"], encoding="utf8")
    except subprocess.CalledProcessError:
        print(
            "LDA package Mallet not found. "
            "Please install it with `conda install -c conda-forge mallet` "
            "or `mamba install -c conda-forge mallet`"
        )
        raise FileNotFoundError("Mallet not found.")
    return path.strip()


MALLET_PATH = get_mallet_path()
MALLET_PATH = pathlib.Path(MALLET_PATH)
_mallet_grandparent = MALLET_PATH.parent.parent
MALLET_JAVA_BASE = (
    "java "
    "-Xmx{mem_gb}g "
    "-ea "
    "-Djava.awt.headless=true "
    "-Dfile.encoding=UTF-8 "
    "-server "
    f"-classpath {_mallet_grandparent}/class:{_mallet_grandparent}/lib/mallet-deps.jar: "
    "{mallet_cmd} "
)


MALLET_COMMAND_MAP = {
    "import-dir": "cc.mallet.classify.tui.Text2Vectors",
    "import-file": "cc.mallet.classify.tui.Csv2Vectors",
    "import-svmlight": "cc.mallet.classify.tui.SvmLight2Vectors",
    "info": "cc.mallet.classify.tui.Vectors2Info",
    "train-classifier": "cc.mallet.classify.tui.Vectors2Classify",
    "classify-dir": "cc.mallet.classify.tui.Text2Classify",
    "classify-file": "cc.mallet.classify.tui.Csv2Classify",
    "classify-svmlight": "cc.mallet.classify.tui.SvmLight2Classify",
    "train-topics": "cc.mallet.topics.tui.TopicTrainer",
    "infer-topics": "cc.mallet.topics.tui.InferTopics",
    "evaluate-topics": "cc.mallet.topics.tui.EvaluateTopics",
    "prune": "cc.mallet.classify.tui.Vectors2Vectors",
    "split": "cc.mallet.classify.tui.Vectors2Vectors",
    "bulk-load": "cc.mallet.util.BulkLoader",
}
