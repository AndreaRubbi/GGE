from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    """
    Abstract base class for evaluators in the gene expression evaluation system.
    """

    def __init__(self, data, output):
        self.data = data
        self.output = output

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        Evaluate the model performance based on the provided data and output.
        This method should be implemented by subclasses.
        """
        pass


class GeneExpressionEvaluator(BaseEvaluator):
    """
    Evaluator for gene expression data.

    Computes various metrics between real and generated gene expression profiles,
    optionally adjusting for control conditions and covariates.

    Parameters
    ----------
    data : GeneExpressionDataModule
        The data module containing gene expression datasets.
    output : AnnData
        The generated gene expression data to evaluate.
    """

    def __init__(self, data, output):
        super().__init__(data, output)

    def evaluate(self, delta=False, plot=False, DEG=None):
        # Implementation of the evaluation logic will go here
        pass