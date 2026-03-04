class BaseModel:
    """
    Base class for all models in the gene expression evaluation system.

    This class provides a foundation for model classes that may be implemented in the future.
    It can include common methods and attributes that all models should have.
    """

    def __init__(self):
        pass

    def fit(self, data):
        """
        Fit the model to the provided data.

        Parameters
        ----------
        data : Any
            The data to fit the model on.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, data):
        """
        Make predictions using the fitted model.

        Parameters
        ----------
        data : Any
            The data to make predictions on.

        Returns
        -------
        Any
            The predictions made by the model.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def evaluate(self, data):
        """
        Evaluate the model's performance on the provided data.

        Parameters
        ----------
        data : Any
            The data to evaluate the model on.

        Returns
        -------
        Any
            The evaluation metrics.
        """
        raise NotImplementedError("Subclasses should implement this method.")