import numpy as np
from abc import ABC, abstractmethod


class BaseConflictDetector(ABC):
    """
    Abstract base class for all three knowledge conflict detection methods.
    Enforces a consistent interface across methods for data processing, 
    training, prediction, and evaluation.
    """

    @abstractmethod
    def extract_features(self, dataset):
        """
        Extract features or representations from the dataset.
        
        Args:
            dataset: The input dataset (e.g., list of dicts, or huggingface Dataset)
            
        Returns:
            np.ndarray or list: Extracted features ready for training/prediction.
        """
        pass

    @abstractmethod
    def train(self, train_data, train_labels):
        """
        Train the conflict detector. For probing this is logistic regression, 
        for NLI it's finetuning an encoder, and for logits it might just be 
        threshold tuning.
        
        Args:
            train_data: The input features/data for training.
            train_labels: The binary labels (1=conflict, 0=non-conflict).
        """
        pass

    @abstractmethod
    def predict(self, test_data) -> np.ndarray:
        """
        Return binary predictions for the given data.
        
        Args:
            test_data: The input features/data for prediction.
            
        Returns:
            np.ndarray: Binary predictions (0 or 1).
        """
        pass

    @abstractmethod
    def predict_proba(self, test_data) -> np.ndarray:
        """
        Return conflict probability scores for the given data.
        
        Args:
            test_data: The input features/data for prediction.
            
        Returns:
            np.ndarray: Probability scores between 0.0 and 1.0.
        """
        pass

    def evaluate(self, test_data, test_labels):
        """
        Run predictions and compute metrics using the shared metric utility.
        
        Args:
            test_data: The input features/data for prediction.
            test_labels: The ground truth binary labels.
            
        Returns:
            dict: Dictionary of evaluation metrics.
        """
        from evaluation.metrics import compute_metrics
        
        # We need to make sure we have both probabilities and hard predictions
        try:
            probas = self.predict_proba(test_data)
        except NotImplementedError:
            probas = None
            
        preds = self.predict(test_data)
        
        return compute_metrics(y_true=test_labels, y_pred=preds, y_proba=(probas[:, 1] if probas is not None and len(probas.shape) > 1 else probas))
