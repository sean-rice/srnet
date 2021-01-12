from typing import Any, Callable, Dict, List, Tuple

import torch


def preprocess_batch_order(
    batched_inputs: List[Dict[str, Any]],
    has_supervision: Callable[[Dict[str, Any]], int],
) -> Tuple[List[Dict[str, Any]], int, int, List[int]]:
    """
        Preprocesses an input batch's list order such that the unlabeled
        examples-- those that return `0` from `has_supervision`-- come before
        the labeled ones. This may be important in some models once the batch
        becomes a single `Tensor` and, for example, the items within the batch
        need to be routed to different components of the model without breaking
        up the original :class:`Tensor` or making a copy.

        Args:
            batched_inputs (list): A list of dicts from a
                :class:`DatasetMapper` for the model to use as input, each item
                of which may or may not contain a supervised label.
            has_supervision (callable): A callable that takes a single input
                dict from the batch and returns whether or not the example
                should be considered to be supervised; 0 for False (no
                label), 1 for True (label available).
        
        Returns:
            batched_inputs (list): The re-ordered inputs. This is a new list
                containing references to the original items in the input list.
            n_unlabeled (int): The number of unlabeled items from the input
                list.
            n_labeled (int): The number of labeled items from the input list.
            sorted_indices (List[int]): The original index in the input of
                each item in the returned list. This allows for undoing the
                sort (putting items back in their original order) at a later
                stage.
        """
    with torch.no_grad():
        has_instances = torch.tensor(
            [has_supervision(inp) for inp in batched_inputs],
            dtype=torch.int,
            device=torch.device("cpu"),
        )
        sorted_indices = [int(i.item()) for i in torch.argsort(has_instances)]
        n_labeled = int(torch.sum(has_instances).item())
        n_unlabeled = len(batched_inputs) - n_labeled
        batched_inputs = [batched_inputs[i] for i in sorted_indices]
    return batched_inputs, n_unlabeled, n_labeled, sorted_indices
