from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from detectron2.config import CfgNode, configurable
from detectron2.layers import ShapeSpec
import torch
import torch.nn

from srnet.utils._utils import find_cfg_node

from ..common.types import Losses
from .unsupervised_head import (
    UNSUPERVISED_HEAD_REGISTRY,
    UnsupervisedHead,
    UnsupervisedOutput,
)

__all__ = ["UnsupervisedMultiObjective"]


@UNSUPERVISED_HEAD_REGISTRY.register()
class UnsupervisedMultiObjective(UnsupervisedHead):
    """
    An unsupervised head that contains multiple other unsupervised objective
    heads evaluated in parallel.
    """

    @configurable
    def __init__(self, unsupervised_heads: Mapping[str, UnsupervisedHead]):
        super(UnsupervisedHead, self).__init__()

        # ensure all of the objective's output & loss keys are unique, because
        # they will each be aggregated into one pair of dicts as a result of
        # forward()
        head_output_keys: Set[str] = set()
        head_loss_keys: Set[str] = set()
        for head_name, head in unsupervised_heads.items():
            for lk in head.loss_keys:
                if lk in head_loss_keys:
                    raise ValueError(
                        f'found non-unique head loss key "{lk}" for unsupervised head "{head_name}"'
                    )
                head_loss_keys.add(lk)
            for ok in head.output_keys:
                if ok in head_output_keys:
                    raise ValueError(
                        f'found non-unique head output key "{ok}" for unsupervised head "{head_name}"'
                    )
                head_output_keys.add(ok)

        self.unsupervised_heads: Union[
            torch.nn.ModuleDict, Mapping[str, UnsupervisedHead]
        ]
        self.unsupervised_heads = torch.nn.ModuleDict(unsupervised_heads)

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        cfg: CfgNode,
        input_shape: Dict[str, ShapeSpec],
        node_path: str = "MODEL.UNSUPERVISED_MULTI_OBJECTIVE",
    ) -> Dict[str, Any]:
        target_node = find_cfg_node(cfg, node_path)
        heads: Dict[str, UnsupervisedHead] = {}
        for head_name, objective_type, head_cfg_path in target_node.OBJECTIVES_LIST:
            head_module = UNSUPERVISED_HEAD_REGISTRY.get(objective_type)
            heads[head_name] = head_module(cfg, input_shape, node_path=head_cfg_path)
        return {"unsupervised_heads": heads}

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[UnsupervisedOutput, Losses]:
        head_outputs: Dict[str, Tuple[UnsupervisedOutput, Losses]] = {
            head_name: head(features, targets)
            for head_name, head in self.unsupervised_heads.items()
        }
        results: UnsupervisedOutput = {}
        losses: Losses = {}
        for head_name, (head_output, _) in head_outputs.items():
            for k in head_output.keys():
                assert (
                    k not in results
                ), f'got non-unique result key "{k}" from head {head_name}'
            results.update(head_output)
        if self.training:
            for head_name, (_, head_losses) in head_outputs.items():
                for k in head_losses.keys():
                    assert (
                        k not in losses
                    ), f'got non-unique loss key "{k}" from head {head_name}'
                losses.update(head_losses)
        return results, losses

    def into_per_item_iterable(
        self, network_output: UnsupervisedOutput
    ) -> List[Dict[str, torch.Tensor]]:
        outputs: List[Dict[str, torch.Tensor]] = []
        for head in self.unsupervised_heads.values():
            head_outputs: List[Dict[str, torch.Tensor]]
            head_outputs = head.into_per_item_iterable(network_output)
            # now we have to do a little extra work to merge the returned dicts
            # "index-wise", to get the right return type
            n_outputs = len(head_outputs)
            if len(outputs) < n_outputs:
                # extend the output list if necessary
                outputs = outputs + [{} for _ in range(len(outputs), n_outputs)]
            for i, ho in enumerate(head_outputs):
                outputs[i].update(ho)
        return outputs

    def postprocess(
        self, result: Mapping[str, Optional[torch.Tensor]], *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        for head in self.unsupervised_heads.values():
            head_outputs: Dict[str, Any]
            head_outputs = head.postprocess(result, *args, **kwargs)
            outputs.update(head_outputs)
        return outputs

    @property
    def loss_keys(self) -> Set[str]:
        s: Set[str] = set.union(
            *(h.loss_keys for h in self.unsupervised_heads.values())
        )
        return s

    @property
    def output_keys(self) -> Set[str]:
        s: Set[str] = set.union(
            *(h.output_keys for h in self.unsupervised_heads.values())
        )
        return s
