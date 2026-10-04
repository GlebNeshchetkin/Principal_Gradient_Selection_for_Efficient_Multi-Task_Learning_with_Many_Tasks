import copy
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from methods.min_norm_solvers import MinNormSolver, gradient_normalizers
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans

EPS = 1e-8 # for numerical stability

def select_gradients_kmeans(grads, keep_ratio=0.25, n_components=64):
    """
    Principal Gradient Selection (PGS).

    Args:
        grads: Tensor [num_params, n_tasks].
        keep_ratio: Fraction of gradients to retain.
        n_components: Number of randomly sampled parameter dimensions.

    Returns:
        selected_grads: Tensor [n_keep, num_params].
        selected_indices: NumPy array of selected task indices.
        cluster_labels: NumPy array of KMeans cluster assignments.
    """

    grads_t = grads.t()  # [n_tasks, num_params]
    n_tasks = grads_t.shape[0]

    if n_tasks <= 2:
        return grads_t, np.arange(n_tasks), None

    # 1. Determine the number of gradients to retain
    n_keep = max(2, int(n_tasks * keep_ratio))

    # 2. Compute full gradient magnitudes
    norms = torch.norm(grads_t, dim=1)

    # 3. Randomly sample parameter dimensions
    d = min(n_components, grads_t.shape[1])

    idx = torch.randperm(
        grads_t.shape[1],
        device=grads.device
    )[:d]

    sampled_grads = grads_t[:, idx]  # [n_tasks, d]

    # 4. Normalize reduced gradients
    sampled_norms = torch.norm(sampled_grads, dim=1, keepdim=True)

    directions = sampled_grads / (sampled_norms + 1e-8)

    directions_np = directions.detach().cpu().numpy()

    # 5. Cluster normalized reduced gradients
    n_clusters = max(2, min(10, n_tasks // 4, n_keep))

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=3
    )

    cluster_labels = kmeans.fit_predict(directions_np)

    # Transfer only the task magnitudes to CPU
    magnitudes = norms.detach().cpu().numpy()

    # 6. Allocate selected gradients proportionally to cluster sizes
    cluster_indices = [
        np.where(cluster_labels == k)[0]
        for k in range(n_clusters)
    ]

    cluster_sizes = np.array(
        [len(indices) for indices in cluster_indices]
    )

    # Ideal proportional allocations
    quotas = n_keep * cluster_sizes / n_tasks

    # Initial integer allocations
    allocations = np.floor(quotas).astype(int)

    # Distribute the remaining positions using largest remainders
    remaining = n_keep - allocations.sum()

    fractional_parts = quotas - allocations

    order = np.argsort(-fractional_parts)

    for k in order[:remaining]:
        allocations[k] += 1

    # 7. Select gradients with largest magnitudes within each cluster
    selected_indices = []

    for k, indices in enumerate(cluster_indices):

        n_cluster_keep = allocations[k]

        if n_cluster_keep == 0:
            continue

        cluster_magnitudes = magnitudes[indices]

        top_local = np.argsort(-cluster_magnitudes)[:n_cluster_keep]

        selected_indices.extend(indices[top_local])

    # 8. Correct the number of selected gradients if necessary
    selected_set = set(selected_indices)

    if len(selected_indices) > n_keep:

        selected_indices.sort(
            key=lambda i: magnitudes[i],
            reverse=True
        )

        selected_indices = selected_indices[:n_keep]

    elif len(selected_indices) < n_keep:

        remaining_indices = [
            i for i in range(n_tasks)
            if i not in selected_set
        ]

        remaining_indices.sort(
            key=lambda i: magnitudes[i],
            reverse=True
        )

        selected_indices.extend(
            remaining_indices[:n_keep - len(selected_indices)]
        )

    selected_indices = np.array(
        sorted(selected_indices),
        dtype=int
    )

    # 9. Construct selected gradient matrix
    selected_grads = grads_t[selected_indices]

    return selected_grads, selected_indices, cluster_labels

class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device, max_norm = 1.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []

class LinearScalarization(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(losses * self.task_weights)
        return loss, dict(weights=self.task_weights)
    
class LinearScalarization_GS(WeightMethod):
    """Linear scalarization with gradient selection."""

    def __init__(self, n_tasks, device, task_weights=None, keep_ratio=0.5, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones(n_tasks)
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)
        self.keep_ratio = keep_ratio
        self.max_norm = max_norm

    def get_weighted_loss(self, losses, shared_parameters, **kwargs):
        grad_dims = [p.data.numel() for p in shared_parameters]
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            losses[i].backward(retain_graph=True)
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            for p in shared_parameters:
                p.grad = None

        selected_grads, selected_indices, _ = select_gradients_kmeans(
            grads, keep_ratio=self.keep_ratio
        )
        new_grad = selected_grads.t().mean(dim=1)
        self.overwrite_grad(shared_parameters, new_grad, grad_dims)

        return None, {
            "weights": self.task_weights,
            "selected_indices": selected_indices,
        }

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        grads[:, task].fill_(0.0)
        cnt = 0
        for param in shared_params:
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg:en, task].copy_(param.grad.data.detach().view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks
        cnt = 0
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            param.grad = newgrad[beg:en].contiguous().view(param.data.size()).clone()
            cnt += 1

    def backward(self, losses, parameters=None, shared_parameters=None,
                 task_specific_parameters=None, **kwargs):
        _, info = self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, info

class PCGrad_GS(WeightMethod):
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        reduction="sum",
        keep_ratio=0.5,
    ):
        super().__init__(n_tasks, device=device)
        assert reduction in ["mean", "sum"]
        self.reduction = reduction
        self.keep_ratio = keep_ratio

    def get_weighted_loss(
        self,
        losses,
        shared_parameters=None,
        task_specific_parameters=None,
        **kwargs,
    ):
        raise NotImplementedError

    def _set_pc_grads(
        self,
        losses,
        shared_parameters,
        task_specific_parameters=None,
    ):
        # Compute task gradients
        shared_grads = [
            torch.autograd.grad(
                loss,
                shared_parameters,
                retain_graph=True,
            )
            for loss in losses
        ]

        if isinstance(shared_parameters, torch.Tensor):
            shared_parameters = [shared_parameters]

        # ---------------------------------------------------------
        # Gradient selection
        # ---------------------------------------------------------
        flat_grads = torch.stack([
            torch.cat([g.flatten() for g in task_grad])
            for task_grad in shared_grads
        ]).T

        _, selected_indices, _ = select_gradients_kmeans(
            flat_grads,
            keep_ratio=self.keep_ratio,
        )

        # Select original gradient tuples
        selected_grads = [
            shared_grads[i]
            for i in selected_indices
        ]

        # ---------------------------------------------------------
        # PCGrad
        # ---------------------------------------------------------
        non_conflict_shared_grads = self._project_conflicting(
            selected_grads
        )

        for p, g in zip(
            shared_parameters,
            non_conflict_shared_grads,
        ):
            p.grad = g

        # ---------------------------------------------------------
        # Task-specific parameters
        # ---------------------------------------------------------
        if task_specific_parameters is not None:
            task_specific_grads = torch.autograd.grad(
                losses.sum(),
                task_specific_parameters,
            )

            if isinstance(task_specific_parameters, torch.Tensor):
                task_specific_parameters = [task_specific_parameters]

            for p, g in zip(
                task_specific_parameters,
                task_specific_grads,
            ):
                p.grad = g

    def _project_conflicting(self, grads):
        pc_grad = copy.deepcopy(grads)

        for g_i in pc_grad:
            shuffled_grads = grads.copy()
            random.shuffle(shuffled_grads)

            for g_j in shuffled_grads:
                g_i_g_j = sum(
                    torch.dot(
                        grad_i.flatten(),
                        grad_j.flatten(),
                    )
                    for grad_i, grad_j in zip(g_i, g_j)
                )

                if g_i_g_j < 0:
                    g_j_norm_square = (
                        torch.norm(
                            torch.cat([
                                g.flatten()
                                for g in g_j
                            ])
                        ) ** 2
                    )

                    for grad_i, grad_j in zip(g_i, g_j):
                        grad_i -= (
                            g_i_g_j
                            * grad_j
                            / (g_j_norm_square + 1e-8)
                        )

        merged_grad = [
            sum(g)
            for g in zip(*pc_grad)
        ]

        if self.reduction == "mean":
            merged_grad = [
                g / len(grads)
                for g in merged_grad
            ]

        return merged_grad

    def backward(
        self,
        losses,
        parameters=None,
        shared_parameters=None,
        task_specific_parameters=None,
        **kwargs,
    ):
        self._set_pc_grads(
            losses,
            shared_parameters,
            task_specific_parameters,
        )

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                shared_parameters,
                self.max_norm,
            )

        return None, {}
    
class PCGrad(WeightMethod):
    """Modification of: https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py

    @misc{Pytorch-PCGrad,
      author = {Wei-Cheng Tseng},
      title = {WeiChengTseng/Pytorch-PCGrad},
      url = {https://github.com/WeiChengTseng/Pytorch-PCGrad.git},
      year = {2020}
    }

    """

    def __init__(self, n_tasks: int, device: torch.device, reduction="sum"):
        super().__init__(n_tasks, device=device)
        assert reduction in ["mean", "sum"]
        self.reduction = reduction

    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def _set_pc_grads(self, losses, shared_parameters, task_specific_parameters=None):
        # shared part
        shared_grads = []
        for l in losses:
            shared_grads.append(
                torch.autograd.grad(l, shared_parameters, retain_graph=True)
            )

        if isinstance(shared_parameters, torch.Tensor):
            shared_parameters = [shared_parameters]
        non_conflict_shared_grads = self._project_conflicting(shared_grads)
        for p, g in zip(shared_parameters, non_conflict_shared_grads):
            p.grad = g

        # task specific part
        if task_specific_parameters is not None:
            task_specific_grads = torch.autograd.grad(
                losses.sum(), task_specific_parameters
            )
            if isinstance(task_specific_parameters, torch.Tensor):
                task_specific_parameters = [task_specific_parameters]
            for p, g in zip(task_specific_parameters, task_specific_grads):
                p.grad = g

    def _project_conflicting(self, grads: List[Tuple[torch.Tensor]]):
        pc_grad = copy.deepcopy(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = sum(
                    [
                        torch.dot(torch.flatten(grad_i), torch.flatten(grad_j))
                        for grad_i, grad_j in zip(g_i, g_j)
                    ]
                )
                if g_i_g_j < 0:
                    g_j_norm_square = (
                        torch.norm(torch.cat([torch.flatten(g) for g in g_j])) ** 2
                    )
                    for grad_i, grad_j in zip(g_i, g_j):
                        grad_i -= g_i_g_j * grad_j / g_j_norm_square

        merged_grad = [sum(g) for g in zip(*pc_grad)]
        if self.reduction == "mean":
            merged_grad = [g / self.n_tasks for g in merged_grad]

        return merged_grad

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        self._set_pc_grads(losses, shared_parameters, task_specific_parameters)
        # make sure the solution for shared params has norm <= self.eps
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {}  # NOTE: to align with all other weight methods

class CAGrad_GS(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, c=0.4, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        self.c = c
        self.max_norm = max_norm

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g, GTG, w_cpu = self.cagrad(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(shared_parameters, g, grad_dims)
        return GTG, w_cpu

    def cagrad(self, grads, alpha=0.5, rescale=1):
        GG_real = grads.t().mm(grads).cpu()  # [40, 40]
        initial_grads_saved = grads.t().detach().cpu().numpy()  # [num_tasks, dim]
        selected_grads, selected_indices, _ = select_gradients_kmeans(grads, keep_ratio=0.5)
        selected_grads_saved = selected_grads.t().detach().cpu().numpy()  # [num_tasks, dim]
        
        save_dict = {
            'initial_grads': initial_grads_saved,
            'selected_grads': selected_grads_saved,
            'selected_indices': torch.tensor(selected_indices),
            'GG_real': torch.tensor(GG_real)
        }
        
        # save_dict = {
        #     'initial_grads': initial_grads_saved,
        #     'selected_grads': selected_grads_saved,
        #     'selected_indices': torch.tensor(selected_indices), # Keep if it's a regular Python list
        #     'GG_real': GG_real.clone().detach() # Fixed line
        # }
                
        original_n_tasks = self.n_tasks
        
        n_selected = len(selected_indices)
    
        # print(f"Original n_tasks: {self.n_tasks}, Selected: {n_selected}")
        
        # print(f"selected_grads.shape = : {selected_grads.shape}")
        
        selected_grads = selected_grads.t()  # [5.17M, 18]
        original_n_tasks = self.n_tasks
        self.n_tasks = n_selected
        GG = selected_grads.t().mm(selected_grads).cpu()  # [18, 18]
        # print(f"GG.shape: {GG.shape}")  # Должно быть [18, 18]
        
    
        
        # self.n_tasks = len(selected_indices)
        
        # GG = selected_grads.t().mm(selected_grads).cpu()
        g0_norm = (GG.mean() + 1e-8).sqrt()
        
        x_start = np.ones(self.n_tasks) / self.n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()
        
        def objfn(x):
            return (
                x.reshape(1, self.n_tasks).dot(A).dot(b.reshape(self.n_tasks, 1))
                + c * np.sqrt(
                    x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1)) + 1e-8
                )
            ).sum()
        
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        
        ww = torch.Tensor(w_cpu).to(selected_grads.device)
        gw = (selected_grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = selected_grads.mean(1) + lmbda * gw
        
        self.n_tasks = original_n_tasks
        
        # print(f"[END] g.shape = ", g.shape)
        
        if rescale == 0:
            return g, GG_real.numpy(), w_cpu
        elif rescale == 1:
            return g / (1 + alpha ** 2), GG_real.numpy(), w_cpu
        else:
            return g / (1 + alpha), GG_real.numpy(), w_cpu
    
    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        GTG, w = self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {"GTG": GTG, "weights": w}  # NOTE: to align with all other weight methods

class CAGrad(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, c=0.4, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        self.c = c
        self.max_norm = max_norm

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g, GTG, w_cpu = self.cagrad(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(shared_parameters, g, grad_dims)
        return GTG, w_cpu

    def cagrad(self, grads, alpha=0.5, rescale=1):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.n_tasks) / self.n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                x.reshape(1, self.n_tasks).dot(A).dot(b.reshape(self.n_tasks, 1))
                + c
                * np.sqrt(
                    x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1))
                    + 1e-8
                )
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g, GG.numpy(), w_cpu
        elif rescale == 1:
            return g / (1 + alpha ** 2), GG.numpy(), w_cpu
        else:
            return g / (1 + alpha), GG.numpy(), w_cpu

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        GTG, w = self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {"GTG": GTG, "weights": w}  # NOTE: to align with all other weight methods

class WeightMethods:
    def __init__(self, method: str, n_tasks: int, device: torch.device, **kwargs):
        """
        :param method:
        """
        assert method in list(METHODS.keys()), f"unknown method {method}."

        self.method = METHODS[method](n_tasks=n_tasks, device=device, **kwargs)

    def get_weighted_loss(self, losses, **kwargs):
        return self.method.get_weighted_loss(losses, **kwargs)

    def backward(
        self, losses, **kwargs
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        return self.method.backward(losses, **kwargs)

    def __ceil__(self, losses, **kwargs):
        return self.backward(losses, **kwargs)

    def parameters(self):
        return self.method.parameters()


METHODS = dict(
    ls=LinearScalarization,
    ls_gs=LinearScalarization_GS,
    pcgrad=PCGrad,
    pcgrad_gs=PCGrad_GS,
    cagrad=CAGrad,
    cagrad_gs=CAGrad_GS,
)
