import copy
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from sklearn.cluster import SpectralClustering

import src.client as attacker
from src import defense, recover
from src.aggregator import average_weights
from src.client import Client
from src.utils import setup_logger
from src.utils.helper import evaluate_model

from .fedavg import FedAvg


logger = setup_logger()


@dataclass
class GroupState:
    """All training and recovery state owned by one isolated client group."""

    group_id: int
    client_ids: List[int]
    global_model: torch.nn.Module
    old_global_models: List[dict] = field(default_factory=list)
    old_client_models: List[List[dict]] = field(default_factory=list)
    privacy_costs: List[float] = field(default_factory=list)
    remaining_budgets: List[float] = field(default_factory=list)
    remaining_budgets_per_client: List[List[float]] = field(default_factory=list)
    select_info: List[List[int]] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    aggr_clients: List[List[int]] = field(default_factory=list)


class GroupedFedAvg(FedAvg):
    """DIARY with one-round similarity pretraining and isolated group training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_groups = self.config['FL'].get('num_groups', 1)
        num_clients = self.config['FL']['num_clients']

        if not 1 <= self.num_groups <= num_clients:
            raise ValueError(
                f"FL.num_groups must be between 1 and {num_clients}, got {self.num_groups}"
            )
        if self.config['FL']['with_defense'] and num_clients // self.num_groups < 2:
            raise ValueError(
                "Each group needs at least two clients when defense is enabled. "
                "Reduce FL.num_groups or increase FL.num_clients."
            )

        self.initial_global_model = copy.deepcopy(self.global_model)
        self.client_groups: List[List[int]] = []
        self.group_states: List[GroupState] = []

    def _make_pretrain_client(self, client_id: int) -> Client:
        """Create a temporary, non-private client used only to obtain a grouping feature."""
        list_dps, num_dps = self.client_data_dict[client_id]
        model = copy.deepcopy(self.initial_global_model)

        if client_id in self.adversary_list:
            return getattr(attacker, self.config['Attack']['name'])(
                client_id,
                model,
                self.train_dataset,
                list_dps,
                num_dps,
                adversary_list=self.adversary_list,
                **self.config['Attack']['args'],
            )

        return Client(
            client_id,
            model,
            self.train_dataset,
            list_dps,
            num_dps,
            **self.config['Trainer'],
        )

    @staticmethod
    def _model_update_vector(local_model, initial_model) -> torch.Tensor:
        updates = []
        for local_param, initial_param in zip(local_model.parameters(), initial_model.parameters()):
            updates.append(
                (local_param.detach().cpu() - initial_param.detach().cpu()).reshape(-1)
            )
        return torch.cat(updates)

    def _pretrain_updates(self) -> torch.Tensor:
        """Run one temporary local epoch sequence for every client and return update vectors."""
        update_vectors = []
        logger.info("_____ Grouping pretraining round _____")

        for client_id in range(self.config['FL']['num_clients']):
            client = self._make_pretrain_client(client_id)
            client.receive_global_model(self.initial_global_model)
            result = client.local_train()
            local_model = client.model if result is None else result[0]
            update_vectors.append(
                self._model_update_vector(local_model, self.initial_global_model)
            )
            del client

        return torch.stack(update_vectors)

    @staticmethod
    def _balance_groups(labels: np.ndarray, affinity: np.ndarray, num_groups: int) -> List[List[int]]:
        """Balance spectral clusters while moving the least-affiliated clients first."""
        num_clients = len(labels)
        target_sizes = [num_clients // num_groups] * num_groups
        for group_id in range(num_clients % num_groups):
            target_sizes[group_id] += 1

        groups = [np.where(labels == group_id)[0].tolist() for group_id in range(num_groups)]

        while any(len(groups[group_id]) > target_sizes[group_id] for group_id in range(num_groups)):
            source = max(
                (group_id for group_id in range(num_groups)
                 if len(groups[group_id]) > target_sizes[group_id]),
                key=lambda group_id: len(groups[group_id]) - target_sizes[group_id],
            )
            targets = [
                group_id for group_id in range(num_groups)
                if len(groups[group_id]) < target_sizes[group_id]
            ]

            best_move = None
            best_score = -float("inf")
            for client_id in groups[source]:
                source_peers = [peer for peer in groups[source] if peer != client_id]
                source_similarity = (
                    float(np.mean(affinity[client_id, source_peers])) if source_peers else 0.0
                )
                for target in targets:
                    target_similarity = (
                        float(np.mean(affinity[client_id, groups[target]]))
                        if groups[target] else 0.0
                    )
                    score = target_similarity - source_similarity
                    if score > best_score:
                        best_score = score
                        best_move = (client_id, target)

            client_id, target = best_move
            groups[source].remove(client_id)
            groups[target].append(client_id)

        return [sorted(group) for group in groups]

    def _make_groups(self) -> List[List[int]]:
        if self.num_groups == 1:
            return [list(range(self.config['FL']['num_clients']))]

        update_vectors = self._pretrain_updates()
        norms = torch.linalg.vector_norm(update_vectors, dim=1, keepdim=True).clamp_min(1e-12)
        normalized_updates = update_vectors / norms
        cosine_similarity = (normalized_updates @ normalized_updates.T).numpy()

        # SpectralClustering requires a non-negative affinity matrix.
        affinity = np.clip((cosine_similarity + 1.0) / 2.0, 0.0, 1.0)
        np.fill_diagonal(affinity, 1.0)

        labels = SpectralClustering(
            n_clusters=self.num_groups,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=self.config.get('seed', 0),
        ).fit_predict(affinity)

        groups = self._balance_groups(labels, affinity, self.num_groups)
        logger.info(f'Client groups: {groups}')
        return groups

    def _new_group_state(self, group_id: int, client_ids: List[int]) -> GroupState:
        state = GroupState(
            group_id=group_id,
            client_ids=client_ids,
            global_model=copy.deepcopy(self.initial_global_model),
        )
        state.remaining_budgets.append(
            sum(self.clients_pool[client_id].remaining_budget for client_id in client_ids)
        )
        return state

    def _select_group_clients(self, client_ids: List[int]) -> List[int]:
        select_num = int(self.selector.client_rate * len(client_ids))
        return np.random.choice(client_ids, select_num, replace=False).tolist()

    def _train_group(self, state: GroupState):
        main_accuracies = []
        attack_accuracies = []

        for rd in range(self.config['FL']['rounds']):
            state.old_global_models.append(copy.deepcopy(state.global_model.state_dict()))
            local_models = []
            local_losses = []
            privacy_cost = 0

            logger.info(
                "----- Group {:2d} Round {:3d} -----".format(state.group_id, rd)
            )
            selected_clients = self._select_group_clients(state.client_ids)
            logger.info(f'group {state.group_id} selected clients:{selected_clients}')
            state.select_info.append(selected_clients)

            old_client_models = []
            remaining_budgets = []
            for client_id in selected_clients:
                client = self.clients_pool[client_id]
                client.receive_global_model(state.global_model)
                local_model, local_loss = client.local_train()
                local_models.append(local_model)
                local_losses.append(local_loss)
                privacy_cost += client.privacy_cost_this_round
                old_client_models.append(copy.deepcopy(local_model.state_dict()))
                remaining_budgets.append(client.remaining_budget)

            state.old_client_models.append(old_client_models)
            state.privacy_costs.append(privacy_cost)
            state.remaining_budgets.append(
                state.remaining_budgets[0] - sum(state.privacy_costs)
            )
            state.remaining_budgets_per_client.append(remaining_budgets)

            if self.config['FL']['with_DP']:
                logger.info(f'group {state.group_id} remaining_budget: {remaining_budgets}')

            if self.config['FL']['with_defense']:
                defender = getattr(defense, self.config['Defense']['name'])(
                    **self.config['Defense']['args'],
                    adversary_list=self.adversary_list,
                )
                state.global_model = defender.exec(
                    state.global_model,
                    local_models,
                    selected_clients,
                )
                state.aggr_clients.append(defender.benign_clients)
            else:
                state.global_model = average_weights(state.global_model, local_models)
                state.aggr_clients.append(selected_clients)

            if rd == self.config['FL']['rounds'] - 1:
                state.old_global_models.append(copy.deepcopy(state.global_model.state_dict()))

            round_loss = sum(local_losses) / len(local_losses)
            state.train_losses.append(round_loss)
            logger.info(f'Group {state.group_id} training average loss: {round_loss:.3f}')

            test_accuracy, test_loss = evaluate_model(
                dataset=self.test_dataset,
                model=state.global_model,
                loss_function=self.config['Trainer']['loss_function'],
                device=self.config['Trainer']['device'],
            )
            logger.info(
                f'Group {state.group_id} testing accuracy: '
                f'{test_accuracy:.2f}%, loss: {test_loss:.3f}'
            )
            main_accuracies.append(round(test_accuracy.item(), 2))

            if self.config['FL']['with_attack']:
                old_global_model = self.global_model
                self.global_model = state.global_model
                attack_accuracy = self.eval_attack()
                self.global_model = old_global_model
                logger.info(
                    f'Group {state.group_id} attack accuracy: {attack_accuracy:.2f}%'
                )
                attack_accuracies.append(round(attack_accuracy.item(), 2))

        logger.debug(f'Group {state.group_id} Main Accuracy:{main_accuracies}')
        logger.debug(f'Group {state.group_id} Attacker Accuracy:{attack_accuracies}')
        logger.debug(f'Group {state.group_id} Privacy Cost:{state.privacy_costs}')
        logger.debug(f'Group {state.group_id} Remaining Budget:{state.remaining_budgets}')

    @torch.no_grad()
    def _aggregate_group_models(self):
        """One-shot weighted aggregation of the independently trained group models."""
        total_clients = sum(len(state.client_ids) for state in self.group_states)
        result = copy.deepcopy(self.initial_global_model)
        result_params = list(result.parameters())
        for result_param in result_params:
            result_param.zero_()

        for state in self.group_states:
            weight = len(state.client_ids) / total_clients
            for result_param, group_param in zip(result_params, state.global_model.parameters()):
                result_param.add_(group_param.detach(), alpha=weight)

        self.global_model = result

    def fl_train(self):
        self.client_groups = self._make_groups()
        self.group_states = [
            self._new_group_state(group_id, client_ids)
            for group_id, client_ids in enumerate(self.client_groups)
        ]

        for state in self.group_states:
            self._train_group(state)

        self._aggregate_group_models()
        test_accuracy, test_loss = evaluate_model(
            dataset=self.test_dataset,
            model=self.global_model,
            loss_function=self.config['Trainer']['loss_function'],
            device=self.config['Trainer']['device'],
        )
        logger.info(
            f'Grouped global model testing accuracy: {test_accuracy:.2f}%, '
            f'loss: {test_loss:.3f}'
        )

    def _recover_group(self, state: GroupState):
        group_malicious_clients = [
            client_id for client_id in self.malicious_clients
            if client_id in state.client_ids
        ]
        if not group_malicious_clients:
            logger.info(
                f'Group {state.group_id} has no malicious client; recovery is skipped.'
            )
            return

        logger.info(f'________Begin Recover Group {state.group_id}________')
        recover_server = getattr(recover, self.config['Recover']['name'])(
            test_dataset=self.test_dataset,
            global_model=state.global_model,
            clients_pool=self.clients_pool,
            old_global_models=state.old_global_models,
            old_client_models=state.old_client_models,
            old_rounds=self.config['FL']['rounds'],
            select_info=state.select_info,
            malicious_clients=group_malicious_clients,
            recover_config=self.config['Recover']['args'],
            loss_function=self.config['Trainer']['loss_function'],
            train_losses=state.train_losses,
            privacy_costs=state.privacy_costs,
            remaining_budgets=state.remaining_budgets,
            remaining_budgets_per_client=state.remaining_budgets_per_client,
            aggr_clients=state.aggr_clients,
            dp_config=self.config['Fed_dp'],
            attack_config=self.config['Attack'],
        )
        state.global_model = recover_server.recover()

    def fl_recover(self):
        if not self.config['FL']['with_recover']:
            return

        for state in self.group_states:
            self._recover_group(state)

        self._aggregate_group_models()
        test_accuracy, test_loss = evaluate_model(
            dataset=self.test_dataset,
            model=self.global_model,
            loss_function=self.config['Trainer']['loss_function'],
            device=self.config['Trainer']['device'],
        )
        logger.info(
            f'Recovered grouped global model testing accuracy: {test_accuracy:.2f}%, '
            f'loss: {test_loss:.3f}'
        )

        if self.config['FL']['with_attack']:
            attack_accuracy = self.eval_attack()
            logger.info(f'Attack accuracy: {attack_accuracy:.2f}%')
