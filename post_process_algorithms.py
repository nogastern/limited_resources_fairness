from prediction import predict, predict_classification
from fairlearn.metrics import MetricFrame
import numpy as np
import pandas as pd
import my_constants
import plot_graphs
import copy
from abc import ABC
from performance_metrics import sensitivity_score
import math
import time

def get_algo(strategy, percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric,
                          filename_prefix, num_of_iters=None, bucket_size=0.01):
    if strategy == my_constants.BY_NUM_OF_ITERS:
        algo = PostProcessIterNum(num_of_iters, percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix)
    elif strategy == my_constants.CHECK_CONVERGENCE:
        algo = PostProcessUntilConvergence(percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix)
    elif strategy == my_constants.BRUTE_FORCE:
        algo = PostProcessBruteForce(percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix)
    elif strategy == my_constants.BUCKETS:
        algo = PostProcessBuckets(percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix, bucket_size)
    elif strategy == my_constants.BINARY_SEARCH:
        algo = PostProcessBinarySearch(percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix)
    else:
        raise Exception("Strategy not supported")
    return algo


def get_cutoffs_per_group(strategy, percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric,
                          filename_prefix, num_of_iters=None, bucket_size=0.01):
    algo = get_algo(strategy, percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric,
                          filename_prefix, num_of_iters, bucket_size)
    return algo.cutoff_for_groups()


class PostProcessAlgo(ABC):

    def __init__(self, percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix):
        self.percent_cutoff = percent_cutoff
        self.X_train = X_train
        self.y_train = y_train
        self.y_pred_train = y_pred_train
        self.sensitive_feature = sensitive_feature
        self.metric = metric
        self.filename_prefix = filename_prefix
        self.X_y_pred = None

    def cutoff_for_groups(self):
        cutoffs_for_groups = {}
        binary_y_pred, cutoff = predict_classification(self.y_pred_train, self.percent_cutoff)
        my_metrics = {
            'Sensitivity': sensitivity_score,
        }
        self.X_y_pred = self.X_train.copy()
        self.X_y_pred['y_pred'] = self.y_pred_train
        self.X_y_pred['binary_y_pred'] = binary_y_pred
        metric_by_group = {}
        max_min_values = []
        final_i = self.find_cutoffs(my_metrics, binary_y_pred, cutoffs_for_groups,
                           max_min_values, metric_by_group, cutoff)

        if len(max_min_values) != 0:
            plot_graphs.max_min_by_iteration(max_min_values, final_i, self.sensitive_feature, filename_prefix=self.filename_prefix)
            plot_graphs.metric_by_iteration(metric_by_group, self.metric, self.sensitive_feature, final_i, filename_prefix=self.filename_prefix)
        return cutoffs_for_groups

    def find_cutoffs(self, my_metrics, binary_y_pred, cutoffs_for_groups, max_min_values, metric_by_group, cutoff):
        return None

    def set_binary_predictions(self, candidates, i):
        candidates.loc[:, 'binary_y_pred'] = 0
        chosen = candidates.iloc[:i, :]
        candidates.loc[chosen.index, 'binary_y_pred'] = 1

    def get_metric_value(self, candidates, i, group):
        if i != 0:
            self.set_binary_predictions(candidates, i)
            metric_value = self.calculate_metric(candidates, self.y_train, self.metric)
        else:
            metric_value = 0
        return metric_value

    def calculate_metric(self, patients, y_true, metric):
        indexes = patients.index
        patients_y_true = y_true.loc[indexes]
        patients_y_pred = patients['binary_y_pred']
        if metric == my_constants.SENSITIVITY:
            score = sensitivity_score(patients_y_true, patients_y_pred)
            return score

    def build_metric_dict_helper(self, group, initial_i, final_i, candidates=None):
        self.metric_group_dict_unique[group] = set()
        if candidates is None:
            candidates = self.X_y_pred[self.X_y_pred[self.sensitive_feature] == group].sort_values(by='y_pred',
                                                                                                   ascending=False)
        for i in range(initial_i, final_i):
            metric_value = self.get_metric_value(candidates, i, group)
            self.metric_group_dict_unique[group].add(metric_value)
            if (group, metric_value) not in self.metric_group_dict:
                self.metric_group_dict[(group, metric_value)] = []
            self.metric_group_dict[(group, metric_value)].append(i)


class PostProcessBruteForce(PostProcessAlgo):

    def __init__(self, percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix):
        self.min_metric_diff = float('inf')
        self.best_combination = []
        self.metric_group_dict_unique = {}
        self.metric_group_dict = {}
        self.possible_combinations = []
        self.best_metric_values = []
        self.possible_metric_vals = []
        self.group_sizes = {}
        self.post_process_by_iters = get_algo(my_constants.BY_NUM_OF_ITERS, percent_cutoff, X_train, y_train,
                                              y_pred_train, sensitive_feature, metric,filename_prefix, 500)
        super().__init__(percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix)


    def set_metric_group_dicts(self, metric_group_dict_unique, metric_group_dict):
        self.metric_group_dict_unique = metric_group_dict_unique
        self.metric_group_dict = metric_group_dict

    def find_cutoffs(self, my_metrics, binary_y_pred, cutoffs_for_groups, max_min_values, metric_by_group, cutoff):
        by_iteration_num_results = self.post_process_by_iters.cutoff_for_groups()
        max_min = self.post_process_by_iters.get_min_metric_diff()
        if max_min == 0:
            return
        self.min_metric_diff = max_min
        print(f'initial diff: {self.min_metric_diff}')
        intervention_group_size = int(self.percent_cutoff/100 * len(self.X_train))
        groups = self.X_y_pred[self.sensitive_feature].value_counts().sort_values()
        groups = groups.index.tolist()
        groups = self.order_groups_by_size(groups)
        if len(self.metric_group_dict) == 0:
            self.build_metric_dict(intervention_group_size, groups)
        self.find_combination_from_metric_dict(self.metric_group_dict_unique, groups, intervention_group_size)
        if self.min_metric_diff == max_min:  # the iteration algorithm already found the optimum
            for k, v in by_iteration_num_results.items():
                cutoffs_for_groups[k] = v
        else:
            self.get_cutoffs_from_combination(groups, cutoffs_for_groups)
        return

    def order_groups_by_size(self, groups):
        # Calculate group sizes first
        for group in groups:
            patients = self.X_y_pred[self.X_y_pred[self.sensitive_feature] == group]
            self.group_sizes[group] = len(patients)

        # Now sort the groups by their sizes
        sorted_groups = sorted(groups, key=lambda group: self.group_sizes[group], reverse=True)

        return sorted_groups

    def find_combination_from_metric_dict(self, metric_group_dict_unique, groups, intervention_group_size):
        print('searching for a sensitivity combination')
        group = groups[0]
        group_metrics = metric_group_dict_unique[group]
        t_end = time.time() + 60 * 60 * 48 # stop after 48 hours
        print(f'starting 48 hours')
        for group_metric_val in group_metrics:
            metric_values = [group_metric_val]
            print(group_metric_val)
            self.add_next_group(intervention_group_size, groups, 1, metric_values, t_end)
        # print(f'found the best combinations: {self.best_combination}. The diff is {self.min_metric_diff}')


    def get_cutoffs_from_combination(self, groups, cutoffs_for_groups):
        if len(self.best_combination) > 1:
            combination = self.choose_combination()
        else:
            combination = self.best_combination[0]
        for i, group in enumerate(groups):
            candidates = self.X_y_pred[self.X_y_pred[self.sensitive_feature] == group].sort_values(by='y_pred',
                                                                                                   ascending=False)
            num_of_patients = combination[i]
            last_patient = candidates.iloc[num_of_patients-1]
            cutoff = last_patient['y_pred'] - 0.0000001
            cutoffs_for_groups[group] = cutoff


    def choose_combination(self):
        best_index = -1
        best_value = float('-inf')
        for i, combination_metric_values in enumerate(self.best_metric_values):
            if min(combination_metric_values) > best_value:
                best_index = i
                best_value = min(combination_metric_values)
        return self.best_combination[best_index]


    def build_metric_dict(self, intervention_group_size, groups):
        print('building metric dictionary')
        for group in groups:
            initial_i = 0
            candidates = self.X_y_pred[self.X_y_pred[self.sensitive_feature] == group].sort_values(by='y_pred',
                                                                                                   ascending=False)
            group_size = len(candidates)
            final_i = min(group_size, intervention_group_size)
            self.build_metric_dict_helper(group, initial_i, final_i, candidates)
        for group in groups:
            self.metric_group_dict_unique[group] = sorted(self.metric_group_dict_unique[group])




    def add_next_group(self, intervention_group_size, groups, group_index, metric_values, end_time):
        if time.time() >= end_time:
            print('out of time')
            return
        group = groups[group_index]
        group_metrics = self.metric_group_dict_unique[group]
        i = 0
        while i < len(group_metrics):
            metric_values = metric_values[:group_index]
            group_metric_value = group_metrics[i]
            metric_values.append(group_metric_value)
            max_metric = max(metric_values)
            min_metric = min(metric_values)
            if max_metric - min_metric > self.min_metric_diff:
                # print(f'skipping {metric_values}')
                if max_metric == group_metric_value:  #  this number can only stay the same or grow
                    i = len(group_metrics)
                else:
                    i += 1
                continue
            if not self.check_if_possible(metric_values, groups, intervention_group_size):
                # print(f'skipping {metric_values}')
                i += 1
                continue
            if len(metric_values) == len(groups):  # last group
                if max_metric - min_metric < self.min_metric_diff:
                    # print(f'checking combinations for {metric_values}')
                    self.find_combination(intervention_group_size, groups, 0, metric_values, [])
                    if len(self.possible_combinations) != 0:  # found combinations
                        self.best_combination = self.possible_combinations
                        self.min_metric_diff = max_metric - min_metric
                        self.best_metric_values = self.possible_metric_vals
                        self.possible_combinations = []
                        self.possible_metric_vals = []
            else:
                self.add_next_group(intervention_group_size, groups, group_index + 1, metric_values, end_time)
            i += 1

    def check_if_possible(self, metric_values, groups, intervention_size):
        max_num_of_patients = 0
        group_index = len(metric_values)
        for i in range(group_index):
            metric_val = metric_values[i]
            group = groups[i]
            patients_num_possibilities = self.metric_group_dict[(group, metric_val)]
            max_num_of_patients += max(patients_num_possibilities)
            if max_num_of_patients >= intervention_size:
                return True
        for group in groups[group_index:]:
            group_size = self.group_sizes[group]
            max_num_of_patients += group_size
            if max_num_of_patients >= intervention_size:
                return True
        return False


    def find_combination(self, intervention_group_size, groups, group_index, metric_values, combination):
        if intervention_group_size < 0:
            return
        group = groups[group_index]
        metric_value = metric_values[group_index]
        patients_num_possibilities = self.metric_group_dict[(group, metric_value)]
        if group_index == len(groups) - 1:  # last group
            return self.check_last_group(intervention_group_size, patients_num_possibilities, metric_values, combination)
        else:
            for num in reversed(patients_num_possibilities):
                combination = combination[:group_index]
                combination.append(num)
                if not self.find_combination(intervention_group_size-num, groups, group_index+1, metric_values,
                                             combination):
                    return False
            return True

    def check_last_group(self, intervention_group_size, patients_num_possibilities, metric_values, combination):
        max_possible = max(patients_num_possibilities)
        if sum(combination) + max_possible < intervention_group_size: # there is no possible combination, there is no need to check the rest
            return False
        if intervention_group_size in patients_num_possibilities:
            combination.append(intervention_group_size)
            self.possible_combinations.append(combination)
            self.possible_metric_vals.append(metric_values)
            max_metric = max(metric_values)
            min_metric = min(metric_values)
            # self.min_metric_diff = max_metric - min_metric
            print(f'found combination: {combination}. diff={max_metric-min_metric}')
        return True


class PostProcessBuckets(PostProcessAlgo):
    def __init__(self, percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix,
                 bucket_size=0.01):
        self.min_metric_diff = float('inf')
        self.best_combination = []
        self.buckets = {}
        self.metric_group_dict_unique = {}
        self.metric_group_dict = {}
        self.buckets_with_zeros = set()
        self.bucket_size = bucket_size
        self.group_sizes = {}
        self.post_process_by_iters = get_algo(my_constants.BY_NUM_OF_ITERS, percent_cutoff, X_train, y_train,
                                              y_pred_train, sensitive_feature, metric, filename_prefix, 300)
        self.post_process_brute_force = get_algo(my_constants. BRUTE_FORCE, percent_cutoff, X_train, y_train,
                                                 y_pred_train, sensitive_feature, metric, filename_prefix)
        super().__init__(percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix)

    def find_cutoffs(self, my_metrics, binary_y_pred, cutoffs_for_groups, max_min_values, metric_by_group, cutoff):
        intervention_group_size = int(self.percent_cutoff / 100 * len(self.X_train))
        groups = self.X_y_pred[self.sensitive_feature].value_counts().sort_values()
        groups = groups.index.tolist()
        groups = self.order_groups_by_size(groups)
        self.build_buckets(intervention_group_size, groups, self.bucket_size)
        bin = self.find_sensitivity(intervention_group_size, self.bucket_size, groups)
        combination = self.get_combination(bin, groups, intervention_group_size)
        next_bucket = self.get_next_bucket(bin, groups)

        self.build_metric_dict(groups, combination, next_bucket)
        self.post_process_brute_force.set_metric_group_dicts(self.metric_group_dict_unique, self.metric_group_dict)
        cutoffs = self.post_process_brute_force.cutoff_for_groups()
        for k, v in cutoffs.items():
            cutoffs_for_groups[k] = v


    def get_cutoffs_from_combination(self, combination, groups, cutoffs_for_groups):
        for i, group in enumerate(groups):
            candidates = self.X_y_pred[self.X_y_pred[self.sensitive_feature] == group].sort_values(by='y_pred',
                                                                                                   ascending=False)
            num_of_patients = combination[i]
            last_patient = candidates.iloc[num_of_patients-1]
            cutoff = last_patient['y_pred'] - 0.0000001
            cutoffs_for_groups[group] = cutoff
        return cutoffs_for_groups

    def order_groups_by_size(self, groups):
        # Calculate group sizes first
        for group in groups:
            patients = self.X_y_pred[self.X_y_pred[self.sensitive_feature] == group]
            self.group_sizes[group] = len(patients)

        # Now sort the groups by their sizes
        sorted_groups = sorted(groups, key=lambda group: self.group_sizes[group], reverse=True)

        return sorted_groups

    def build_buckets(self, intervention_group_size, groups, bucket_size):
        self.buckets_with_zeros.clear()
        bins = np.arange(0, 1+bucket_size, bucket_size)
        for group in groups:
            candidates = self.X_y_pred[self.X_y_pred[self.sensitive_feature] == group].sort_values(by='y_pred',
                                                                                                   ascending=False)
            num_of_iterations = min(len(candidates), intervention_group_size)
            # num_of_iterations = len(candidates)
            metric_values = []
            for i in range(num_of_iterations):
                metric_value = self.get_metric_value(candidates, i, group)
                metric_values.append(metric_value)
            bucket_counts = np.histogram(metric_values, bins)[0]
            cumulative_sum = 0
            for i in range(len(bucket_counts)):
                curr_cumulative_sum = cumulative_sum
                if bucket_counts[i] == 0:
                    self.buckets_with_zeros.add(i)
                cumulative_sum += bucket_counts[i]
                bucket_counts[i] += curr_cumulative_sum
            self.buckets[group] = bucket_counts
        print(f'buckets with zeros: {self.buckets_with_zeros}')


    def find_sensitivity(self, intervention_group_size, bucket_size, groups):
        bins = np.arange(0, 1 + bucket_size, bucket_size)
        high = 1.0
        low = 0.0
        smaller_than_group_size = []
        bigger_than_group_size = []
        while True:
            overall_sensitivity = (high + low)/2
            bin = np.digitize(overall_sensitivity, bins, right=True) - 1
            patients_in_bin = 0
            for group in groups:
                patients_in_bin += self.buckets[group][bin]
            if patients_in_bin > intervention_group_size:
                if (bin-1) in smaller_than_group_size:
                    bin_to_return = bin - 1
                    break
                    # return bin - 1
                bigger_than_group_size.append(bin)
                high = overall_sensitivity
            elif patients_in_bin < intervention_group_size:
                if (bin+1) in bigger_than_group_size:
                    bin_to_return = bin
                    break
                    # return bin
                smaller_than_group_size.append(bin)
                low = overall_sensitivity
            else:
                bin_to_return = bin
                break
                # return bin
        print(f'bucket to return: {bin_to_return}')
        if bin_to_return in self.buckets_with_zeros:
            new_bucket_size = 2*bucket_size
            self.buckets.clear()
            print(f'Found a solution in a bucket with zero. Setting the bucket size to {new_bucket_size}')
            self.build_buckets(intervention_group_size, groups, new_bucket_size)
            return self.find_sensitivity(intervention_group_size, new_bucket_size, groups)
        return bin_to_return


    def get_next_bucket(self, bin, groups):
        next_bucket = []
        for group in groups:
            next_bucket.append(self.buckets[group][bin+1])
        return next_bucket

    def get_combination(self, bin, groups, intervention_group_size):
        num_of_patients = 0
        combination = []
        for group in groups:
            group_patients_num = self.buckets[group][bin]
            num_of_patients += group_patients_num
            combination.append(group_patients_num)
        return combination

    def build_metric_dict(self, groups, initial_combination, next_bucket):
        for group_i, group in enumerate(groups):
            initial_i = initial_combination[group_i]
            final_i = next_bucket[group_i]
            self.build_metric_dict_helper(group, initial_i, final_i+1)
        for group in groups:
            self.metric_group_dict_unique[group] = sorted(self.metric_group_dict_unique[group])




class PostProcessBinarySearch(PostProcessAlgo):
    def __init__(self, percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix):
        super().__init__(percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix)


    def find_cutoffs(self, my_metrics, binary_y_pred, cutoffs_for_groups, max_min_values, metric_by_group, cutoff):
        intervention_group_size = int(self.percent_cutoff/100 * len(self.X_train))
        groups = self.X_y_pred[self.sensitive_feature].value_counts().sort_values()
        groups = groups.index.tolist()
        sensitivity = self.find_sensitivity(groups, intervention_group_size)


    def find_sensitivity(self, groups, intervention_group_size):
        high = 1
        low = 0
        while low < high:
            sensitivity = (low + high) /2
            patients = 0
            for group in groups:
                group_patients = self.find_group_patients(group, sensitivity)
                patients += group_patients
            if patients == intervention_group_size:
                return sensitivity
            elif patients < intervention_group_size:
                low = sensitivity
            else:
                high = sensitivity

    def find_group_patients(self, group, sensitivity):
        candidates = self.X_y_pred[self.X_y_pred[self.sensitive_feature] == group]
        y_group = self.y_train.loc[candidates.index]
        candidates = candidates.sort_values(by='y_pred', ascending=False)
        high = len(candidates)
        low = 0
        group_sensitivity = 0
        i = 0
        while abs(sensitivity - group_sensitivity) > 0.01:
            i = (low + high)/2
            group_sensitivity = self.calculate_sensitivity(candidates, i, y_group)
            if group_sensitivity < sensitivity:
                low = i
            elif group_sensitivity > sensitivity:
                high = i
            else:
                return i
        return i


    def calculate_sensitivity(self, candidates, i, y_true_group):
        d, i = math.modf(i)
        i = int(i)
        i_sensitivity = self.get_metric_value(candidates, i, None)
        fractional_candidate_index = candidates.index[i+1]
        d = d * int(y_true_group.loc[fractional_candidate_index])  # if this patient died we want to add it to the sensitivity, otherwise d will be 0
        d_sensitvity = d/int(sum(y_true_group.values))
        return i_sensitivity + d_sensitvity



    def build_metric_dict(self, intervention_group_size, groups):
        for group in groups:
            self.metric_group_dict[group] = {}
            candidates = self.X_y_pred[self.X_y_pred[self.sensitive_feature] == group].sort_values(by='y_pred',
                                                                                                   ascending=False)
            num_of_iterations = min(len(candidates), intervention_group_size)
            for i in range(num_of_iterations):
                metric_value = self.get_metric_value(candidates, i, group)
                self.metric_group_dict[group][i] = metric_value






class PostProcessUntilConvergence(PostProcessAlgo):

    def __init__(self, percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix):
        super().__init__(percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix)

    def find_cutoffs(self, my_metrics, binary_y_pred, cutoffs_for_groups, max_min_values, metric_by_group, cutoff):
        cannot_add = set()
        cannot_remove = set()
        i = 0
        while True:
            # print(i)
            mf = MetricFrame(
                metrics=my_metrics,
                y_true=np.array(self.y_train),
                y_pred=np.array(binary_y_pred),
                sensitive_features=self.X_train[self.sensitive_feature]
            )
            group_dict = mf.by_group[self.metric]
            max_group = max(group_dict)
            min_group = min(group_dict)
            max_min = max_group - min_group
            if max_min == 0:
                return i
            max_min_values.append(max_min)
            for group in group_dict.keys():
                if group not in metric_by_group:
                    metric_by_group[group] = []
                metric_by_group[group].append(group_dict[group])
                if group not in cutoffs_for_groups:  # only happens in the first iteration
                    cutoffs_for_groups[group] = cutoff
            if self.metric == my_constants.SENSITIVITY:
                self.remove_from_max_group(self.X_y_pred, self.sensitive_feature, group_dict, cutoffs_for_groups, cannot_add,
                                      cannot_remove)
                if self.check_for_convergence(group_dict, cannot_remove, cannot_add):
                    break
                self.add_to_min_group(self.X_y_pred, self.sensitive_feature, group_dict, cutoffs_for_groups, cannot_add, cannot_remove)
                if self.check_for_convergence(group_dict, cannot_remove, cannot_add):
                    break
            elif self.metric == my_constants.SPECIFICITY:
                self.remove_from_min_group(self.X_y_pred, self.sensitive_feature, group_dict, cutoffs_for_groups, cannot_add,
                                      cannot_remove)
                if self.check_for_convergence(group_dict, cannot_remove, cannot_add):
                    break
                self.add_to_max_group(self.X_y_pred, self.sensitive_feature, group_dict, cutoffs_for_groups, cannot_add, cannot_remove)
                if self.check_for_convergence(group_dict, cannot_remove, cannot_add):
                    break
            else:
                raise Exception('Post-processing only supports the Sensitivity and Specificity metrics')
            binary_y_pred = np.array(self.X_y_pred['binary_y_pred'])
            i += 1
        return i


    def check_for_convergence(self, group_dict, cannot_remove, cannot_add):
        return len(cannot_remove) == len(group_dict) or len(cannot_add) == len(group_dict)


    def remove_from_max_group(self, X_y_pred, sensitive_feature, group_dict, cutoffs_dict, cannot_add, cannot_remove):
        group_dict_copy = copy.deepcopy(group_dict)
        max_group = group_dict_copy.idxmax()
        cannot_add.add(max_group)
        while max_group in cannot_remove:
            cannot_add.add(max_group)
            del group_dict_copy[max_group]
            max_group = group_dict_copy.idxmax()
        cannot_add.add(max_group)
        if len(cannot_add) == len(group_dict):
            return
        max_group_patients = X_y_pred[(X_y_pred[sensitive_feature] == max_group) &
                                      (X_y_pred['binary_y_pred'] == 1)]
        # print(f'max: group: {max_group}, value: {max_value:.2f}, size of group: {len(max_group_patients):.2f}')
        patient_index = max_group_patients['y_pred'].idxmin()
        patient_pred = max_group_patients['y_pred'].min()
        X_y_pred.at[patient_index, 'binary_y_pred'] = 0
        cutoffs_dict[max_group] = patient_pred  # only patients with higher predictions will get 1


    def add_to_min_group(self, X_y_pred, sensitive_feature, group_dict, cutoffs_dict, cannot_add, cannot_remove):
        group_dict_copy = copy.deepcopy(group_dict)
        min_group = group_dict_copy.idxmin()
        cannot_remove.add(min_group)
        while min_group in cannot_add:
            cannot_remove.add(min_group)
            del group_dict_copy[min_group]
            min_group = group_dict_copy.idxmin()
        cannot_remove.add(min_group)
        min_group_patients = X_y_pred[(X_y_pred[sensitive_feature] == min_group) &
                                      (X_y_pred['binary_y_pred'] == 0)]
        # print(f'min: group: {min_group}, value: {min_value:.2f}, size of group: {len(min_group_patients):.2f}')
        patient_index = min_group_patients['y_pred'].idxmax()
        patient_pred = min_group_patients['y_pred'].max()
        X_y_pred.at[patient_index, 'binary_y_pred'] = 1
        cutoffs_dict[min_group] = patient_pred - 0.0000001  # the new patient is now above the cutoff. TODO find the
        # difference between this and the next one and define cutoff accordingly


    def add_to_max_group(self, X_y_pred, sensitive_feature, group_dict, cutoffs_dict, cannot_add, cannot_remove):
        group_dict_copy = copy.deepcopy(group_dict)
        max_value = group_dict_copy.max()
        max_group = group_dict_copy.idxmax()
        cannot_remove.add(max_group)
        while max_group in cannot_add:
            cannot_remove.add(max_group)
            del group_dict_copy[max_group]
            max_group = group_dict_copy.idxmax()
        cannot_remove.add(max_group)
        max_group_patients = X_y_pred[(X_y_pred[sensitive_feature] == max_group) &
                                      (X_y_pred['binary_y_pred'] == 0)]
        # print(f'max: group: {max_group}, value: {max_value:.2f}, size of group: {len(max_group_patients):.2f}')
        patient_index = max_group_patients['y_pred'].idxmax()
        patient_pred = max_group_patients['y_pred'].max()
        X_y_pred.at[patient_index, 'binary_y_pred'] = 1
        cutoffs_dict[max_group] = patient_pred - 0.0000001  # the new patient is now above the cutoff. TODO find the
        # difference between this and the next one and define cutoff accordingly


    def remove_from_min_group(self, X_y_pred, sensitive_feature, group_dict, cutoffs_dict, cannot_add, cannot_remove):
        group_dict_copy = copy.deepcopy(group_dict)
        min_value = group_dict_copy.min()
        min_group = group_dict_copy.idxmin()
        cannot_add.add(min_group)
        while min_group in cannot_remove:
            cannot_add.add(min_group)
            del group_dict_copy[min_group]
            min_group = group_dict_copy.idxmin()
        cannot_add.add(min_group)
        if len(cannot_add) == len(group_dict):
            return
        min_group_patients = X_y_pred[(X_y_pred[sensitive_feature] == min_group) &
                                      (X_y_pred['binary_y_pred'] == 1)]
        # print(f'min: group: {min_group}, value: {min_value:.2f}, size of group: {len(min_group_patients):.2f}')
        patient_index = min_group_patients['y_pred'].idxmin()
        patient_pred = min_group_patients['y_pred'].min()
        X_y_pred.at[patient_index, 'binary_y_pred'] = 0
        cutoffs_dict[min_group] = patient_pred  # only patients with higher predictions will get 1


class PostProcessIterNum(PostProcessAlgo):
    def __init__(self, num_of_iters, percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix):
        self.num_of_iters = num_of_iters
        self.min_metric_diff = None
        super().__init__(percent_cutoff, X_train, y_train, y_pred_train, sensitive_feature, metric, filename_prefix)

    def find_cutoffs(self, my_metrics, binary_y_pred, cutoffs_for_groups, max_min_values, metric_by_group, cutoff):
        i = 0
        while i < self.num_of_iters:
            # print(i)
            mf = MetricFrame(
                metrics=my_metrics,
                y_true=np.array(self.y_train),
                y_pred=np.array(binary_y_pred),
                sensitive_features=self.X_train[self.sensitive_feature]
            )
            group_dict = mf.by_group[self.metric]
            max_group = max(group_dict)
            min_group = min(group_dict)
            max_min = max_group - min_group
            if max_min == 0:
                return cutoffs_for_groups
            max_min_values.append(max_min)
            for group in group_dict.keys():
                if group not in metric_by_group:
                    metric_by_group[group] = []
                metric_by_group[group].append(group_dict[group])
                if group not in cutoffs_for_groups:  # only happens in the first iteration
                    cutoffs_for_groups[group] = cutoff
            if self.metric == my_constants.SENSITIVITY:
                self.remove_from_max_group(self.X_y_pred, self.sensitive_feature, group_dict, cutoffs_for_groups)
                self.add_to_min_group(self.X_y_pred, self.sensitive_feature, group_dict, cutoffs_for_groups)
            elif self.metric == my_constants.SPECIFICITY:
                self.remove_from_min_group(self.X_y_pred, self.sensitive_feature, group_dict, cutoffs_for_groups)
                self.add_to_max_group(self.X_y_pred, self.sensitive_feature, group_dict, cutoffs_for_groups)
            else:
                raise Exception('Post-processing only supports the Sensitivity and Specificity metrics')
            binary_y_pred = np.array(self.X_y_pred['binary_y_pred'])
            i += 1
        self.min_metric_diff = max_min_values[-1]
        return i

    def get_min_metric_diff(self):
        return self.min_metric_diff

    def remove_from_max_group(self, X_y_pred, sensitive_feature, group_dict, cutoffs_dict):
        max_group = group_dict.idxmax()
        max_group_patients = X_y_pred[(X_y_pred[sensitive_feature] == max_group) &
                                      (X_y_pred['binary_y_pred'] == 1)]
        # print(f'max: group: {max_group}, value: {max_value:.2f}, size of group: {len(max_group_patients):.2f}')
        patient_index = max_group_patients['y_pred'].idxmin()
        patient_pred = max_group_patients['y_pred'].min()
        X_y_pred.at[patient_index, 'binary_y_pred'] = 0
        cutoffs_dict[max_group] = patient_pred  # only patients with higher predictions will get 1

    def add_to_min_group(self, X_y_pred, sensitive_feature, group_dict, cutoffs_dict):
        min_group = group_dict.idxmin()
        min_group_patients = X_y_pred[(X_y_pred[sensitive_feature] == min_group) &
                                      (X_y_pred['binary_y_pred'] == 0)]
        # print(f'min: group: {min_group}, value: {min_value:.2f}, size of group: {len(min_group_patients):.2f}')
        patient_index = min_group_patients['y_pred'].idxmax()
        patient_pred = min_group_patients['y_pred'].max()
        X_y_pred.at[patient_index, 'binary_y_pred'] = 1
        cutoffs_dict[min_group] = patient_pred - 0.0000001  # the new patient is now above the cutoff. TODO find the
        # difference between this and the next one and define cutoff accordingly

    def add_to_max_group(self, X_y_pred, sensitive_feature, group_dict, cutoffs_dict):
        max_value = group_dict.max()
        max_group = group_dict.idxmax()
        max_group_patients = X_y_pred[(X_y_pred[sensitive_feature] == max_group) &
                                      (X_y_pred['binary_y_pred'] == 0)]
        # print(f'max: group: {max_group}, value: {max_value:.2f}, size of group: {len(max_group_patients):.2f}')
        patient_index = max_group_patients['y_pred'].idxmax()
        patient_pred = max_group_patients['y_pred'].max()
        X_y_pred.at[patient_index, 'binary_y_pred'] = 1
        cutoffs_dict[max_group] = patient_pred - 0.0000001  # the new patient is now above the cutoff. TODO find the
        # difference between this and the next one and define cutoff accordingly

    def remove_from_min_group(self, X_y_pred, sensitive_feature, group_dict, cutoffs_dict):
        min_value = group_dict.min()
        min_group = group_dict.idxmin()
        min_group_patients = X_y_pred[(X_y_pred[sensitive_feature] == min_group) &
                                      (X_y_pred['binary_y_pred'] == 1)]
        # print(f'min: group: {min_group}, value: {min_value:.2f}, size of group: {len(min_group_patients):.2f}')
        patient_index = min_group_patients['y_pred'].idxmin()
        patient_pred = min_group_patients['y_pred'].min()
        X_y_pred.at[patient_index, 'binary_y_pred'] = 0
        cutoffs_dict[min_group] = patient_pred  # only patients with higher predictions will get 1