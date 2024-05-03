import numpy as np
import pandas as pd

# 示例：计算PSI
def calculate_model_psi(expected_array, actual_array, buckets=10):
    """Calculate the PSI (population stability index) across two distributions of model scores.

    Args:
    expected_array (np.array): Model scores for the training set or time-in sample.
    actual_array (np.array): Model scores for the test set or time-out sample.
    buckets (int): Number of percentile buckets to use in the calculation.

    Returns:
    float: The PSI value.
    """
    # 将分数切割成分位数分箱
    breakpoints = np.linspace(0, 100, buckets + 1)
    breakpoints = np.percentile(expected_array, breakpoints)
    # 确保分箱边界的唯一性
    breakpoints = np.unique(breakpoints)

    # 使用分位数计算分箱后每个箱子的实际和预期人数
    expected_counts = np.histogram(expected_array, breakpoints)[0]
    actual_counts = np.histogram(actual_array, breakpoints)[0]

    # 计算每个箱子的占比
    expected_probs = expected_counts / expected_counts.sum()
    actual_probs = actual_counts / actual_counts.sum()

    # 处理0概率的情况，以避免0概率导致的计算错误
    actual_probs += np.finfo(float).eps
    expected_probs += np.finfo(float).eps

    # 计算PSI
    psi_values = (actual_probs - expected_probs) * np.log(actual_probs / expected_probs)
    psi = np.sum(psi_values)

    return psi

def calculate_interval_psi(expected_array, actual_array, buckets=10):
    # 将分数切割成分位数分箱
    breakpoints = np.linspace(0, 100, buckets + 1)
    breakpoints = np.percentile(expected_array, breakpoints)
    # 确保分箱边界的唯一性
    breakpoints = np.unique(breakpoints)

    # 使用分位数计算分箱后每个箱子的实际和预期人数
    expected_counts = np.histogram(expected_array, breakpoints)[0]
    actual_counts = np.histogram(actual_array, breakpoints)[0]

    # 计算每个箱子的占比
    expected_probs = expected_counts / expected_counts.sum()
    actual_probs = actual_counts / actual_counts.sum()

    # 处理0概率的情况，以避免0概率导致的计算错误
    actual_probs += np.finfo(float).eps
    expected_probs += np.finfo(float).eps

    # 计算每个区间的PSI
    interval_psis = (actual_probs - expected_probs) * np.log(actual_probs / expected_probs)

    # 构建输出表格
    results = pd.DataFrame({
        'Breakpoint Start': breakpoints[:-1],
        'Breakpoint End': breakpoints[1:],
        'Expected Proportion': expected_probs,
        'Actual Proportion': actual_probs,
        'Interval PSI': interval_psis
    })

    return results


def calculate_bin_woe_id(col, data):

    df = pd.DataFrame(data)

    # 计算每个箱子中好客户和坏客户的数量
    grouped = df.groupby(col).agg(
        total=('target', 'count'),
        bad=('target', 'sum')
    )
    grouped['good'] = grouped['total'] - grouped['bad']

    # 计算坏客户和好客户的总数
    total_bad = grouped['bad'].sum()
    total_good = grouped['good'].sum()

    # 计算分布比率并添加拉普拉斯平滑
    # grouped['bad_dist'] = (grouped['bad'] + 0.5) / total_bad
    # grouped['good_dist'] = (grouped['good'] + 0.5) / total_good
    grouped['bad_dist'] = (grouped['bad']) / total_bad
    grouped['good_dist'] = (grouped['good']) / total_good
    # 计算WOE
    grouped['WOE'] = np.log(grouped['bad_dist'] / grouped['good_dist'])

    # 计算IV
    grouped['IV'] = (grouped['bad_dist'] - grouped['good_dist']) * grouped['WOE']
    total_iv = grouped['IV'].sum()
    grouped['feature'] = col
    # 将总的IV值添加到DataFrame
    # grouped.loc['Total'] = pd.Series({'IV': total_iv}, index=['IV'])
    # print(grouped[['bad', 'good', 'bad_dist', 'good_dist', 'WOE', 'IV']])
    # 输出DataFrame
    return grouped


