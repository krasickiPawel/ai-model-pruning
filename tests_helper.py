import constants


def print_result_difference_and_significance(result):
    direction = "worse" if result.statistic > 0 else "better"
    significance = "statistically significant" if result.pvalue < constants.P_VALUE else "NOT statistically significant"
    print(f"{direction} result after pruning, {significance}")


def print_result_basic_info(first_score, second_score, pruning_val, result):
    print(f"Original score: {first_score}")
    print(f"Pruning: {pruning_val*100:.0f}% score: {second_score}")
    print("Stat:", result.statistic)
    print("p-val:", f"{result.pvalue:.6f}")
