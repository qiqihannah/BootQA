from scipy.stats import mannwhitneyu

def a12(lst1,lst2,rev=True):
  "how often is x in lst1 more than y in lst2?"
  more = same = 0.0
  for x in lst1:
    for y in lst2:
      if x==y : same += 1
      elif rev and x > y : more += 1
      elif not rev and x < y : more += 1
  return (more + 0.5*same)  / (len(lst1)*len(lst2))

def stat_test(app1, app2):
    statistic, pvalue = mannwhitneyu(app1, app2, alternative='two-sided')

    # Calculate the A12 effect size using Vargha and Delaney's formula
    a12_effect_size = a12(app1, app2)
    return statistic, pvalue, a12_effect_size