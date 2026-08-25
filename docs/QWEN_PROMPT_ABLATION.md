# Qwen2.5-VL-3B prompt ablation

All runs used the same frozen 12-case development queue on a free Colab Tesla T4. The protected
test set remained untouched.

| Prompt | Parse | Accuracy | Clean | Corrosion | Crack | Coating |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline labels | 100% | 33.3% | 3/3 | 1/3 | 0/3 | 0/3 |
| Defect-sensitive definitions | 100% | 25.0% | 3/3 | 0/3 | 0/3 | 0/3 |
| Balanced forced choice | 100% | 50.0% | 0/3 | 3/3 | 0/3 | 3/3 |

The balanced prompt raises aggregate accuracy but is not promoted: it introduces a complete
regression on clean panels and still detects no cracks. Together, the runs demonstrate prompt-driven
class collapse rather than robust perception. Further prompt tuning on twelve development examples
would increasingly overfit the benchmark. The next justified step is adaptation with a separate
training corpus, followed by evaluation on these unchanged development cases and finally one frozen
protected-test run.
