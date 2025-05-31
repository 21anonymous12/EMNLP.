# MIAS: Multi-Inference Answer Selection for Reliable and Flexible Table Understanding

## How to Use check_exact_match.py

```bash
from check_exact_match import check_exact_match

pred = '1/4'
target = '0.25'
check = check_exact_match(pred, target)
print(check)
# Output: True
```

Running this script will print True since '1/4' and '0.25' are considered equivalent by check_exact_match.
