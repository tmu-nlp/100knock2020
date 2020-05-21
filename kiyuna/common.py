from typing import Any, List, Sequence, Tuple


def n_gram(seq: Sequence[Any], n: int) -> List[Tuple[Any]]:
    return list(zip(*[seq[i:] for i in range(n)]))
