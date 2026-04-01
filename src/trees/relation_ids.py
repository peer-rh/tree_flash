from __future__ import annotations

RANK_RELATION_CAP = 8

REL_SELF = 0
REL_PARENT = 1
REL_CHILD = 2
REL_SIBLING = 3
REL_ANCESTOR = 4
REL_DESCENDANT = 5
REL_OTHER = 6

REL_PARENT_RANK_BASE = 7
REL_CHILD_RANK_BASE = REL_PARENT_RANK_BASE + RANK_RELATION_CAP
REL_SIBLING_RANK_BASE = REL_CHILD_RANK_BASE + RANK_RELATION_CAP
RELATION_VOCAB_SIZE = REL_SIBLING_RANK_BASE + (RANK_RELATION_CAP * RANK_RELATION_CAP)


def clamp_relation_rank(rank: int) -> int:
    rank = int(rank)
    if rank <= 0:
        return 0
    return min(rank, RANK_RELATION_CAP)


def relation_id_for_parent_rank(rank: int) -> int:
    bucket = clamp_relation_rank(rank)
    if bucket == 0:
        return REL_PARENT
    return REL_PARENT_RANK_BASE + bucket - 1


def relation_id_for_child_rank(rank: int) -> int:
    bucket = clamp_relation_rank(rank)
    if bucket == 0:
        return REL_CHILD
    return REL_CHILD_RANK_BASE + bucket - 1


def relation_id_for_sibling_ranks(rank_i: int, rank_j: int) -> int:
    bucket_i = clamp_relation_rank(rank_i)
    bucket_j = clamp_relation_rank(rank_j)
    if bucket_i == 0 or bucket_j == 0:
        return REL_SIBLING
    return REL_SIBLING_RANK_BASE + ((bucket_i - 1) * RANK_RELATION_CAP) + (bucket_j - 1)
