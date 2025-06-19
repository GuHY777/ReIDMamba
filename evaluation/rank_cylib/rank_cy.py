# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3
import cython
import numpy as np


# Main interface
def evaluate_cy(indices: cython.long[:, :], q_pids: cython.long[:], g_pids: cython.long[:], q_camids: cython.long[:], g_camids: cython.long[:], max_rank: cython.long):
    num_q: cython.long = indices.shape[0]
    num_g: cython.long = indices.shape[1]
    matches: cython.long[:]
    all_cmc: cython.float[:, :] = np.zeros((num_q, max_rank), dtype=np.float32)
    all_AP: cython.float[:] = np.zeros(num_q, dtype=np.float32)
    all_INP: cython.float[:] = np.zeros(num_q, dtype=np.float32)
    num_valid_q: cython.float = 0. # number of valid query
    valid_index: cython.long = 0
    q_idx: cython.long
    q_pid: cython.long
    q_camid: cython.long
    g_idx: cython.long
    order: cython.long[:] = np.zeros(num_g, dtype=np.int64)
    raw_cmc: cython.float[:] = np.zeros(num_g, dtype=np.float32) # binary vector, positions with value 1 are correct matches
    cmc: cython.float[:] = np.zeros(num_g, dtype=np.float32)
    max_pos_idx: cython.long = 0
    inp: cython.float
    num_g_real: cython.long
    rank_idx: cython.long
    meet_condition: cython.ulong
    num_rel: cython.float
    tmp_cmc: cython.float[:] = np.zeros(num_g, dtype=np.float32)
    tmp_cmc_sum: cython.float
    
    # q_pids = np.asarray(q_pids, dtype=np.int64)
    # g_pids = np.asarray(g_pids, dtype=np.int64)
    # q_camids = np.asarray(q_camids, dtype=np.int64)
    # g_camids = np.asarray(g_camids, dtype=np.int64)

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))
        
    avg_cmc: cython.float[:] = np.zeros(max_rank, dtype=np.float32)

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        for g_idx in range(num_g):
            order[g_idx] = indices[q_idx, g_idx]
        num_g_real = 0
        meet_condition = 0
        matches = (np.asarray(g_pids)[np.asarray(order)] == q_pid).astype(np.int64)

        # remove gallery samples that have the same pid and camid with query
        for g_idx in range(num_g):
            if (g_pids[order[g_idx]] != q_pid) or (g_camids[order[g_idx]] != q_camid):
                raw_cmc[num_g_real] = matches[g_idx]
                num_g_real += 1
                # this condition is true if query appear in gallery
                if matches[g_idx] > 1e-31:
                    meet_condition = 1

        if not meet_condition:
            # this condition is true when query identity does not appear in gallery
            continue

        # compute cmc
        function_cumsum(raw_cmc, cmc, num_g_real)
        # compute mean inverse negative penalty
        # reference : https://github.com/mangye16/ReID-Survey/blob/master/utils/reid_metric.py
        max_pos_idx = 0
        for g_idx in range(num_g_real):
            if (raw_cmc[g_idx] == 1) and (g_idx > max_pos_idx):
                max_pos_idx = g_idx
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP[valid_index] = inp

        for g_idx in range(num_g_real):
            if cmc[g_idx] > 1:
                cmc[g_idx] = 1

        for rank_idx in range(max_rank):
            all_cmc[q_idx, rank_idx] = cmc[rank_idx]
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        function_cumsum(raw_cmc, tmp_cmc, num_g_real)
        num_rel = 0
        tmp_cmc_sum = 0
        for g_idx in range(num_g_real):
            tmp_cmc_sum += (tmp_cmc[g_idx] / (g_idx + 1.)) * raw_cmc[g_idx]
            num_rel += raw_cmc[g_idx]
        all_AP[valid_index] = tmp_cmc_sum / num_rel
        valid_index += 1

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    # compute averaged cmc
    for rank_idx in range(max_rank):
        for q_idx in range(num_q):
            avg_cmc[rank_idx] += all_cmc[q_idx, rank_idx]
        avg_cmc[rank_idx] /= num_valid_q

    return np.asarray(avg_cmc).astype(np.float32), np.asarray(all_AP[:valid_index]), np.asarray(all_INP[:valid_index])


# Compute the cumulative sum
@cython.cfunc
def function_cumsum(src: cython.numeric[:], dst: cython.numeric[:], n: cython.long):
    i: cython.long
    dst[0] = src[0]
    for i in range(1, n):
        dst[i] = src[i] + dst[i - 1]