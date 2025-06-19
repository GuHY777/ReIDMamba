# encoding: utf-8
# copy from: https://github.com/open-mmlab/OpenUnReID/blob/66bb2ae0b00575b80fbe8915f4d4f4739cc21206/openunreid/core/utils/faiss_utils.py

import faiss
import torch
from faiss import swigfaiss


def swig_ptr_from_FloatTensor(x):
    '''
    ```python
    import torch

    # 创建一个包含4个float元素的存储
    storage = torch.FloatStorage.from_sequence([1.0, 2.0, 3.0, 4.0])

    # 使用这个存储创建一个张量，从存储中的第二个元素开始
    offset = 1  # 第二个元素的偏移量（索引从0开始）
    tensor = torch.tensor(storage[offset:])
    ```

    在这个例子中，`storage` 包含了4个元素 `[1.0, 2.0, 3.0, 4.0]`。我们创建的 `tensor` 是一个子视图，它从 `storage` 中的第二个元素 `2.0` 开始，包含 `storage` 中的后三个元素。

    现在，我们来计算 `x.storage().data_ptr() + x.storage_offset() * 4`：

    1. `x.storage()` 返回 `storage` 对象。
    2. `x.storage().data_ptr()` 返回指向 `storage` 底层数据的指针。由于 `storage` 是连续的，这个指针将指向 `1.0` 的内存地址（在这个例子中，我们假设它是第一个元素的实际内存地址）。
    3. `x.storage_offset()` 返回 `offset`，即 `1`。
    4. `x.storage_offset() * 4` 计算为 `1 * 4 = 4`。这个值表示我们需要从 `storage` 的起始指针跳过 `1` 个 `float` 类型元素的字节长度，因为 `float` 类型占用4个字节。

    因此，`x.storage().data_ptr() + x.storage_offset() * 4` 的值将是指向 `storage` 中 `2.0` 这个元素的指针。如果我们打印这个指针的值（假设我们有一个函数可以打印指针指向的值），它应该显示 `2.0`。

    在实际代码中，这个表达式通常用于获取指向张量数据的指针，以便可以将其传递给需要直接访问内存的C或C++函数。在这种情况下，指针的值将取决于具体的内存布局和操作系统的内存管理。
    
    '''
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.untyped_storage().data_ptr() + x.storage_offset() * 4
    )


def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, "dtype=%s" % x.dtype
    return faiss.cast_integer_to_idx_t_ptr(
        x.untyped_storage().data_ptr() + x.storage_offset() * 8 # 64bit / 8bit = 8byte
    )


def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous() #确保x是连续的，方便后续C语言对底层存储空间的操作
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize() #torch.cuda.synchronize() 在这个函数中充当一个安全措施，确保 GPU 上的操作按正确的顺序执行，并且在进行搜索之前，所有相关的数据都已正确设置和同步。
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr, k, Dptr, Iptr) # 这个函数通常在 GPU 或 CPU 上执行，具体取决于 FAISS 索引的配置和执行环境。由于 index.search_c 是用 C 语言编写的，它通常比纯 Python 实现更快，因为它减少了 Python 层面的开销，并允许更有效地利用底层硬件。
    '''
    n: 一个整数，表示查询向量的数量。
    xptr: 一个指向查询向量数组的指针，数据类型为 float*。这个指针指向包含 n 个查询向量的内存位置，每个查询向量具有相同的维度 d。
    k: 一个整数，表示每个查询向量要检索的最近邻（nearest neighbors, NN）的数量。
    Dptr: 一个指向距离数组的指针，数据类型为 float*。这个数组用于存储检索到的最近邻向量与查询向量之间的距离。Dptr 指向的数组应该有大小至少为 n * k，以存储所有查询向量的结果。
    Iptr: 一个指向索引数组的指针，数据类型为 long*（或 int64_t*）。这个数组用于存储检索到的最近邻向量的索引。与 Dptr 一样，Iptr 指向的数组也应该有至少 n * k 的大小。
    '''
    torch.cuda.synchronize()
    return D, I


def search_raw_array_pytorch(res, xb, xq, k, D=None, I=None, metric=faiss.METRIC_L2):
    assert xb.device == xq.device

    nq, d = xq.size()
    if xq.is_contiguous():
        xq_row_major = True
    elif xq.t().is_contiguous():
        xq = xq.t()  # I initially wrote xq:t(), Lua is still haunting me :-)
        xq_row_major = False
    else:
        raise TypeError("matrix should be row or column-major")

    xq_ptr = swig_ptr_from_FloatTensor(xq)

    nb, d2 = xb.size()
    assert d2 == d
    if xb.is_contiguous():
        xb_row_major = True
    elif xb.t().is_contiguous():
        xb = xb.t()
        xb_row_major = False
    else:
        raise TypeError("matrix should be row or column-major")
    xb_ptr = swig_ptr_from_FloatTensor(xb)

    if D is None:
        D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    else:
        assert D.shape == (nq, k)
        assert D.device == xb.device

    if I is None:
        I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)
    else:
        assert I.shape == (nq, k)
        assert I.device == xb.device

    D_ptr = swig_ptr_from_FloatTensor(D)
    I_ptr = swig_ptr_from_LongTensor(I)

    # faiss.bruteForceKnn(
    #     res,
    #     metric,
    #     xb_ptr,
    #     xb_row_major,
    #     nb,
    #     xq_ptr,
    #     xq_row_major,
    #     nq,
    #     d,
    #     k,
    #     D_ptr,
    #     I_ptr,
    # )   
    
    gpu_distance_params = faiss.GpuDistanceParams()
    gpu_distance_params.metric = metric
    gpu_distance_params.dims = d
    gpu_distance_params.k = k
    gpu_distance_params.queries = xq_ptr
    gpu_distance_params.numQueries = nq
    gpu_distance_params.vectors = xb_ptr
    gpu_distance_params.numVectors = nb
    gpu_distance_params.outDistances = D_ptr
    gpu_distance_params.outIndices = I_ptr
    gpu_distance_params.vectorsRowMajor = xb_row_major
    gpu_distance_params.queriesRowMajor = xq_row_major
    gpu_distance_params.vectorType = swigfaiss.DistanceDataType_F32  # 假设基向量是 float32
    gpu_distance_params.queryType = swigfaiss.DistanceDataType_F32   # 假设查询向量是 float32
    gpu_distance_params.outIndicesType = swigfaiss.IndicesDataType_I64  # 输出索引类型是 int64
    gpu_distance_params.device = -1  # 使用当前 CUDA 线程本地设备

    # 调用 bfKnn 包装器
    faiss.bfKnn(res, gpu_distance_params)

    return D, I


def index_init_gpu(ngpus, feat_dim):
    flat_config = []
    for i in range(ngpus):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    res = [faiss.StandardGpuResources() for i in range(ngpus)]
    indexes = [
        faiss.GpuIndexFlatL2(res[i], feat_dim, flat_config[i]) for i in range(ngpus)
    ]
    index = faiss.IndexShards(feat_dim) # 索引分片，可以将索引划分为多个部分，并在多个 GPU 上并行处理。
    for sub_index in indexes:
        index.add_shard(sub_index)
    index.reset()
    return index


def index_init_cpu(feat_dim):
    return faiss.IndexFlatL2(feat_dim)


def faiss_search(qf, gf=None, k=-1, search_option=0):
    '''
    qf: query features (torch tensor)
    gf: gallery features (torch tensor)
    k: number of neighbors to retrieve
    search_option: 
        0: GPU + PyTorch CUDA Tensors (1), #2
        1: GPU + PyTorch CUDA Tensors (2), #1 only search in qf
        2: GPU,                            #3
        3: CPU.                            #4 The result may be different from GPU search.
                                              Mismatched elements: 3316 / 4000000 (0.0829%)
    '''
    if search_option < 2:
        # torch.cuda.empty_cache()
        qf = qf.cuda()
        if gf is not None:
            gf = gf.cuda()
    
    if gf is None:
        gf = qf
            
    if k == -1:
        k = gf.size(0)

    ngpus = faiss.get_num_gpus()

    if search_option == 0:
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, I = search_raw_array_pytorch(res, gf, qf, k)
        I = I.cpu().numpy()
    elif search_option == 1:
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, qf.size(-1))
        index.add(qf.cpu())
        _, I = search_index_pytorch(index, qf, k)
        res.syncDefaultStreamCurrentDevice()
        I = I.cpu().numpy()
    elif search_option == 2:
        # GPU
        index = index_init_gpu(ngpus, qf.size(-1))
        index.add(gf)
        _, I = index.search(qf, k)
    else:
        # CPU
        index = index_init_cpu(qf.size(-1))
        index.add(gf)
        _, I = index.search(qf, k)
    
    return I
        

""" 测试各搜索方法的速度
import timeit

N=20

setup = '''
import torch
from evaluation.rerank.faiss_utils import faiss_search
torch.random.manual_seed(0)
qf = torch.rand(2000, 64)
gf = None
'''

f1 = '''faiss_search(qf, gf, k=-1, search_option=0)
'''

f2 = '''faiss_search(qf, gf, k=-1, search_option=1)
'''

f3 = '''faiss_search(qf, gf, k=-1, search_option=2)
'''

f4 = '''faiss_search(qf, gf, k=-1, search_option=3)
'''

time1 = timeit.timeit(
    f1,
    setup=setup,
    number=N
)
time2 = timeit.timeit(
    f2,
    setup=setup,
    number=N
)
time3 = timeit.timeit(
    f3,
    setup=setup,
    number=N
)
time4 = timeit.timeit(
    f4,
    setup=setup,
    number=N
)

print('Time1: {} s'.format(time1))
print('Time2: {} s'.format(time2))
print('Time3: {} s'.format(time3))
print('Time4: {} s'.format(time4))


import torch
import numpy as np
from evaluation.rerank.faiss_utils import faiss_search
torch.random.manual_seed(0)
qf = torch.rand(2000, 64)
gf = None

I1 = faiss_search(qf, gf, k=-1, search_option=0)
I2 = faiss_search(qf, gf, k=-1, search_option=1)
I3 = faiss_search(qf, gf, k=-1, search_option=2)
I4 = faiss_search(qf, gf, k=-1, search_option=3)

np.testing.assert_array_equal(I1, I2)
np.testing.assert_array_equal(I1, I3)
np.testing.assert_array_equal(I3, I4)
"""    
