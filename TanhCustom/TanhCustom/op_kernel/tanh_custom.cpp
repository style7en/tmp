#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

// 算子输入/输出数据类型定义（本算子仅支持FP16；若编译环境未通过编译选项注入，则此处兜底定义为half）
#ifndef DTYPE_X
#define DTYPE_X half
#endif
#ifndef DTYPE_Y
#define DTYPE_Y half
#endif

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // totalLength为host侧已按核数均分后的每核元素个数，直接作为blockLength
        this->blockLength = totalLength;
        this->tileNum = tileNum;
        // 每次CopyIn/Compute/CopyOut处理tileLength个元素：
        // 每核数据blockLength被切分为 tileNum * BUFFER_NUM 份（双缓冲）
        this->tileLength = this->blockLength / (this->tileNum * BUFFER_NUM);

        // 本核处理第GetBlockIdx()个block的数据
        this->xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + AscendC::GetBlockIdx() * this->blockLength,
                                  this->blockLength);
        this->yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + AscendC::GetBlockIdx() * this->blockLength,
                                  this->blockLength);

        // 初始化队列（双缓冲）与计算临时Buffer
        this->pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        this->pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        this->pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(DTYPE_X));
        this->pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(DTYPE_X));
        this->pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(DTYPE_X));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 从GlobalMemory拷贝tileLength个元素到UnifiedBuffer（VECIN队列）
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue<DTYPE_X>(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_X> expX = tmpBuf0.Get<DTYPE_X>();  // exp(x)
        AscendC::LocalTensor<DTYPE_X> expNX = tmpBuf1.Get<DTYPE_X>(); // exp(-x)
        AscendC::LocalTensor<DTYPE_X> tmp = tmpBuf2.Get<DTYPE_X>();   // 分子

        AscendC::Muls(tmp, xLocal, (DTYPE_X)(-1.0f), this->tileLength); // tmp = -x
        AscendC::Exp(expX, xLocal, this->tileLength);                   // expX = exp(x)
        AscendC::Exp(expNX, tmp, this->tileLength);                     // expNX = exp(-x)
        AscendC::Sub(tmp, expX, expNX, this->tileLength);               // tmp = exp(x) - exp(-x)
        AscendC::Add(expX, expX, expNX, this->tileLength);              // expX = exp(x) + exp(-x)（原地复用为分母）
        AscendC::Div(yLocal, tmp, expX, this->tileLength);              // y = tanh(x)

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 计算结果从UnifiedBuffer（VECOUT队列）拷贝回GlobalMemory
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0,tmpBuf1,tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // 核函数入口：从tiling中取出总长度与tile切分个数，初始化并启动流水
    KernelTanh op;
    op.Init(x, y, tilingData.blockLength, tilingData.tileNum);
    op.Process();
}