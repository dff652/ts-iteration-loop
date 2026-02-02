from scipy.stats import ks_2samp
import numpy as np
import pandas as pd
from check_outlier.wavelet import split_continuous_outliers
import math
from .base import BaseEvaluator
from .registry import register_evaluator

def eva(raw_data, ais):
    """
    评估异常数据段的质量并生成评分数据框。

    该函数对输入的异常数据段进行统计分析，计算每个异常段与正常数据的差异程度，
    并生成包含各项评估指标的数据框。

    Args:
        raw_data: 原始数据数组
        ais: 异常数据点的索引数组

    Returns:
        eval_df: pd.DataFrame: 包含以下列的数据框:
            - index: 异常段的起止位置
            - raw_p: 置信度
            - left: 左边界差异得分
            - right: 右边界差异得分
            - score: 综合评分
    """
    sub = split_continuous_outliers(ais,gp=1)
    print(f"初始分段数: {len(sub)}")
    
    # 后处理：合并间隔很小的相邻段
    def merge_close_segments(segments, max_gap=5):
        """合并间隔小于max_gap的相邻段"""
        if len(segments) <= 1:
            return segments
        
        merged = []
        current_segment = segments[0].copy()
        
        for i in range(1, len(segments)):
            next_segment = segments[i]
            gap = next_segment[0] - current_segment[-1]
            
            if gap <= max_gap:
                # 合并段
                current_segment.extend(next_segment)
            else:
                # 保存当前段，开始新段
                merged.append(current_segment)
                current_segment = next_segment.copy()
        
        # 添加最后一个段
        merged.append(current_segment)
        return merged
    
    # 合并间隔≤5的相邻段
    sub = merge_close_segments(sub, max_gap=5)
    print(f"合并后段数: {len(sub)}")
    print(f"异常点总数: {len(ais)}")
    mask0 = np.ones(len(raw_data), dtype=bool)
    mask0[ais.astype(int)] = False
    psq = raw_data[mask0]

    psq_midx=np.arange(len(raw_data))[mask0]

    ol0 = psq.shape[0]
    ol = raw_data.shape[0]
    a0,a1,a2,a3,a4,a5,b1,b2,b3=[],[],[],[],[],[],[],[],[]
    data_len = []
    
    ks_stat0, ks_pvalue0 = ks_2samp(psq,raw_data)
    
    for sai in sub:
        mask = np.ones(len(raw_data), dtype=bool)
        sa = [int(x) for x in sai]
        mask[sa] = False
        ps = raw_data[mask]
        ks_stat, ks_pvalue = ks_2samp(ps,psq)
        ks_stat1, ks_pvalue1 = ks_2samp(ps,raw_data)

        pos0,pos1 = np.searchsorted(psq_midx, sa[0]),np.searchsorted(psq_midx, sa[-1])
        
        lf_eg = psq[max(0,pos0-int(0.03*ol)):pos0]
        rt_eg = psq[pos1:min(ol0,pos1+int(0.03*ol))]

        sv_eg = raw_data[sa]
        sv1m = np.median(sa)

        lfd = max(np.abs(np.mean(lf_eg)-np.max(sv_eg)),
                  np.abs(np.mean(lf_eg)-np.min(sv_eg)))/(np.std(psq)+1e-6)
        rtd = max(np.abs(np.mean(rt_eg)-np.max(sv_eg)),
                  np.abs(np.mean(rt_eg)-np.min(sv_eg)))/(np.std(psq)+1e-6)
        a0.append(len(sa))
        # a1.append(xx)
        a2.append(ks_stat)
        a3.append(ks_stat1)
        a4.append(ks_pvalue)
        a5.append(ks_pvalue1)
        b1.append(lfd)#左边界得分
        b2.append(rtd)#右边界得分
        b3.append(f'{sa[0]}-{sa[-1]}')
        data_len.append(len(sa))
    

    """
    计算总得分
    """
    evadf=pd.DataFrame({'index':b3,'data_len':data_len,'raw_p':a5,'left':b1,'right':b2})
    evadf0 = gen(evadf)
    # evadf['score']=(evadf['left']+evadf['right'])/2-np.log(evadf['raw_p'])
    return evadf0

def avg_score(scores, p=5):
    '''
    计算点位各异常段平均值
    Args:
        scores: evadf0['score']
    '''
    n = len(scores)
    pm = np.linalg.norm(scores, ord=p) / n**(1/p)
    return pm

def gen(df):
    # df = df1.drop(['normal_ks','raw_ks','normal_p'],axis=1)
    df['left']=df['left'].fillna(df['right'])
    df['right']=df['right'].fillna(df['left'])
    df['avgl']=(df['left']+df['right'])/2
    df['p0']=-np.log(df['raw_p']+1e-12)
    df['sc_lf'] = df['avgl'].apply(lr_sc)
    df['sc_p0'] = df['p0'].apply(lr_sc)
    df['W_sc'] = df['sc_lf'].apply(w_sc)
    df['score_adj'] = (100-1e-3)*(df['sc_lf']*(1-df['W_sc'])+df['sc_p0']*df['W_sc'])
    df['score_adj'] = df['score_adj'].fillna(0.0)  # 空值结果兜底处理
    # 归一化公式，确保 score_adj=100 时 score 严格等于 100
    # 归一化因子 = 100 / (10 * log1p(exp(100/10)))
    df['score'] = (10 * np.log1p(np.exp(df['score_adj'] / 10))).round(2)
    # NORM_FACTOR = 100.0 / (10.0 * np.log1p(np.exp(100.0 / 10.0)))
    # df['score'] = (10 * np.log1p(np.exp(df['score_adj'] / 10)) * NORM_FACTOR).round(2)
    # df['score'] = np.clip(df['score'], 0, 100)  # 防止round(2)后超过100
    return df

def lr_sc(x):
    s1 = 2-x
    s2 = -math.exp(s1)
    return math.exp(s2)


def w_sc(x):
    if 1 + 25*x - 25*x**2 <= 0 or x == -1:
        return float('nan')
    return 1/5*(math.log(1 + 25*x - 25*(x**2)) / (1 + 5*x)-1) + 1/2 * x

@register_evaluator("lb_eval")
class LbEvaluator(BaseEvaluator):
    """`lb_eval`评估算法的适配器。

    该类将现有的``eva(raw_data, ais)``算法适配为``BaseEvaluator``接口，
    实现方式为：将``y_true``映射为原始序列``raw_data``，将``y_pred``映射为异常点索引``ais``。
    """

    def evaluate(self, y_true, y_pred, **kwargs):
        """运行`lb_eval`并以字典形式返回结果。

        参数
        ----
        y_true : array-like
            原始序列数据（raw data）。
        y_pred : array-like of int
            异常点的索引列表（ais）。

        返回
        ----
        dict
            包含键"result"的数据，其中"result"对应一个`pandas.DataFrame`，
            含有列`index/raw_p/left/right/score`。
        """
        df = eva(y_true, y_pred)
        return {"result": df}