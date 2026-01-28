import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import ScalarFormatter
from tsdownsample import M4Downsampler
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from iotdb.Session import Session

warnings.filterwarnings("ignore")

def sanitize(name: str) -> str:
    """清理文件名中的非法字符"""
    # 替换文件名中不允许的字符
    return re.sub(r'[\\/:*?"<>|]', '_', name)

def read_iotdb(
        host="192.168.199.185",
        port="6667",
        user='root',
        password='root',
        path='root.zhlh_202307_202412.ZHLH_4C_1216',
        target_column='*',
        st='2023-07-18 12:00:00',
        et='2024-11-05 23:59:59',
        limit=1000000000,
):
    """从IoTDB数据库读取时序数据
    
    Args:
        host: IoTDB服务器地址
        port: IoTDB服务端口
        user: 用户名
        password: 密码
        path: 数据路径
        target_column: 目标列名
        st: 开始时间
        et: 结束时间
        limit: 数据限制条数
        
    Returns:
        pd.DataFrame: 包含时序数据的DataFrame
    """
    session = Session(host, port, user, password, fetch_size=2000000)
    session.open(False)

    ststring = ">=" + st.replace(' ', 'T')
    etstring = "<=" + et.replace(' ', 'T')

    query = f"select `{target_column}` from {path}"

    if st:
        ststring = ">=" + st.replace(' ', 'T')
    if et:
        etstring = "<=" + et.replace(' ', 'T')
    if st and not et:
        query = query + f" where time {ststring}"
    if et and not st:
        query = query + f" where time {etstring}"
    if st and et:
        query = query + f" where time {ststring} and time {etstring}"
    if limit:
        query = query + f" limit {limit}"

    result = session.execute_query_statement(query)
    result.set_fetch_size(3000000)
    df = result.todf()

    # 设置时间索引并转换时区
    df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
    df.set_index('Time', inplace=True)

    # 重命名列
    column_rename = {}
    for column in df.columns:
        if column.endswith('`'):
            column_new = column[(column.rindex('`', 0, len(column) - 2) + 1): len(column) - 1]
        else:
            column_new = column.split('.')[-1].replace('`', '')
        column_rename[column] = column_new

    df = df.rename(columns=column_rename)
    session.close()
    return df

def downsample_data(data: pd.Series, ratio: float = 5000) -> pd.DataFrame:
    """对时序数据进行M4降采样
    
    Args:
        data: 输入时序数据
        ratio: 降采样比例或目标长度
        
    Returns:
        pd.DataFrame: 降采样后的数据
    """
    if len(data) < 4:
        return pd.DataFrame({
            'date': data.index,
            'value': data.values
        })

    if ratio > 1:
        target_length = ratio
    else:
        target_length = max(4, int(len(data) * ratio))
        target_length = (target_length // 4) * 4
    
    downsampler = M4Downsampler()
    indices = downsampler.downsample(data.values, n_out=target_length)
    
    return pd.DataFrame({
        'date': data.index[indices],
        'value': data.values[indices]
    })

def create_and_save_plot(data: pd.Series, save_path: str):
    """创建并保存时序数据可视化图表
    
    Args:
        data: 时序数据
        save_path: 图片保存路径
        
    Returns:
        bool: 保存成功返回True，失败返回False
    """
    try:
        # 设置图形参数
        fig_width = 20
        fig_height = 4
        fig = Figure(figsize=(fig_width, fig_height), dpi=200, facecolor='white')
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # 设置字体样式
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 12,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14
        })
        
        # 绘制数据
        ax.plot(data.index, data.values, color='black', alpha=1.0, linewidth=0.8)
        
        # 设置刻度
        n_ticks = 25
        if len(data) <= n_ticks:
            ax.set_xticks(range(len(data)))
        else:
            tick_positions = np.linspace(0, len(data)-1, n_ticks, dtype=int)
            ax.set_xticks(tick_positions)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # 设置Y轴范围
        y_min, y_max = data.min(), data.max()
        if y_max > y_min:
            y_range = y_max - y_min
            margin = y_range * 0.05
            ax.set_ylim(y_min - margin, y_max + margin)
        
        # 设置坐标轴格式
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.tick_params(axis='both', which='both', direction='in', length=3, width=0.8, pad=2)
        
        # 美化图形
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        
        # 调整布局并保存
        fig.tight_layout(pad=0.1)
        fig.savefig(save_path, format='jpg', dpi=200, 
                   bbox_inches='tight', pad_inches=0.02,
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"保存图片失败: {save_path} | 错误: {e}")
        return False

import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='IoTDB Data Acquisition and Downsampling')
    parser.add_argument('--host', default="192.168.199.185", help='IoTDB Host')
    parser.add_argument('--port', default="6667", help='IoTDB Port')
    parser.add_argument('--user', default='root', help='IoTDB User')
    parser.add_argument('--password', default='root', help='IoTDB Password')
    parser.add_argument('--source', required=True, help='IoTDB Source Path (e.g. root.db.device)')
    parser.add_argument('--column', default=None, help='Target Column Name (optional, defaults to all/auto)')
    parser.add_argument('--start-time', default=None, help='Start Time (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end-time', default=None, help='End Time (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--target-points', type=int, default=5000, help='Target points for downsampling')
    parser.add_argument('--output-dir', default='./data_downsampled', help='Output directory for CSV files')
    parser.add_argument('--image-dir', default=None, help='Output directory for Image files (optional)')
    return parser.parse_args()

def process_single_request(args):
    """处理单个请求"""
    try:
        # 如果指定了具体列名，只处理该列
        target_columns = [args.column] if args.column else ['*']
        
        # 如果是 *，可能需要先获取列名，或者依赖 read_iotdb 的 * 查询
        # 这里为了兼容 process_single 的逻辑，我们假设 column 是具体的
        
        # 注意：原来的 process_single 需要 path, col, st, et
        # 我们稍微调整一下调用方式
        
        path = args.source
        st = args.start_time
        et = args.end_time
        
        # 这里的 col 实际上是 point name
        # 如果 args.column 为空，我们可能需要先查询一下有哪些列，或者尝试直接查询 *
        # 为了简化，如果未指定 column，我们尝试从 IoTDB 获取 metadata 或者假设 path 就是包含列的路径？
        # 根据用户需求：路径 'root.zhlh...1216', 点位 'FI_10401C.PV'
        # source 对应 path, column 对应 点位名称
        
        cols_to_process = []
        if args.column:
            # 支持逗号分隔
            cols_to_process = args.column.split(',')
        else:
            # 如果没有指定列，尝试查询该路径下的所有列（这个脚本原来逻辑是硬编码列表）
            # 由于时间关系，暂时要求必须指定列，或者修改 read_iotdb 支持 *
            # 原 logic read_iotdb(target_column=col)
            # 如果 args.column 为 None, 这里的逻辑会有问题。
            # 修改策略：如果未指定，报错提示需要指定列，或者默认只取第一列
             print("Warning: No column specified, attempting to query '*'")
             cols_to_process = ['*']

        success_count = 0
        for col in cols_to_process:
            col = col.strip()
            print(f"Processing {col} from {path}...")
            
            # 复用 process_single 的核心逻辑，但需要适配参数
            # 原 process_single: path, col, st, et, path_name
            # 需要把 host/port 等传进去，但 process_single 内部调用的 read_iotdb 参数是硬编码默认值的
            
            # 为了不破坏 process_single 太多，我们可以修改 read_iotdb 的默认值，或者传递参数
            # 更好的是：修改 read_iotdb 接受参数，修改 process_single 接受 session 参数
            
            # 由于是脚本，直接修改全局默认值或通过参数传递
            
            # Let's modify process_single locally or inline it here
            # Reuse read_iotdb but with args
            
            df = read_iotdb(
                host=args.host,
                port=args.port,
                user=args.user,
                password=args.password,
                path=path,
                target_column=col,
                st=st,
                et=et
            )
            
            if df.empty:
                print(f"No data found for {col}")
                continue
                
            # 确定数值列
            value_col = col if col in df.columns else next((c for c in df.columns if c != 'time'), None)
            
            # 特殊处理：IoTDB 返回的列名可能包含 path
            # read_iotdb 已经做了 rename
            if value_col is None:
                 # 尝试找第一个非时间列
                 if len(df.columns) > 0:
                     value_col = df.columns[0]
            
            if value_col is None or value_col not in df.columns:
                 print(f"Could not identify value column for {col}")
                 continue

            data_series = df[value_col].dropna()
            if len(data_series) == 0:
                print(f"Empty series for {col}")
                continue

            # 降采样
            df_downsampled = downsample_data(data_series, ratio=args.target_points)
            df_downsampled['date'] = pd.date_range(start=pd.Timestamp.now().floor('s'), periods=len(df_downsampled), freq='1s')

            # 文件名处理：包含时间范围元数据
            base_name = sanitize(col)
            # 使用 source 的最后一段作为 path_name 标识
            path_name = args.source.split('.')[-1]
            
            # 从数据中提取实际时间范围
            actual_start = data_series.index.min()
            actual_end = data_series.index.max()
            
            # 格式化时间为文件名安全格式 (YYYYMMDD_HHMMSS)
            start_str = actual_start.strftime('%Y%m%d_%H%M%S') if pd.notna(actual_start) else 'unknown'
            end_str = actual_end.strftime('%Y%m%d_%H%M%S') if pd.notna(actual_end) else 'unknown'
            
            # 新文件名格式: {path_name}_{point_name}_{start}_{end}.csv
            file_name = f"{path_name}_{base_name}_{start_str}_to_{end_str}"

            # 确保目录存在
            os.makedirs(args.output_dir, exist_ok=True)
            # os.makedirs('./data/picture_data', exist_ok=True)

            # 保存
            # 注意：DataProcessingAdapter 扫描 data_downsampled 下的 .csv
            csv_path = os.path.join(args.output_dir, f'{file_name}.csv')
            df_downsampled.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"Saved CSV: {csv_path}")

            # 图片
            # 图片
            if args.image_dir:
                os.makedirs(args.image_dir, exist_ok=True)
                
                plot_series = pd.Series(df_downsampled['value'].values, index=range(len(df_downsampled)))
                img_path = os.path.join(args.image_dir, f'{file_name}.jpg')
                img_success = create_and_save_plot(plot_series, img_path)
                
                if img_success:
                   print(f"Saved Image: {img_path}")
            success_count += 1
        
        return success_count > 0

    except Exception as e:
        print(f"Error executing task: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    args = parse_args()
    success = process_single_request(args)
    if not success:
        sys.exit(1)