#!/bin/bash

# @File    :   run.sh
# @Time    :   2024/10/30
# @Author  :   DouFengfeng
# @Version :   1.0.0
# @Contact :   ff.dou@cyber-insight.com
# @License :   (C)Copyright 2019-2026, CyberInsight
# @Desc    :   数据质量指标计算
# @Update  :   DouFengfeng, 2024/12/26

source /opt/miniconda3/bin/activate /opt/miniconda3/envs/chatts_test  # 使用克隆的测试环境
echo "已激活 chatts_test 环境"
python --version  # 检查 Python 版本

# 修复 CUDA_VISIBLE_DEVICES 问题：
# 只清除可能存在的空值，不要显式设置，让 PyTorch 自动检测所有 GPU
# 设置 CUDA_VISIBLE_DEVICES=0,1 反而会在某些 PyTorch 版本中触发 bug
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    unset CUDA_VISIBLE_DEVICES
    echo "已清除空的 CUDA_VISIBLE_DEVICES，让 PyTorch 自动检测 GPU"
else
    echo "当前 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# 加载 JSON 参数的函数
load_params() {
  local file=$1
  local key=$2
  # 构建配置文件的完整路径
  local config_path="./configs/$file"
  # 使用 jq 提取 JSON 文件中指定键的内容
  if [ -f "$config_path" ]; then
    jq -c ".$key" "$config_path"
  else
    echo "{}"  # 如果文件不存在，返回空对象
  fi
}

# 展示参数并确认
confirm_params() {
  local params_json=$1
  echo "当前参数设置如下："
  echo "$params_json" | jq
  echo -n "确认以上设置吗？ (y/n): "
  read confirm
  if [ "$confirm" != "y" ]; then
    echo "取消操作。"
    return 1
  fi
  return 0
}

find_latest_params() {
  local mode=$1
  local dir="./params"
  # 找到最近的参数文件 - 支持子目录结构
  latest_file=$(find "$dir" -name "${mode}_params_*.json" 2>/dev/null | sort | tail -n 1)
  if [ -z "$latest_file" ]; then
    echo ""  # 如果未找到，返回空字符串
  else
    echo "$latest_file"
  fi
}


# 将 JSON 参数转换为命令行参数格式
json_to_args() {
  local json=$1
  echo "$json" | jq -r '
    def recurse_to_args(prefix):
      to_entries[] | 
      if .value == null then
        empty  # 跳过值为 null 的字段
      elif .value | type == "object" then
        .value | recurse_to_args((prefix + .key + "."))
      else
        # 处理布尔值
        if .key == "use_trend" or .key == "use_seasonal" or .key == "use_resid" then
          "--" + prefix + .key + " " + (.value | tostring)
        elif .key == "use_direct_data" and .value == true then
          "--" + prefix + .key  # store_true 类型的参数，只需要参数名，不需要值
        else
          "--" + prefix + .key + " " + 
          (.value | 
            if type == "string" then 
              . # 保留原始字符串
            else 
              tostring # 数字直接转字符串
            end)
        end
      end;
    recurse_to_args("")
  '
}







# 主交互逻辑
run_interactive() {
  local mode=$1  # global 或 local
  local task_type=$2 # full, detection_only

  echo "请选择参数配置方式："
  echo "1. 手动逐项配置"
  echo "2. 使用默认参数"
  echo "3. 使用上一次参数配置"
  echo -n "请输入选项 [1/2/3]: "
  read config_choice

  local params
  local latest_params_file
  case $config_choice in
    1)  # 手动逐项配置
      echo "开始逐项配置参数："
      # 读取当前模式下的默认参数JSON
      default_json=$(jq -c ".${mode}" ./configs/default_params.json)
      # 获取 JSON 中的默认值（支持嵌套路径）
      get_default() {
        local json=$1
        local key=$2
        echo "$json" | jq -r --arg k "$key" '.[$k] // empty'
      }
      get_method_default() {
        local json=$1
        local method=$2
        local key=$3
        echo "$json" | jq -r --arg m "$method" --arg k "$key" '.method_params[$m][$k] // empty'
      }
      read_input() {
        local prompt=$1
        local default=$2
        read -p "$prompt (按回车使用默认值: $default): " input
        if [ -z "$input" ]; then
          input="$default"
        fi
        echo "$input"
      }
      
      echo ""
      echo "========== 通用参数配置 =========="
      method=$(read_input "请输入检测方法 (stl_wavelet, adtk_hbos, chatts, timer)" "$(get_default "$default_json" "method")")
      n_jobs=$(read_input "请输入并行任务数" "$(get_default "$default_json" "n_jobs")")
      config_file=$(read_input "请输入点位配置文件名 (相对 configs 目录)" "$(get_default "$default_json" "config_file")")
      ratio=$(read_input "请输入降采样参数 (0-1为比例降采样)" "$(get_default "$default_json" "ratio")")
      min_threshold=$(read_input "请输入最小数据点阈值" "$(get_default "$default_json" "min_threshold")")
      downsampler=$(read_input "请输入降采样方法 (m4, minmax, none)" "$(get_default "$default_json" "downsampler")")
      enable_plotting=$(read_input "是否启用绘图流程 (true 或 false)" "$(get_default "$default_json" "enable_plotting")")
      save_mode=$(read_input "保存模式 (raw 或 downsample)" "$(get_default "$default_json" "save_mode")")
      data_path=$(read_input "请输入数据路径 (data_path)" "$(get_default "$default_json" "data_path")")
      label_path=$(read_input "请输入标签保存路径 (label_path)" "$(get_default "$default_json" "label_path")")
      fig_path=$(read_input "请输入图片保存路径 (fig_path)" "$(get_default "$default_json" "fig_path")")
      fig_type=$(read_input "请输入图表类型 (combined 或 focus)" "$(get_default "$default_json" "fig_type")")
      run_heatmap=$(read_input "是否运行热力图生成 (true 或 false)" "$(get_default "$default_json" "run_heatmap")")
      
      # 根据选择的 method 配置专属参数
      echo ""
      echo "========== ${method} 专属参数配置 =========="
      
      # 初始化所有方法参数为默认值或空值
      device="cpu"
      threshold=8; num_points=1200; ratio_classify=0.2; post_filter="mean"; threshold_filter=0.5
      use_trend="true"; use_seasonal="false"; use_resid="true"; decompose="true"; insert_missing="true"
      bin_nums=20; hbos_ratio=""
      n_downsample=10000
      chatts_model_path=""; chatts_device=""; chatts_use_cache=""; chatts_lora_adapter_path=""; chatts_max_new_tokens=4096; chatts_prompt_template="default"; chatts_load_in_4bit="auto"
      timer_model_path=""; timer_device=""; timer_lookback_length=256
      timer_threshold_k=3.5; timer_method="mad"
      
      case $method in
        stl_wavelet|piecewise_linear|standardized|iforest|cv)
          device=$(read_input "请输入设备类型 (cpu, gpu)" "$(get_method_default "$default_json" "stl_wavelet" "device")")
          threshold=$(read_input "请输入异常阈值" "$(get_method_default "$default_json" "stl_wavelet" "threshold")")
          num_points=$(read_input "请输入窗口内合并点数" "$(get_method_default "$default_json" "stl_wavelet" "num_points")")
          ratio_classify=$(read_input "请输入分类比例" "$(get_method_default "$default_json" "stl_wavelet" "ratio_classify")")
          use_trend=$(read_input "是否使用趋势组件 (true 或 false)" "$(get_method_default "$default_json" "stl_wavelet" "use_trend")")
          use_seasonal=$(read_input "是否使用季节性组件 (true 或 false)" "$(get_method_default "$default_json" "stl_wavelet" "use_seasonal")")
          use_resid=$(read_input "是否使用残差组件 (true 或 false)" "$(get_method_default "$default_json" "stl_wavelet" "use_resid")")
          decompose=$(read_input "是否进行STL分解 (true 或 false)" "$(get_method_default "$default_json" "stl_wavelet" "decompose")")
          insert_missing=$(read_input "是否插值补齐缺失 (true 或 false)" "$(get_method_default "$default_json" "stl_wavelet" "insert_missing")")
          ;;
        adtk_hbos)
          bin_nums=$(read_input "直方图分箱数量 (整数)" "$(get_method_default "$default_json" "adtk_hbos" "bin_nums")")
          hbos_ratio=$(read_input "跳变过滤阈值比例 [0,1]，留空表示自适应" "$(get_method_default "$default_json" "adtk_hbos" "hbos_ratio")")
          ;;
        chatts)
          n_downsample=$(read_input "降采样点数" "$(get_method_default "$default_json" "chatts" "n_downsample")")
          chatts_model_path=$(read_input "ChatTS 模型路径" "$(get_method_default "$default_json" "chatts" "chatts_model_path")")
          chatts_device=$(read_input "ChatTS GPU 设备" "$(get_method_default "$default_json" "chatts" "chatts_device")")
          chatts_use_cache=$(read_input "ChatTS 是否使用 KV cache (true/false/null)" "$(get_method_default "$default_json" "chatts" "chatts_use_cache")")
          chatts_lora_adapter_path=$(read_input "LoRA 微调适配器路径 (留空使用原始模型)" "$(get_method_default "$default_json" "chatts" "chatts_lora_adapter_path")")
          chatts_max_new_tokens=$(read_input "最大生成token数" "$(get_method_default "$default_json" "chatts" "chatts_max_new_tokens")")
          chatts_prompt_template=$(read_input "Prompt模板 (default/detailed/minimal/industrial/english)" "$(get_method_default "$default_json" "chatts" "chatts_prompt_template")")
          chatts_load_in_4bit=$(read_input "4-bit量化 (true/false/auto，8B模型建议auto)" "$(get_method_default "$default_json" "chatts" "chatts_load_in_4bit")")
          ;;
        timer)
          n_downsample=$(read_input "降采样点数" "$(get_method_default "$default_json" "timer" "n_downsample")")
          timer_model_path=$(read_input "Timer 模型路径" "$(get_method_default "$default_json" "timer" "timer_model_path")")
          timer_device=$(read_input "Timer GPU 设备" "$(get_method_default "$default_json" "timer" "timer_device")")
          timer_lookback_length=$(read_input "滚动预测窗口长度" "$(get_method_default "$default_json" "timer" "timer_lookback_length")")
          timer_threshold_k=$(read_input "异常检测阈值系数" "$(get_method_default "$default_json" "timer" "timer_threshold_k")")
          timer_method=$(read_input "残差检测方法 (mad, sigma)" "$(get_method_default "$default_json" "timer" "timer_method")")
          ;;
        *)
          echo "未知方法: $method，使用默认参数"
          ;;
      esac

      params=$(jq -n \
        --arg mode "$mode" \
        --arg method "$method" \
        --argjson n_jobs "$n_jobs" \
        --arg device "$device" \
        --arg config_file "$config_file" \
        --argjson ratio "$ratio" \
        --argjson min_threshold "$min_threshold" \
        --arg downsampler "$downsampler" \
        --argjson n_downsample "$n_downsample" \
        --argjson num_points "$num_points" \
        --argjson threshold "$threshold" \
        --argjson ratio_classify "$ratio_classify" \
        --arg use_trend "$use_trend" \
        --arg use_seasonal "$use_seasonal" \
        --arg use_resid "$use_resid" \
        --arg decompose "$decompose" \
        --arg insert_missing "$insert_missing" \
        --argjson bin_nums "$bin_nums" \
        --arg hbos_ratio "$hbos_ratio" \
        --arg enable_plotting "$enable_plotting" \
        --arg save_mode "$save_mode" \
        --arg data_path "$data_path" \
        --arg label_path "$label_path" \
        --arg fig_path "$fig_path" \
        --arg fig_type "$fig_type" \
        --arg run_heatmap "$run_heatmap" \
        --arg chatts_model_path "$chatts_model_path" \
        --arg chatts_device "$chatts_device" \
        --arg chatts_use_cache "$chatts_use_cache" \
        --arg chatts_lora_adapter_path "$chatts_lora_adapter_path" \
        --argjson chatts_max_new_tokens "$chatts_max_new_tokens" \
        --arg chatts_prompt_template "$chatts_prompt_template" \
        --arg chatts_load_in_4bit "$chatts_load_in_4bit" \
        --arg timer_model_path "$timer_model_path" \
        --arg timer_device "$timer_device" \
        --argjson timer_lookback_length "$timer_lookback_length" \
        --argjson timer_threshold_k "$timer_threshold_k" \
        --arg timer_method "$timer_method" \
        '{
          task_name: $mode,
          method: $method,
          n_jobs: $n_jobs,
          device: $device,
          config_file: $config_file,
          ratio: $ratio,
          min_threshold: $min_threshold,
          downsampler: $downsampler,
          n_downsample: $n_downsample,
          num_points: $num_points,
          threshold: $threshold,
          ratio_classify: $ratio_classify,
          use_trend: ($use_trend == "true"),
          use_seasonal: ($use_seasonal == "true"),
          use_resid: ($use_resid == "true"),
          decompose: ($decompose == "true"),
          insert_missing: ($insert_missing == "true"),
          bin_nums: ($bin_nums | tonumber?),
          hbos_ratio: (if $hbos_ratio == "" or $hbos_ratio == "null" then null else ($hbos_ratio | tonumber) end),
          enable_plotting: ($enable_plotting == "true"),
          save_mode: $save_mode,
          data_path: $data_path,
          label_path: $label_path,
          fig_path: $fig_path,
          fig_type: $fig_type,
          run_heatmap: ($run_heatmap == "true"),
          chatts_model_path: $chatts_model_path,
          chatts_device: $chatts_device,
          chatts_use_cache: $chatts_use_cache,
          chatts_lora_adapter_path: (if $chatts_lora_adapter_path == "" or $chatts_lora_adapter_path == "null" then null else $chatts_lora_adapter_path end),
          chatts_max_new_tokens: $chatts_max_new_tokens,
          chatts_prompt_template: $chatts_prompt_template,
          chatts_load_in_4bit: $chatts_load_in_4bit,
          timer_model_path: $timer_model_path,
          timer_device: $timer_device,
          timer_lookback_length: $timer_lookback_length,
          timer_threshold_k: $timer_threshold_k,
          timer_method: $timer_method
        }')
      ;;
    2)  # 使用默认参数
      params=$(load_params "default_params.json" "$mode")
      ;;
    3)  # 使用上一次参数配置
      latest_params_file=$(find_latest_params "$mode")
      if [ -n "$latest_params_file" ]; then
        params=$(jq '.' "$latest_params_file")
      else
        echo "未找到最近的参数文件，使用默认参数。"
        params=$(load_params "default_params.json" "$mode")
      fi
      ;;
    *)
      echo "无效选项，返回主菜单。"
      return 1
      ;;
  esac

  # 确认参数
  confirm_params "$params" || return 1

  # 执行 Python 脚本
  echo "正在运行 Python 脚本..."

  # 合并 method_params 中当前 method 的参数到顶层
  # 这样 timer_model_path 等嵌套参数可以被正确提取
  current_method=$(echo "$params" | jq -r '.method')
  merged_params=$(echo "$params" | jq -c --arg m "$current_method" '. + (.method_params[$m] // {})')
  
  # 调用 Python 脚本
  echo "正在运行 Python 脚本..."
  python run.py $(json_to_args "$(echo "$merged_params" | jq -c '{path_iotdb, task_name, processing_interval, method, threshold, num_points, n_jobs, data_path, downsampler, n_downsample, decompose, insert_missing, config_file, ratio, ratio_classify, post_filter, use_trend, use_seasonal, use_resid, device, min_threshold, bin_nums, hbos_ratio, save_mode, chatts_model_path, chatts_device, chatts_use_cache, chatts_lora_adapter_path, chatts_max_new_tokens, chatts_prompt_template, chatts_load_in_4bit, timer_model_path, timer_device, timer_lookback_length, timer_threshold_k, timer_method} + {threhold_filter: .threshold_filter}')") || {
    echo "运行 run.py 时发生错误";
    return 1;
  }

  if [ "$task_type" != "detection_only" ]; then
    # 检查是否启用绘图流程
    if [ "$(echo "$params" | jq -r '.enable_plotting')" = "true" ]; then
      echo "正在运行绘图流程..."
      
      # 清洗 methods 中的空格，防止被 argparse 拆成多个参数
      params_clean=$(echo "$params" | jq '.methods |= (if . == null then . else (gsub("\\s+"; "")) end)')
      python combine_data.py $(json_to_args "$(echo "$params_clean" | jq -c '{task_name, methods, data_path, label_path, n_jobs}')") || {
        echo "运行 combine_data.py 时发生错误";
        return 1;
      }

      python save_fig.py $(json_to_args "$(echo "$params" | jq -c '{task_name, fig_type, fig_path, config_file, n_jobs}')") || {
        echo "运行 save_fig.py 时发生错误";
        return 1;
      }

      # 根据配置决定是否运行热力图生成
      if [ "$(echo "$params" | jq -r '.run_heatmap')" = "true" ]; then
        echo "正在运行热力图生成..."
        python heatmap.py $(json_to_args "$(echo "$params" | jq -c '{task_name, method, data_path, n_jobs}')") || {
          echo "运行 heatmap.py 时发生错误";
          return 1;
        }
      else
        echo "跳过热力图生成（run_heatmap = false）"
      fi
      
      echo "绘图流程完成！"
    else
      echo "跳过绘图流程（enable_plotting = false）"
    fi
  else
    echo "跳过绘图流程（仅运行异常检测）"
  fi

}



# 主菜单逻辑
while true; do
  echo "请选择操作："
  echo "1. 获取异常结果"
  echo "2. 删除数据"
  echo "3. 对比分析两种算法结果"
  echo "4. 只运行异常检测（不绘图）"
  echo "5. 只运行绘图流程（需要先有异常检测结果）"
  echo "0. 退出"
  echo -n "请输入选项 [1/2/3/4/5/0]: "

  read main_choice

  case $main_choice in
    1) # 获取异常结果（完整流程）
      echo "获取异常结果（完整流程）："
      echo "1. global"
      echo "2. local"
      echo -n "请输入选项 [1/2]: "
      read sub_choice
      case $sub_choice in
        1)
          echo "正在获取 global 异常结果..."
          run_interactive "global" "full"
          ;;
        2)
          echo "正在获取 local 异常结果..."
          run_interactive "local" "full"
          ;;
        0)
          break
          ;;
        *)
          echo "无效选项，请重新输入。"
          ;;
      esac
      ;;
    2) # 删除数据
      while true; do
        echo "删除数据："
        echo "1. global"
        echo "2. local"
        echo "0. 返回主菜单"
        echo -n "请输入选项 [1/2/0]: "

        read sub_choice
        case $sub_choice in
          1)
            task_name="global"
            echo "正在加载 ${task_name} 的配置参数..."
            latest_params_file=$(find_latest_params "$task_name")
            if [ -n "$latest_params_file" ]; then
              # 从参数文件中提取最新的参数（取最后一个）
              saved_params=$(jq -r ".${task_name}[-1]" "$latest_params_file")
              # 加载默认参数作为基础
              default_params=$(load_params "default_params.json" "$task_name")
              # 合并参数，保存的参数优先
              params=$(echo "$default_params" | jq --argjson saved "$saved_params" '. * $saved')
              echo "加载了最后一次配置文件：$latest_params_file"
            else
              params=$(load_params "default_params.json" "$task_name")
              echo "未找到最后一次配置文件，加载默认配置：default_params.json"
            fi

            # 修正路径提取
            data_path=$(echo "$params" | jq -r '.data_path // empty')
            label_path=$(echo "$params" | jq -r '.label_path // empty')
            fig_path=$(echo "$params" | jq -r '.fig_path // empty')
            
            # 处理 null 值（jq 可能返回字符串 "null"）
            if [[ "$data_path" == "null" ]]; then
              data_path=""
            fi
            if [[ "$label_path" == "null" ]]; then
              label_path=""
            fi
            if [[ "$fig_path" == "null" ]]; then
              fig_path=""
            fi
            
            # 规范化路径：确保以 /opt/results 开头，并添加末尾斜杠（如果需要）
            normalize_path() {
              local path=$1
              if [[ -n "$path" ]]; then
                # 如果路径以 /opt/results 开头，确保末尾有斜杠
                if [[ "$path" =~ ^(/home/share/results|/opt/results) ]]; then
                  # 如果末尾没有斜杠，添加一个
                  if [[ ! "$path" =~ /$ ]]; then
                    echo "${path}/"
                  else
                    echo "$path"
                  fi
                else
                  echo "$path"
                fi
              else
                echo ""
              fi
            }
            
            data_path=$(normalize_path "$data_path")
            label_path=$(normalize_path "$label_path")
            fig_path=$(normalize_path "$fig_path")

            # 调试信息：显示提取的路径
            echo "调试信息："
            echo "data_path: '$data_path'"
            echo "label_path: '$label_path'"
            echo "fig_path: '$fig_path'"

            # 严格检查路径是否为空或不安全
            # label_path 可以为空（某些配置中可能为 null）
            if [[ -z "$data_path" || -z "$fig_path" ]]; then
              echo "错误：必需路径为空，删除操作被终止。"
              echo "data_path 为空: $([[ -z "$data_path" ]] && echo "是" || echo "否")"
              echo "fig_path 为空: $([[ -z "$fig_path" ]] && echo "是" || echo "否")"
              break
            fi
            
            # 安全检查：路径必须以 /opt/results/ 开头
            # label_path 可以为空，所以只检查非空路径
            path_check_failed=false
            if [[ -n "$data_path" && ! "$data_path" =~ ^(/home/share/results/|/opt/results/) ]]; then
              echo "data_path 安全检查失败: $data_path"
              path_check_failed=true
            fi
            if [[ -n "$label_path" && ! "$label_path" =~ ^(/home/share/results/|/opt/results/) ]]; then
              echo "label_path 安全检查失败: $label_path"
              path_check_failed=true
            fi
            if [[ -n "$fig_path" && ! "$fig_path" =~ ^/opt/results/ ]]; then
              echo "fig_path 安全检查失败: $fig_path"
              path_check_failed=true
            fi
            
            if [[ "$path_check_failed" == "true" ]]; then
              echo "错误：路径不安全，删除操作被终止。"
              break
            fi

           

            echo "正在删除路径："
            echo "data_path: $data_path"
            echo "label_path: $label_path (可能为空)"
            echo "fig_path: $fig_path"

            # 删除文件
            if [ -d "$data_path/$task_name" ]; then
              echo "正在删除 ${data_path}/${task_name} 下的所有内容..."
              rm -rf "${data_path}/${task_name}/"* || echo "删除 ${data_path}/${task_name} 内容失败！"
            else
              echo "提示：目录 ${data_path}/${task_name} 不存在或不是目录，跳过。"
            fi

            if [[ -n "$label_path" ]]; then
              if [ -d "$label_path/$task_name" ]; then
                echo "正在删除 ${label_path}/${task_name} 下的所有内容..."
                rm -rf "${label_path}/${task_name}/"* || echo "删除 ${label_path}/${task_name} 内容失败！"
              else
                echo "提示：目录 ${label_path}/${task_name} 不存在或不是目录，跳过。"
              fi
            else
              echo "提示：label_path 为空，跳过删除。"
            fi

            if [ -d "$fig_path/$task_name" ]; then
              echo "正在删除 ${fig_path}/${task_name} 下的所有内容..."
              rm -rf "${fig_path}/${task_name}/"* || echo "删除 ${fig_path}/${task_name} 内容失败！"
            else
              echo "提示：目录 ${fig_path}/${task_name} 不存在或不是目录，跳过。"
            fi
            
            echo "删除操作完成！"

            ;;
          2)
            # 处理 local 模式，逻辑类似
            task_name="local"
            echo "正在加载 ${task_name} 的配置参数..."
            
            latest_params_file=$(find_latest_params "$task_name")
            if [ -n "$latest_params_file" ]; then
              # 从参数文件中提取最新的参数（取最后一个）
              saved_params=$(jq -r ".${task_name}[-1]" "$latest_params_file")
              # 加载默认参数作为基础
              default_params=$(load_params "default_params.json" "$task_name")
              # 合并参数，保存的参数优先
              params=$(echo "$default_params" | jq --argjson saved "$saved_params" '. * $saved')
              echo "加载了最后一次配置文件：$latest_params_file"
            else
              params=$(load_params "default_params.json" "$task_name")
              echo "未找到最后一次配置文件，加载默认配置：default_params.json"
            fi

            # 修正路径提取
            data_path=$(echo "$params" | jq -r '.data_path // empty')
            label_path=$(echo "$params" | jq -r '.label_path // empty')
            fig_path=$(echo "$params" | jq -r '.fig_path // empty')
            
            # 处理 null 值（jq 可能返回字符串 "null"）
            if [[ "$data_path" == "null" ]]; then
              data_path=""
            fi
            if [[ "$label_path" == "null" ]]; then
              label_path=""
            fi
            if [[ "$fig_path" == "null" ]]; then
              fig_path=""
            fi
            
            # 规范化路径：确保以 /opt/results 开头，并添加末尾斜杠（如果需要）
            normalize_path() {
              local path=$1
              if [[ -n "$path" ]]; then
                # 如果路径以 /opt/results 开头，确保末尾有斜杠
                if [[ "$path" =~ ^/opt/results ]]; then
                  # 如果末尾没有斜杠，添加一个
                  if [[ ! "$path" =~ /$ ]]; then
                    echo "${path}/"
                  else
                    echo "$path"
                  fi
                else
                  echo "$path"
                fi
              else
                echo ""
              fi
            }
            
            data_path=$(normalize_path "$data_path")
            label_path=$(normalize_path "$label_path")
            fig_path=$(normalize_path "$fig_path")

            # 调试信息：显示提取的路径
            echo "调试信息："
            echo "data_path: '$data_path'"
            echo "label_path: '$label_path'"
            echo "fig_path: '$fig_path'"

            # 严格检查路径是否为空或不安全
            # label_path 可以为空（某些配置中可能为 null）
            if [[ -z "$data_path" || -z "$fig_path" ]]; then
              echo "错误：必需路径为空，删除操作被终止。"
              echo "data_path 为空: $([[ -z "$data_path" ]] && echo "是" || echo "否")"
              echo "fig_path 为空: $([[ -z "$fig_path" ]] && echo "是" || echo "否")"
              break
            fi
            
            # 安全检查：路径必须以 /opt/results/ 开头
            # label_path 可以为空，所以只检查非空路径
            path_check_failed=false
            if [[ -n "$data_path" && ! "$data_path" =~ ^/opt/results/ ]]; then
              echo "data_path 安全检查失败: $data_path"
              path_check_failed=true
            fi
            if [[ -n "$label_path" && ! "$label_path" =~ ^/opt/results/ ]]; then
              echo "label_path 安全检查失败: $label_path"
              path_check_failed=true
            fi
            if [[ -n "$fig_path" && ! "$fig_path" =~ ^/opt/results/ ]]; then
              echo "fig_path 安全检查失败: $fig_path"
              path_check_failed=true
            fi
            
            if [[ "$path_check_failed" == "true" ]]; then
              echo "错误：路径不安全，删除操作被终止。"
              break
            fi

            echo "正在删除路径："
            echo "data_path: $data_path"
            echo "label_path: $label_path (可能为空)"
            echo "fig_path: $fig_path"

            # 删除文件
            if [ -d "$data_path/$task_name" ]; then
              echo "正在删除 ${data_path}/${task_name} 下的所有内容..."
              rm -rf "${data_path}/${task_name}/"* || echo "删除 ${data_path}/${task_name} 内容失败！"
            else
              echo "提示：目录 ${data_path}/${task_name} 不存在或不是目录，跳过。"
            fi

            if [[ -n "$label_path" ]]; then
              if [ -d "$label_path/$task_name" ]; then
                echo "正在删除 ${label_path}/${task_name} 下的所有内容..."
                rm -rf "${label_path}/${task_name}/"* || echo "删除 ${label_path}/${task_name} 内容失败！"
              else
                echo "提示：目录 ${label_path}/${task_name} 不存在或不是目录，跳过。"
              fi
            else
              echo "提示：label_path 为空，跳过删除。"
            fi

            if [ -d "$fig_path/$task_name" ]; then
              echo "正在删除 ${fig_path}/${task_name} 下的所有内容..."
              rm -rf "${fig_path}/${task_name}/"* || echo "删除 ${fig_path}/${task_name} 内容失败！"
            else
              echo "提示：目录 ${fig_path}/${task_name} 不存在或不是目录，跳过。"
            fi
            
            echo "删除操作完成！"
            ;;
          0)
            break
            ;;
          *)
            echo "无效选项，请重新输入。"
            ;;
        esac
      done
      ;;
    3) # 对比分析两种算法结果
      echo "对比分析两种算法结果："
      echo "1. global 对比分析"
      echo "2. local 对比分析"
      echo -n "请输入选项 [1/2]: "
      read sub_choice
      case $sub_choice in
        1)
          task_name="global"
          echo "正在加载 ${task_name} 的配置参数进行对比分析..."
          ;;
        2)
          task_name="local"
          echo "正在加载 ${task_name} 的配置参数进行对比分析..."
          ;;
        *)
          echo "无效选项，返回主菜单。"
          continue
          ;;
      esac
      
      # 加载配置参数
      latest_params_file=$(find_latest_params "$task_name")
      if [ -n "$latest_params_file" ]; then
        # 从参数文件中提取最新的参数（取最后一个）
        saved_params=$(jq -r ".${task_name}[-1]" "$latest_params_file")
        # 加载默认参数作为基础
        default_params=$(load_params "default_params.json" "$task_name")
        # 合并参数，保存的参数优先
        params=$(echo "$default_params" | jq --argjson saved "$saved_params" '. * $saved')
        echo "加载了配置文件：$latest_params_file"
      else
        # 没有历史配置文件时，使用默认参数
        echo "未找到历史配置文件，使用默认参数。"
        params=$(load_params "default_params.json" "$task_name")
      fi
      
      # 执行对比分析
      echo "正在执行对比分析..."
      
      # 选择对比绘图模式
      echo "请选择对比绘图模式："
      echo "1. downsample - 使用保存的降采样结果对齐"
      echo "2. raw        - 将降采样掩码映射回原始长度对齐"
      echo -n "请输入选项 [1/2] (默认 1): "
      read plot_mode_choice
      case $plot_mode_choice in
        2)
          plot_mode="raw"
          ;;
        *)
          plot_mode="downsample"
          ;;
      esac
      
      # 运行两算法对比分析脚本（使用配置中的 methods 前两个方法）
      python compare_methods.py \
        --task_name "$task_name" \
        --fig_path "$(echo "$params" | jq -r '.fig_path')" \
        --data_path "$(echo "$params" | jq -r '.data_path')" \
        --n_jobs "$(echo "$params" | jq -r '.n_jobs')" \
        --methods "$(echo "$params" | jq -r '.methods')" \
        --mode "$plot_mode" || {
        echo "运行两算法对比分析时发生错误";
        continue;
      }
      
      echo "对比分析完成！"
      ;;
    4) # 只运行异常检测（不绘图）
      echo "只运行异常检测（不绘图）："
      echo "1. global"
      echo "2. local"
      echo -n "请输入选项 [1/2]: "
      read sub_choice
      case $sub_choice in
        1)
          echo "正在运行 global 异常检测..."
          run_interactive "global" "detection_only"
          ;;
        2)
          echo "正在运行 local 异常检测..."
          run_interactive "local" "detection_only"
          ;;
        0)
          break
          ;;
        *)
          echo "无效选项，请重新输入。"
          ;;
      esac
      ;;
    5) # 只运行绘图流程
      echo "只运行绘图流程（需要先有异常检测结果）："
      echo "1. global"
      echo "2. local"
      echo -n "请输入选项 [1/2]: "
      read sub_choice
      case $sub_choice in
        1)
          task_name="global"
          echo "正在加载 ${task_name} 的配置参数进行绘图..."
          ;;
        2)
          task_name="local"
          echo "正在加载 ${task_name} 的配置参数进行绘图..."
          ;;
        *)
          echo "无效选项，返回主菜单。"
          continue
          ;;
      esac
      
      # 加载配置参数
      latest_params_file=$(find_latest_params "$task_name")
      if [ -n "$latest_params_file" ]; then
        # 从参数文件中提取最新的参数（取最后一个）
        saved_params=$(jq -r ".${task_name}[-1]" "$latest_params_file")
        # 加载默认参数作为基础
        default_params=$(load_params "default_params.json" "$task_name")
        # 合并参数，保存的参数优先
        params=$(echo "$default_params" | jq --argjson saved "$saved_params" '. * $saved')
        echo "加载了配置文件：$latest_params_file"
      else
        # 没有历史配置文件时，使用默认参数
        echo "未找到历史配置文件，使用默认参数。"
        params=$(load_params "default_params.json" "$task_name")
      fi
      
      # 选择图表类型
      echo "请选择图表类型："
      echo "1. focus - 局部聚焦图（基于算法标签的详细视图）"
      echo "2. combined - 算法标签 vs 人工标签对比图"
      echo "3. normal - 普通视图（单算法结果图）"
      echo -n "请输入选项 [1/2/3] (默认使用配置中的 fig_type): "
      read fig_type_choice
      case $fig_type_choice in
        1)
          fig_type="focus"
          ;;
        2)
          fig_type="combined"
          ;;
        3)
          fig_type="normal"
          ;;
        "")
          # 使用配置中的默认值
          fig_type=$(echo "$params" | jq -r '.fig_type // "focus"')
          echo "使用配置中的图表类型: $fig_type"
          ;;
        *)
          echo "无效选项，使用配置中的默认图表类型。"
          fig_type=$(echo "$params" | jq -r '.fig_type // "focus"')
          ;;
      esac
      
      # 更新参数中的 fig_type
      params=$(echo "$params" | jq --arg fig_type "$fig_type" '.fig_type = $fig_type')
      
      # 执行绘图流程
      echo "正在执行绘图流程（图表类型: $fig_type）..."
      
      # 根据图表类型决定是否需要运行 combine_data.py
      if [ "$fig_type" = "combined" ]; then
        # combined 类型：异常标签都来自算法结果CSV，区别仅在于原始数据来源
        echo "combined 类型支持两种原始数据来源："
        echo "1. 从算法结果CSV读取（可能是降采样数据，更快，无需IoTDB）"
        echo "2. 从IoTDB读取全量原始数据（更准确，但较慢）"
        echo -n "请选择模式 [1/2] (默认: 1): "
        read combine_mode
        if [ -z "$combine_mode" ]; then
          combine_mode="1"
        fi
        
        if [ "$combine_mode" = "2" ]; then
          echo "将从IoTDB读取全量原始数据..."
        else
          echo "将直接从算法结果CSV读取数据..."
        fi
      elif [ "$fig_type" = "focus" ]; then
        # focus 类型需要运行 combine_data.py
        # 运行数据合并脚本
        # 清洗 methods 中的空格，防止被 argparse 拆成多个参数
        params_clean=$(echo "$params" | jq '.methods |= (if . == null then . else (gsub("\\s+"; "")) end)')
        python combine_data.py $(json_to_args "$(echo "$params_clean" | jq -c '{task_name, methods, data_path, label_path, n_jobs}')") || {
          echo "运行 combine_data.py 时发生错误";
          continue;
        }
      else
        # normal 类型不需要运行 combine_data.py
        echo "normal 类型不生成标签数据，直接使用算法结果 CSV 绘图..."
      fi

      # 运行绘图脚本
      # 根据图表类型选择不同的参数
      if [ "$fig_type" = "normal" ]; then
        # 选择具体算法方法（默认使用当前配置中的 method）
        current_method=$(echo "$params" | jq -r '.method // "stl_wavelet"')
        echo -n "请输入算法方法 (如: stl_wavelet, adtk_hbos) [默认: $current_method]: "
        read method_choice
        if [ -z "$method_choice" ]; then
          method_choice="$current_method"
        fi
        # 更新参数中的 method
        params=$(echo "$params" | jq --arg method "$method_choice" '.method = $method')

        # normal 类型需要 data_path 和 method
        python save_fig.py $(json_to_args "$(echo "$params" | jq -c '{task_name, fig_type, fig_path, config_file, data_path, method, n_jobs}')") || {
          echo "运行 save_fig.py 时发生错误";
          continue;
        }
      elif [ "$fig_type" = "combined" ]; then
        # 选择对比的两种算法（默认使用当前配置中的 methods）
        current_methods=$(echo "$params" | jq -r '.methods // "stl_wavelet,adtk_hbos"')
        echo -n "请输入要对比的两种算法(逗号分隔，如: stl_wavelet,adtk_hbos) [默认: $current_methods]: "
        read methods_choice
        if [ -z "$methods_choice" ]; then
          methods_choice="$current_methods"
        fi
        # 更新参数中的 methods
        params=$(echo "$params" | jq --arg methods "$methods_choice" '.methods = $methods')

        # 清洗 methods 字符串，去除空白字符，避免被 argparse 拆分
        params_clean=$(echo "$params" | jq '.methods |= (if . == null then . else (gsub("\\s+"; "")) end)')
        params="$params_clean"

        # 询问人工标签文件路径（可选）
        echo -n "请输入人工标签文件路径（可选，留空则跳过）[默认: 无]: "
        read manual_label_path_choice
        if [ -n "$manual_label_path_choice" ]; then
          params=$(echo "$params" | jq --arg manual_label_path "$manual_label_path_choice" '.manual_label_path = $manual_label_path')
        fi

        # 根据 combine_mode 设置 use_direct_data 参数
        # 注意：由于 argparse 使用 action='store_true'，只有值为 true 时才传递参数
        if [ "$combine_mode" = "1" ]; then
          params=$(echo "$params" | jq '.use_direct_data = true')
        else
          # 模式2时，不设置 use_direct_data（或设置为 false），这样就不会传递该参数
          params=$(echo "$params" | jq 'del(.use_direct_data)')
        fi

        # combined 类型需要 methods、data_path 和 label_path
        # data_path 用于直接从算法结果CSV文件读取数据
        # label_path 用于读取已生成的标签文件（如果存在）
        python save_fig.py $(json_to_args "$(echo "$params" | jq -c '{task_name, fig_type, fig_path, config_file, label_path, methods, data_path, n_jobs, manual_label_path, use_direct_data}')") || {
          echo "运行 save_fig.py 时发生错误";
          continue;
        }
      else
        # focus 类型只需要基本参数
        python save_fig.py $(json_to_args "$(echo "$params" | jq -c '{task_name, fig_type, fig_path, config_file, n_jobs}')") || {
          echo "运行 save_fig.py 时发生错误";
          continue;
        }
      fi

      # 根据配置决定是否运行热力图生成
      if [ "$(echo "$params" | jq -r '.run_heatmap')" = "true" ]; then
        echo "正在运行热力图生成..."
        python heatmap.py $(json_to_args "$(echo "$params" | jq -c '{task_name, method, data_path, n_jobs}')") || {
          echo "运行 heatmap.py 时发生错误";
          continue;
        }
      else
        echo "跳过热力图生成（run_heatmap = false）"
      fi
      
      echo "绘图流程完成！"
      ;;
    0)
      echo "退出脚本。"
      exit 0
      ;;
    *)
      echo "无效选项，请重新输入。"
      ;;
  esac
done


