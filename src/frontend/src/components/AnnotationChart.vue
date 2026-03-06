<template>
  <div class="chart-wrapper">
    <!-- 模式切换工具栏 -->
    <div class="chart-toolbar">
      <el-radio-group v-model="interactionMode" size="small" @change="onModeChange">
        <el-radio-button value="brush">✏️ 标注模式</el-radio-button>
        <el-radio-button value="zoom">🔍 缩放模式</el-radio-button>
      </el-radio-group>
      <el-button size="small" @click="resetZoom">重置视图</el-button>
      <span class="mode-hint" v-if="interactionMode === 'brush'">
        在图表上拖动以选择标注区间
      </span>
      <span class="mode-hint" v-else>
        滚轮缩放 · 拖动平移
      </span>
    </div>
    <div ref="chartContainer" class="annotation-chart" />
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted, computed, nextTick } from 'vue'
import * as echarts from 'echarts'

// ==================== Props & Events ====================
export interface DataPoint {
  idx: number
  val: number
  label: string
}

export interface AnnotationSegment {
  start: number
  end: number
  label: { id: string; text: string; color: string }
}

export interface AnnotationItem {
  id: string
  label: { id: string; text: string; color: string }
  segments: AnnotationSegment[]
  local_change?: Record<string, string>
}

const props = defineProps<{
  data: DataPoint[]
  seriesName: string
  annotations: AnnotationItem[]
  selectedLabel: { id: string; text: string; color: string } | null
}>()

const emit = defineEmits<{
  (e: 'brush-select', range: { start: number; end: number }): void
  (e: 'point-click', point: DataPoint): void
}>()

// ==================== State ====================
const chartContainer = ref<HTMLDivElement | null>(null)
const interactionMode = ref<'brush' | 'zoom'>('brush')
let chart: echarts.ECharts | null = null

// 构建 x 轴数据（用原始索引）
const xData = computed(() => props.data.map(d => d.idx))
const yData = computed(() => props.data.map(d => d.val))

// ==================== Mark Areas from Annotations ====================
function buildMarkAreas(): Array<[Record<string, unknown>, Record<string, unknown>]> {
  const areas: Array<[Record<string, unknown>, Record<string, unknown>]> = []
  for (const ann of props.annotations) {
    for (const seg of ann.segments) {
      // 找到 xData 中最接近的索引位置
      const startPos = xData.value.indexOf(seg.start)
      const endPos = xData.value.indexOf(seg.end)
      if (startPos === -1 || endPos === -1) continue
      areas.push([
        {
          xAxis: xData.value[startPos],
          itemStyle: { color: ann.label.color + '35' },
          label: { show: true, formatter: ann.label.text, position: 'insideTop', color: ann.label.color, fontSize: 10 },
        },
        { xAxis: xData.value[endPos] },
      ])
    }
  }
  return areas
}

// ==================== Chart Option Builder ====================
function buildOption(): echarts.EChartsOption {
  const isBrush = interactionMode.value === 'brush'

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(30,30,50,0.9)',
      borderColor: 'rgba(255,255,255,0.1)',
      textStyle: { color: '#e0e0e0', fontSize: 12 },
      axisPointer: { type: 'cross', lineStyle: { color: 'rgba(200,120,60,0.3)' } },
      formatter: (params: unknown) => {
        const p = Array.isArray(params) ? params[0] : params
        const d = p as { dataIndex: number; value: number }
        const pt = props.data[d.dataIndex]
        if (!pt) return ''
        return `<b>索引:</b> ${pt.idx}<br/><b>值:</b> ${pt.val.toFixed(4)}${pt.label ? '<br/><b>标签:</b> ' + pt.label : ''}`
      },
    },
    grid: {
      left: 60,
      right: 30,
      top: 40,
      bottom: 80,
    },
    xAxis: {
      type: 'category',
      data: xData.value,
      axisLine: { lineStyle: { color: '#555' } },
      axisLabel: { color: '#999', fontSize: 11 },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: '#555' } },
      axisLabel: { color: '#999', fontSize: 11 },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } },
    },
    // 底部 slider dataZoom 始终显示
    dataZoom: [
      {
        type: 'slider',
        start: 0,
        end: 100,
        height: 24,
        bottom: 10,
        borderColor: 'rgba(255,255,255,0.1)',
        backgroundColor: 'rgba(255,255,255,0.03)',
        fillerColor: 'rgba(200,120,60,0.15)',
        textStyle: { color: '#999' },
        handleStyle: { color: '#c87830' },
      },
      // inside zoom 仅在 zoom 模式启用
      ...(isBrush ? [] : [{
        type: 'inside' as const,
        start: 0,
        end: 100,
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
      }]),
    ],
    series: [
      {
        name: props.seriesName,
        type: 'line',
        data: yData.value,
        smooth: false,
        symbol: 'none',
        lineStyle: { width: 1.2, color: '#c87830' },
        areaStyle: { color: 'rgba(200,120,60,0.06)' },
        sampling: 'lttb',
        markArea: {
          silent: true,
          data: buildMarkAreas(),
        },
      },
    ],
  }
}

// ==================== Brush 模式处理 ====================
let brushStartX: number | null = null
let brushEndX: number | null = null
let brushOverlay: HTMLDivElement | null = null

function enableBrushMode(): void {
  if (!chartContainer.value) return
  const container = chartContainer.value

  // 创建覆盖层接收鼠标事件
  cleanupBrushOverlay()
  brushOverlay = document.createElement('div')
  brushOverlay.className = 'brush-overlay'
  brushOverlay.style.cssText = 'position:absolute;top:40px;left:60px;right:30px;bottom:80px;cursor:crosshair;z-index:10;'
  container.style.position = 'relative'
  container.appendChild(brushOverlay)

  let selectionDiv: HTMLDivElement | null = null

  brushOverlay.addEventListener('mousedown', (e: MouseEvent) => {
    if (!chart) return
    const rect = brushOverlay!.getBoundingClientRect()
    brushStartX = e.clientX - rect.left
    brushEndX = brushStartX

    // 创建选择框
    selectionDiv = document.createElement('div')
    selectionDiv.className = 'brush-selection'
    selectionDiv.style.cssText = `position:absolute;top:0;bottom:0;background:rgba(200,120,60,0.2);border:1px solid rgba(200,120,60,0.6);pointer-events:none;`
    selectionDiv.style.left = brushStartX + 'px'
    selectionDiv.style.width = '0px'
    brushOverlay!.appendChild(selectionDiv)
  })

  brushOverlay.addEventListener('mousemove', (e: MouseEvent) => {
    if (brushStartX === null || !selectionDiv || !brushOverlay) return
    const rect = brushOverlay.getBoundingClientRect()
    brushEndX = e.clientX - rect.left
    const left = Math.min(brushStartX, brushEndX)
    const width = Math.abs(brushEndX - brushStartX)
    selectionDiv.style.left = left + 'px'
    selectionDiv.style.width = width + 'px'
  })

  brushOverlay.addEventListener('mouseup', () => {
    if (brushStartX === null || brushEndX === null || !chart || !brushOverlay) {
      brushStartX = null
      brushEndX = null
      return
    }
    const width = Math.abs(brushEndX - brushStartX)
    if (width < 5) {
      // 太小，忽略
      brushStartX = null
      brushEndX = null
      if (selectionDiv) { selectionDiv.remove(); selectionDiv = null }
      return
    }

    // 将像素坐标转换为数据索引
    const overlayRect = brushOverlay.getBoundingClientRect()
    const overlayWidth = overlayRect.width
    const pixelLeft = Math.min(brushStartX, brushEndX)
    const pixelRight = Math.max(brushStartX, brushEndX)

    // 获取当前 dataZoom 范围
    const option = chart.getOption() as { dataZoom?: Array<{ start?: number; end?: number }> }
    const zoom = option.dataZoom?.[0] || { start: 0, end: 100 }
    const zoomStart = (zoom.start || 0) / 100
    const zoomEnd = (zoom.end || 100) / 100
    const totalLen = xData.value.length
    const visibleStart = Math.floor(zoomStart * totalLen)
    const visibleEnd = Math.ceil(zoomEnd * totalLen)
    const visibleLen = visibleEnd - visibleStart

    // 像素比例 → 数据索引
    const ratioLeft = pixelLeft / overlayWidth
    const ratioRight = pixelRight / overlayWidth
    const dataIdxStart = visibleStart + Math.floor(ratioLeft * visibleLen)
    const dataIdxEnd = visibleStart + Math.floor(ratioRight * visibleLen)

    const startVal = xData.value[Math.max(0, Math.min(dataIdxStart, totalLen - 1))]
    const endVal = xData.value[Math.max(0, Math.min(dataIdxEnd, totalLen - 1))]

    if (startVal !== undefined && endVal !== undefined) {
      emit('brush-select', { start: startVal, end: endVal })
    }

    // 清理选择框
    brushStartX = null
    brushEndX = null
    if (selectionDiv) { selectionDiv.remove(); selectionDiv = null }
  })
}

function cleanupBrushOverlay(): void {
  if (brushOverlay) {
    brushOverlay.remove()
    brushOverlay = null
  }
  brushStartX = null
  brushEndX = null
}

function disableBrushMode(): void {
  cleanupBrushOverlay()
}

// ==================== Mode Change ====================
function onModeChange(): void {
  if (interactionMode.value === 'brush') {
    enableBrushMode()
  } else {
    disableBrushMode()
  }
  updateChart()
}

function resetZoom(): void {
  chart?.dispatchAction({ type: 'dataZoom', start: 0, end: 100 })
}

// ==================== Chart Update ====================
function updateChart(): void {
  if (!chart) return
  chart.setOption(buildOption(), true)
}

function handleResize(): void {
  chart?.resize()
  // 重建 brush overlay 如果在 brush 模式
  if (interactionMode.value === 'brush') {
    nextTick(() => enableBrushMode())
  }
}

onMounted(() => {
  if (!chartContainer.value) return
  chart = echarts.init(chartContainer.value, 'dark')
  updateChart()

  chart.on('click', (params) => {
    if (params.dataIndex !== undefined) {
      if (props.data && props.data[params.dataIndex]) {
        emit('point-click', props.data[params.dataIndex] as DataPoint)
      }
    }
  })

  window.addEventListener('resize', handleResize)

  // 默认进入标注模式
  nextTick(() => enableBrushMode())
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  cleanupBrushOverlay()
  chart?.dispose()
  chart = null
})

watch([() => props.data, () => props.annotations], () => {
  updateChart()
  if (interactionMode.value === 'brush') {
    nextTick(() => enableBrushMode())
  }
}, { deep: true })
</script>

<style scoped>
.chart-wrapper {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.chart-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 4px 8px;
  flex-shrink: 0;
}

.mode-hint {
  font-size: 11px;
  color: #888;
  margin-left: auto;
}

.annotation-chart {
  flex: 1;
  min-height: 350px;
}
</style>

<style>
/* 全局样式 - brush overlay */
.brush-overlay {
  user-select: none;
}
.brush-selection {
  border-radius: 2px;
}
</style>
