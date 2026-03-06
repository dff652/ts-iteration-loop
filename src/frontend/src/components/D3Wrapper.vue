<template>
  <div class="d3-wrapper">
    <!-- Hover Info -->
    <div id="hoverbox">
      <div id="hoverinfo" class="hover-card" style="display: none;">
        <div>时间: {{ hoverinfo.time }}</div>
        <div>数值: {{ hoverinfo.val }}</div>
        <div>标签: {{ hoverinfo.label }}</div>
      </div>
    </div>

    <!-- D3 Chart Container -->
    <div id="maindiv" ref="maindiv"></div>

    <!-- Loading -->
    <div v-if="loading" class="loader"></div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'

// D3 drawLabeler is imported as JS (non-TS)
import { drawLabeler } from '@/d3/LabelerD3.js'

// ─── Props ───────────────────────────────────────────────
interface Props {
  csvData: any[]
  filename: string
  headerStr: string
  seriesList: string[]
  labelList: { name: string; color: string }[]
  selectedLabel: string
}

const props = defineProps<Props>()

// ─── Emits ───────────────────────────────────────────────
const emit = defineEmits<{
  (e: 'chart-selection', start: number, end: number): void
  (e: 'hover-update', info: { time: string; val: string; label: string }): void
  (e: 'edit-axis'): void
  (e: 'label-change', label: string): void
  (e: 'selection-range'): void
}>()

// ─── State ───────────────────────────────────────────────
const maindiv = ref<HTMLDivElement>()
const loading = ref(false)
const hoverinfo = reactive({ time: '', val: '', label: '' })

// plottingApp namespace — shared with D3
let plottingApp: any = {}

// ─── Exposed API for parent to call D3 functions ────────
function changeSeries(series: string) {
  plottingApp.changeSeries?.(series)
}

function changeReference(series: string) {
  plottingApp.changeReference?.(series)
}

function changeLabel(label: string) {
  plottingApp.changeLabelSelect?.(label)
}

function resetView() {
  plottingApp.resetView?.()
}

function clearAllLabels() {
  plottingApp.clearAllLabels?.()
}

function setSelection(startIdx: number, endIdx: number) {
  plottingApp.setSelection?.(startIdx, endIdx)
}

function clearSelection() {
  plottingApp.clearSelection?.()
}

function exportCSV() {
  plottingApp.exportCSV?.()
}

function triggerReplot() {
  plottingApp.triggerReplot?.()
}

function triggerRecolor() {
  plottingApp.triggerRecolor?.()
}

function getAllData() {
  return plottingApp.allData || []
}

function getData() {
  return plottingApp.data || []
}

function getSelection() {
  return plottingApp.selection || null
}

function getAxisBounds() {
  return plottingApp.axisBounds || {}
}

function setAxisBounds(series: string, bounds: number[]) {
  if (plottingApp.axisBounds) {
    plottingApp.axisBounds[series] = bounds.slice()
    plottingApp.triggerReplot?.()
  }
}

function setLabelColor(color: string) {
  if (plottingApp) {
    plottingApp.labelColor = color
  }
}

function setPreventBrushSearch(val: boolean) {
  if (plottingApp) {
    plottingApp.preventBrushSearch = val
  }
}

function setIsEditing(val: boolean) {
  if (plottingApp) {
    plottingApp.isEditing = val
  }
}

defineExpose({
  changeSeries,
  changeReference,
  changeLabel,
  resetView,
  clearAllLabels,
  setSelection,
  clearSelection,
  exportCSV,
  triggerReplot,
  triggerRecolor,
  getAllData,
  getData,
  getSelection,
  getAxisBounds,
  setAxisBounds,
  setLabelColor,
  setPreventBrushSearch,
  setIsEditing,
})

// ─── Watch selectedLabel → push to D3 ───────────────────
watch(() => props.selectedLabel, (newLabel) => {
  if (plottingApp.changeLabelSelect) {
    plottingApp.changeLabelSelect(newLabel)
  } else {
    plottingApp.selectedLabel = newLabel
  }
})

// ─── Watch labelList → push to D3 ───────────────────
watch(() => props.labelList, (newVal) => {
  if (plottingApp) {
    plottingApp.labelList = [...newVal].sort((a, b) => a.name.localeCompare(b.name))
    if (plottingApp.triggerRecolor) {
      plottingApp.triggerRecolor()
    }
  }
}, { deep: true })

// ─── Lifecycle ──────────────────────────────────────────
onMounted(() => {
  if (props.csvData && props.csvData.length > 0) {
    initD3()
  }
})

watch(() => props.csvData, (newData) => {
  if (newData && newData.length > 0) {
    // Destroy old chart, re-init
    destroyD3()
    nextTick(() => initD3())
  }
})

onBeforeUnmount(() => {
  destroyD3()
})

function initD3() {
  if (!maindiv.value) return

  loading.value = true

  // Clear previous content
  maindiv.value.innerHTML = ''
  maindiv.value.innerHTML = '<div class="loader"></div>'

  // Build plottingApp namespace
  plottingApp = {}
  plottingApp.headerStr = props.headerStr
  plottingApp.filename = props.filename
  plottingApp.csvData = props.csvData
  plottingApp.seriesList = props.seriesList
  plottingApp.labelList = props.labelList.sort((a, b) => a.name.localeCompare(b.name))
  plottingApp.selectedLabel = props.selectedLabel

  // Set up callbacks (replaces window.vueApp + jQuery hidden buttons)
  plottingApp.onChartSelection = (start: number, end: number) => {
    emit('chart-selection', start, end)
  }
  plottingApp.onHoverUpdate = () => {
    hoverinfo.time = plottingApp.hoverinfo?.time || ''
    hoverinfo.val = plottingApp.hoverinfo?.val || ''
    hoverinfo.label = plottingApp.hoverinfo?.label || ''
    emit('hover-update', { ...hoverinfo })
  }
  plottingApp.onEditAxis = () => {
    emit('edit-axis')
  }
  plottingApp.onLabelChange = () => {
    emit('label-change', plottingApp.selectedLabel)
  }
  plottingApp.onSelectionRange = () => {
    emit('selection-range')
  }

  // Draw D3 chart
  drawLabeler(plottingApp)
  loading.value = false
}

function destroyD3() {
  // Clean up keyboard listeners
  import('d3').then(d3 => {
    d3.select(window).on('keydown.labeler', null)
    d3.select(window).on('keyup.labeler', null)
  })

  // Clear DOM
  if (maindiv.value) {
    maindiv.value.innerHTML = ''
  }
  plottingApp = {}
}
</script>

<style scoped>
.d3-wrapper {
  position: relative;
  width: 100%;
}

#maindiv {
  text-align: left;
  width: 100%;
}

#hoverbox {
  position: relative;
  float: right;
  z-index: 5;
}

#hoverinfo {
  position: absolute;
  text-align: left;
  padding: 10px;
  width: 220px;
  border: 1px solid var(--el-border-color, #ddd);
  border-radius: 4px;
  background: var(--el-bg-color, #fff);
  top: 10px;
  right: 30px;
  font-size: 13px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  z-index: 10;
}

.loader {
  position: absolute;
  left: 45%;
  top: 25%;
  border: 8px solid #f3f3f3;
  border-top: 8px solid #3498db;
  border-radius: 50%;
  width: 60px;
  height: 60px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>

<style>
/* D3 chart global styles (not scoped because D3 creates elements dynamically) */
#maindiv svg {
  font: 10px sans-serif;
  display: block;
  overflow: auto;
}

#maindiv .line {
  fill: none;
  stroke: black;
  stroke-width: 1.5px;
  clip-path: url(#clip);
  pointer-events: none;
}

#maindiv .point {
  fill: black;
  stroke: none;
  clip-path: url(#clip);
}

#maindiv .axis path,
#maindiv .axis line {
  fill: none;
  stroke: #000;
  shape-rendering: crispEdges;
}

#maindiv .main_brush .extent,
#maindiv .context_brush .extent {
  stroke: #fff;
  fill-opacity: .125;
  shape-rendering: crispEdges;
}

#maindiv .editBtn:hover {
  fill: #E3E3E3;
}

#maindiv #secondary_line {
  stroke: grey;
}

#maindiv .chartText {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  font-size: 1rem;
  font-weight: 400;
  line-height: 1.5;
}

#maindiv #chartTitle {
  font-size: 1.4rem !important;
}
</style>
