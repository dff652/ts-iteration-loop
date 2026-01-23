/**
 * Data transformation utilities for Time Series Annotator V2
 * Handles conversion between backend API format and D3 expected format
 */

const { DateTime } = require("luxon");

/**
 * Transform backend API data to D3 chart format
 * IMPORTANT: Keep time as ISO string - LabelerD3.js type() function will convert it
 * 
 * @param {Array} apiData - Data from backend API
 * @returns {Array} - Data formatted for D3
 */
export function transformForD3(apiData) {
    return apiData.map((d, idx) => ({
        id: idx.toString(),
        val: parseFloat(d.val),
        time: d.time,  // Keep as ISO string, D3 type() will convert
        series: d.series || 'value',
        label: d.label || ''
    }));
}

/**
 * Transform data from local CSV upload (time is already DateTime object)
 * This handles the case where file upload already converted to DateTime
 * 
 * @param {Array} csvData - Parsed CSV data with DateTime objects
 * @returns {Array} - Data formatted for D3
 */
export function transformFromLocalCSV(csvData) {
    return csvData.map((d, idx) => ({
        id: idx.toString(),
        val: parseFloat(d.val),
        time: d.time,  // Already DateTime object from CSV parsing
        series: d.series || 'value',
        label: d.label || ''
    }));
}

/**
 * Check if time value is already a DateTime object
 * 
 * @param {*} time - Time value to check
 * @returns {boolean}
 */
export function isDateTime(time) {
    return time && typeof time === 'object' && typeof time.toISO === 'function';
}

/**
 * Ensure time is ISO string format (for API data consistency)
 * 
 * @param {*} time - Time value (string or DateTime)
 * @returns {string} - ISO string
 */
export function ensureISOString(time) {
    if (isDateTime(time)) {
        return time.toISO();
    }
    return time;
}

/**
 * Build annotation export format matching the target JSON structure
 * 
 * @param {Object} options - Annotation options
 * @returns {Object} - Formatted annotation
 */
export function buildAnnotationExport({
    overallLabels,
    localLabels,
    indexRange,
    input,
    output,
    filename
}) {
    const annotation = {
        categories: {},
        local_change: {
            name: "局部变化",
            categories: {}
        }
    };

    // Build overall categories
    Object.entries(overallLabels).forEach(([catId, label]) => {
        if (label) {
            annotation.categories[catId] = {
                name: getCategoryName(catId),
                labels: [label]
            };
        }
    });

    // Build local change categories
    localLabels.forEach(label => {
        const categoryId = label.categoryId || 'custom';
        if (!annotation.local_change.categories[categoryId]) {
            annotation.local_change.categories[categoryId] = {
                name: label.categoryName || categoryId,
                labels: []
            };
        }
        annotation.local_change.categories[categoryId].labels.push({
            id: label.id,
            text: label.text,
            color: label.color,
            index: indexRange,
            input: input || '',
            output: output || ''
        });
    });

    return annotation;
}

/**
 * Get display name for category ID
 */
function getCategoryName(catId) {
    const names = {
        trend: '趋势',
        seasonal: '周期性',
        frequency: '频率',
        noise: '噪声'
    };
    return names[catId] || catId;
}
