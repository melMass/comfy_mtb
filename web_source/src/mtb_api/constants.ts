/** Color for API-marked nodes (refined blue) */
export const API_COLOR = '#4f6ef7'

/** Color for API output nodes (warm gold) */
export const OUTPUT_COLOR = '#f7b84f'

/** CSS custom properties for API styling */
export const API_CSS_VARS = `
:root {
  --mtb-api-color: ${API_COLOR};
  --mtb-api-output-color: ${OUTPUT_COLOR};
  --mtb-api-radius: 8px;
}
`
