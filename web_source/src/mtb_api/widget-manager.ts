/**
 * API Settings Widget Manager
 * Manages DOM widgets for configuring API inputs on nodes
 */

import type { IWidget, LGraphCanvas } from '@comfyorg/litegraph'
import { $el } from '@/scripts/ui'
import * as shared from '@mtb/shared'
import { API_COLOR, OUTPUT_COLOR } from './constants'
import { API_INPUT_TYPES, type APIInputType, type APINodeSettings, type MTBNode } from './types'
import { notifyAPIChanged } from './panel.svelte'
import CSS from './api_nodes.css?inline'

declare const LiteGraph: {
  NODE_TITLE_HEIGHT: number
  NODE_COLLAPSED_RADIUS: number
}

/**
 * Manages the creation and lifecycle of API settings widgets on nodes
 */
export class APISettingsWidgetManager {
  /**
   * Creates the API settings DOM widget for a node
   */
  createAPISettingsWidget(node: MTBNode, force?: boolean): void {
    const existingWidget = node.widgets?.find((w) => w.name === 'apiSettings')

    if (!existingWidget || force) {
      if (existingWidget) {
        this.removeAPISettingsWidget(node, existingWidget)
      }

      const element = document.createElement('div')
      element.classList.add(
        node.properties.mtb_api?.isAPIOutput
          ? 'mtb_api_output'
          : 'mtb_api_settings',
      )

      if (node.widgets) {
        const otherWidgets = node.widgets.filter(
          (w) => w.name !== 'apiSettings',
        )
        for (const w of otherWidgets) {
          const section = this.createWidgetSection(node, w)
          if (section) {
            element.appendChild(section)
          }
        }
      }

      this.createNodeSettingsSection(node, element)

      // @ts-expect-error - addDOMWidget is a ComfyUI extension
      node.addDOMWidget('apiSettings', 'API_SETTINGS', element, {
        hideOnZoom: false,
        getHeight: () => element.children.length * 80,
      })
    }
  }

  /**
   * Creates the node-level settings section with export toggle
   */
  createNodeSettingsSection(node: MTBNode, element: HTMLElement): void {
    const includeInExport = node.properties.mtb_api?.includeInExport !== false

    const nodeSettings = $el('div.mtb_api_node_settings', {}, [
      $el('span.mtb_api_node_settings_label', { textContent: 'Include in API export' }),
      $el('label.mtb_api_toggle', {}, [
        $el('input', {
          type: 'checkbox',
          checked: includeInExport,
          onchange: (e: Event) => {
            const checked = (e.target as HTMLInputElement).checked
            this.applySettings(node, { includeInExport: checked })
          },
        }),
        $el('span.mtb_api_toggle_track'),
      ]),
    ])
    element.appendChild(nodeSettings)
  }

  /**
   * Applies new settings to the node, merging with existing
   */
  applySettings(
    node: MTBNode,
    newSettings: Partial<APINodeSettings>,
  ): APINodeSettings {
    const settings = node.properties.mtb_api || {}
    const result = shared.deepMerge(settings, newSettings) as APINodeSettings

    node.setProperty('mtb_api', result)

    return result
  }

  /**
   * Ensures widgets are created/removed based on node state
   */
  ensureWidgets(node: MTBNode, force?: boolean): void {
    const apiSettings = node.widgets?.find((w) => w.name === 'apiSettings')
    if (node.properties.useAPI) {
      this.createAPISettingsWidget(node, force)
    } else {
      this.removeAPISettingsWidget(node, apiSettings)
    }
  }

  /**
   * Removes the API settings widget from a node
   */
  removeAPISettingsWidget(node: MTBNode, apiSettings?: IWidget): void {
    if (apiSettings) {
      ;(apiSettings as IWidget & { onRemoved?: () => void }).onRemoved?.()
      ;(apiSettings as IWidget & { onRemove?: () => void }).onRemove?.()
      node.widgets = node.widgets?.filter((w) => w.name !== 'apiSettings')
    }
  }

  /**
   * Ensures the CSS styles are injected into the document
   */
  updateStyleElement(): void {
    let styleElement = document.getElementById('mtb-api-widget-styles')
    if (!styleElement) {
      styleElement = document.createElement('style')
      styleElement.id = 'mtb-api-widget-styles'
      document.head.appendChild(styleElement)
      styleElement.textContent = CSS
    }
  }

  /**
   * Converts a ComfyUI widget type to an API input type
   */
  apiTypeFromComfyType(comfyType: string): APIInputType {
    switch (comfyType) {
      case 'number':
        return 'NUMBER'
      case 'text':
      case 'string':
      case 'customtext':
        return 'STRING'
      case 'combo':
        return 'COMBO'
      case 'toggle':
      case 'boolean':
        return 'BOOLEAN'
      default:
        console.log('UNHANDLED widget type:', comfyType, '- defaulting to STRING')
        return 'STRING'
    }
  }

  /**
   * Creates a configuration section for a single widget
   */
  createWidgetSection(node: MTBNode, widget: IWidget): HTMLElement | undefined {
    const section = document.createElement('div')
    const options = node.properties.mtb_api || {}
    section.classList.add('mtb_api_section')

    // Checkbox to enable/disable section
    const enableCheckbox = document.createElement('input')
    enableCheckbox.type = 'checkbox'

    let enabled = options.inputs?.[widget.name]?.enabled
    enabled = enabled === undefined ? true : enabled

    this.applySettings(node, {
      inputs: { [widget.name]: { enabled } },
    })

    if (options.showDisabled === false && !enabled) {
      return undefined
    }

    enableCheckbox.checked = enabled

    const checkboxLabel = document.createElement('label')
    checkboxLabel.classList.add('mtb_api_checkbox_label')
    checkboxLabel.appendChild(enableCheckbox)
    checkboxLabel.appendChild(document.createTextNode('Enable '))

    const contentContainer = document.createElement('div')

    enableCheckbox.addEventListener('change', () => {
      if (enableCheckbox.checked) {
        this.applySettings(node, {
          inputs: { [widget.name]: { enabled: true } },
        })
        contentContainer.classList.remove('mtb_api_disabled')
      } else {
        this.applySettings(node, {
          inputs: { [widget.name]: { enabled: false } },
        })
        contentContainer.classList.add('mtb_api_disabled')
      }
      notifyAPIChanged()
    })

    const title = document.createElement('span')
    title.classList.add('mtb_api_title')
    title.textContent = widget.name

    checkboxLabel.appendChild(title)

    // Type selector
    const typeLabel = document.createElement('label')
    typeLabel.textContent = 'type:'
    typeLabel.htmlFor = 'inputType'

    const typeSelect = document.createElement('select')
    typeSelect.id = 'inputType'

    const selectedValue =
      options.inputs?.[widget.name]?.type ||
      this.apiTypeFromComfyType(widget.type as string)

    for (const option of API_INPUT_TYPES) {
      const optionElement = document.createElement('option')
      optionElement.value = option
      optionElement.text = option
      if (option === selectedValue) {
        this.applySettings(node, {
          inputs: { [widget.name]: { type: option } },
        })
        optionElement.selected = true
      }
      typeSelect.appendChild(optionElement)
    }

    typeSelect.addEventListener('change', () => {
      this.applySettings(node, {
        inputs: { [widget.name]: { type: typeSelect.value as APIInputType } },
      })
      notifyAPIChanged()
    })

    // Name input
    const nameLabel = document.createElement('label')
    nameLabel.textContent = 'name:'
    nameLabel.htmlFor = 'name'

    const nameInput = document.createElement('input')
    nameInput.type = 'text'
    nameInput.id = 'name'
    nameInput.value = options.inputs?.[widget.name]?.name || widget.name

    nameInput.addEventListener('change', () => {
      this.applySettings(node, {
        inputs: { [widget.name]: { name: nameInput.value } },
      })
      notifyAPIChanged()
    })

    const separator = document.createElement('hr')

    contentContainer.appendChild(nameLabel)
    contentContainer.appendChild(nameInput)
    contentContainer.appendChild(typeLabel)
    contentContainer.appendChild(typeSelect)

    if (!enabled) {
      contentContainer.classList.add('mtb_api_disabled')
    }

    section.appendChild(checkboxLabel)
    section.appendChild(contentContainer)
    section.appendChild(separator)
    this.updateStyleElement()

    return section
  }

  /**
   * Draws the API indicator overlay on the node
   * Clean, minimal design with subtle glow effect
   */
  drawForeground(
    node: MTBNode,
    ctx: CanvasRenderingContext2D,
    _canvas: LGraphCanvas,
  ): void {
    const isOutput = node.properties.mtb_api?.isAPIOutput === true
    const isInput = node.properties.useAPI

    if (!isInput && !isOutput) {
      return
    }

    this.ensureWidgets(node)

    const color = isOutput ? OUTPUT_COLOR : API_COLOR
    const label = isOutput ? 'OUTPUT' : 'API'
    const padding = 4
    const borderRadius = 8
    const titleHeight = LiteGraph.NODE_TITLE_HEIGHT

    ctx.save()

    // Subtle outer glow
    ctx.shadowColor = color
    ctx.shadowBlur = 12
    ctx.shadowOffsetX = 0
    ctx.shadowOffsetY = 0

    // Draw subtle border
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.roundRect(
      -padding,
      -(titleHeight + padding),
      node.size[0] + padding * 2,
      node.size[1] + titleHeight + padding * 2,
      borderRadius,
    )
    ctx.stroke()

    // Reset shadow for badge
    ctx.shadowBlur = 0

    // Draw compact badge
    const badgeHeight = 18
    const badgeY = -(titleHeight + padding + badgeHeight + 4)

    ctx.font = '600 10px -apple-system, BlinkMacSystemFont, sans-serif'
    const textMetrics = ctx.measureText(label)
    const badgeWidth = textMetrics.width + 12

    // Badge background
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.roundRect(0, badgeY, badgeWidth, badgeHeight, 4)
    ctx.fill()

    // Badge text
    ctx.fillStyle = '#fff'
    ctx.textBaseline = 'middle'
    ctx.fillText(label, 6, badgeY + badgeHeight / 2)

    // Draw small icon indicator in corner
    const iconSize = 8
    const iconX = node.size[0] - iconSize - 4
    const iconY = -titleHeight + 4

    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(iconX + iconSize / 2, iconY + iconSize / 2, iconSize / 2, 0, Math.PI * 2)
    ctx.fill()

    ctx.restore()
  }
}

/** Singleton instance */
export const apiSettingsWidgetManager = new APISettingsWidgetManager()
