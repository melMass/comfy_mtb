/**
 * API Settings Widget Manager
 * Manages DOM widgets for configuring API inputs on nodes
 */

import type { IWidget, LGraphCanvas } from '@comfyorg/litegraph'
import { $el } from '@/scripts/ui'
import * as shared from '@mtb/shared'
import { API_COLOR, OUTPUT_COLOR } from './constants'
import { API_INPUT_TYPES, type APIInputType, type APINodeSettings, type MTBNode } from './types'
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
   * Creates the node-level settings section
   */
  createNodeSettingsSection(_node: MTBNode, element: HTMLElement): void {
    const nodeSettings = $el('div', {}, [
      $el('p', { textContent: 'Settings' }),
      $el('input', { type: 'checkbox' }),
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
  apiTypeFromComfyType(comfyType: string): APIInputType | '' {
    switch (comfyType) {
      case 'number':
        return 'NUMBER'
      case 'customtext':
        return 'STRING'
      case 'combo':
        return 'COMBO'
      default:
        console.log('UNHANDLED widget type:', comfyType)
        return ''
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
   */
  drawForeground(
    node: MTBNode,
    ctx: CanvasRenderingContext2D,
    _canvas: LGraphCanvas,
  ): void {
    if (
      !node.properties.useAPI &&
      !(node.properties.mtb_api?.isAPIOutput === true)
    ) {
      return
    }

    this.ensureWidgets(node)

    const border = 10
    const borderH = border / 2
    const offset = 10

    ctx.save()
    ctx.beginPath()

    const color = node.properties.mtb_api?.isAPIOutput
      ? OUTPUT_COLOR
      : API_COLOR

    // Draw border
    ctx.fillStyle = color
    ctx.strokeStyle = color
    ctx.lineWidth = border
    ctx.roundRect(
      -(borderH + offset / 2),
      -(LiteGraph.NODE_TITLE_HEIGHT + borderH + offset / 2),
      node.size[0] + border + offset,
      node.size[1] + border + LiteGraph.NODE_TITLE_HEIGHT + offset,
      LiteGraph.NODE_COLLAPSED_RADIUS,
      LiteGraph.NODE_COLLAPSED_RADIUS,
    )
    ctx.stroke()

    // Draw label
    const message = node.properties.mtb_api?.isAPIOutput ? 'API Output' : 'API'

    ctx.font = '24px monospace'
    const textSize = ctx.measureText(message)
    ctx.beginPath()
    ctx.roundRect(
      0,
      -(LiteGraph.NODE_TITLE_HEIGHT + borderH + offset / 2 + 32),
      textSize.width + 12,
      32,
      5,
    )
    ctx.fill()

    ctx.fillStyle = '#fff'
    ctx.fillText(message, 6, -LiteGraph.NODE_TITLE_HEIGHT - 18)
    ctx.restore()
  }
}

/** Singleton instance */
export const apiSettingsWidgetManager = new APISettingsWidgetManager()
