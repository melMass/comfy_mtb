/**
 * File: debug.js
 * Project: comfy_mtb
 * Author: Mel Massadian
 *
 * Copyright (c) 2023 Mel Massadian
 *
 */

// Reference the shared typedefs file
/// <reference path="../types/typedefs.js" />

import { app } from '../../scripts/app.js'

import {
  setupDynamicConnections,
  cleanupNode,
  infoLogger,
} from './comfy_shared.js'
import * as mtb_ui from './mtb_ui.js'

function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
}

function createDebugSection(title) {
  const section = mtb_ui.makeElement('div', {
    margin: '8px 0',
    padding: '8px',
    borderRadius: '4px',
    backgroundColor: 'rgba(0,0,0,0.2)',
  })

  const header = mtb_ui.makeElement('h3', {
    margin: '0 0 8px 0',
    padding: '4px 0',
    borderBottom: '1px solid rgba(255,255,255,0.1)',
    fontSize: '14px',
    fontWeight: 'bold',
    color: '#9f9',
  })
  header.textContent = title
  section.appendChild(header)

  return section
}

function createDebugContent(item) {
  const wrapper = mtb_ui.makeElement('div', {
    margin: '4px 0',
  })

  if (item.kind === 'text') {
    const text = mtb_ui.makeElement('div', {
      margin: '2px 0',
      fontFamily: 'monospace',
      whiteSpace: 'pre-wrap',
    })
    text.innerHTML = item.data
    wrapper.appendChild(text)
  } else if (item.kind === 'b64_images') {
    const img = mtb_ui.makeElement('img', {
      width: '100%',
      borderRadius: '2px',
    })
    img.src = item.data
    wrapper.appendChild(img)
  }

  return wrapper
}

app.registerExtension({
  name: 'mtb.Debug',

  /**
   * @param {NodeType} nodeType
   * @param {NodeData} nodeData
   * @param {*} app
   */
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === 'Debug (mtb)') {
      const clear_widgets = (target) => {
        if (target.widgets) {
          let tgt_len = target.widgets.length
          for (let i = 0; i < target.widgets.length; i++) {
            if (
              ![
                'output_to_console',
                'deep_inspect',
                'as_detailed_types',
                'rich_mode',
              ].includes(target.widgets[i].name)
            ) {
              target.widgets[i].onRemove?.()
              target.widgets[i].onRemoved?.()
              tgt_len -= 1
            }
          }
          target.widgets.length = tgt_len
        }
      }

      const original_getExtraMenuOptions =
        nodeType.prototype.getExtraMenuOptions
      nodeType.prototype.getExtraMenuOptions = function (_, options) {
        original_getExtraMenuOptions?.apply(this, arguments)
        options.push({
          content: '🐛 Clear Outputs',
          callback: async () => {
            clear_widgets(this)
          },
        })
      }

      setupDynamicConnections(nodeType, 'var', '*')

      const onExecuted = nodeType.prototype.onExecuted
      nodeType.prototype.onExecuted = function (...args) {
        onExecuted?.apply(this, args)
        const [data, ..._rest] = args

        clear_widgets(this)

        const inputData = {}

        const uiData = data.ui || data

        const name_to_label = this.inputs.reduce((acc, input) => {
          acc[input.name] = input.label || input.name
          return acc
        }, {})


        if (uiData.items) {
          uiData.items.forEach((item) => {
            const inputName = item.input
            inputData[inputName] = item.items
          })
        }
        const mainDebugContainer = mtb_ui.makeElement('div', {
          width: '100%',
        })
        let hasContent = false
        for (const [inputName, content] of Object.entries(inputData)) {
          if (!content || content?.length === 0) {
            continue
          }
          hasContent = true

          const section = createDebugSection(name_to_label[inputName])

          for (const item of content) {
            section.appendChild(createDebugContent(item))
          }
          mainDebugContainer.appendChild(section)
        }
        if (hasContent) {
          this.addDOMWidget('debug_output', 'CUSTOM', mainDebugContainer, {
            hideOnZoom: false,
          })
        }

        this.onRemoved = function () {
          for (const widget of this.widgets) {
            if (widget.canvas) {
              widget.canvas.remove()
            }
            widget.onRemoved?.()
            widget.onRemove?.()
          }
          cleanupNode(this)
        }
        this.setDirtyCanvas(true, true)
      }
    }
  },
})
