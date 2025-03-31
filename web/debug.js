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
import * as shared from './comfy_shared.js'
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
    backgroundColor: 'rgba(0,0,0,0.2)'
  })

  const header = mtb_ui.makeElement('h3', {
    margin: '0 0 8px 0',
    padding: '4px 0',
    borderBottom: '1px solid rgba(255,255,255,0.1)',
    fontSize: '14px',
    fontWeight: 'bold',
    color: '#9f9'
  })
  header.textContent = title
  section.appendChild(header)

  return section
}

function createDebugContent(content, type) {
  const wrapper = mtb_ui.makeElement('div', {
    margin: '4px 0'
  })

  if (type === 'text') {
    const text = mtb_ui.makeElement('p', {
      margin: '2px 0',
      fontFamily: 'monospace',
      whiteSpace: 'pre-wrap'
    })
    text.innerHTML = content
    wrapper.appendChild(text)
  } else if (type === 'image') {
    const img = mtb_ui.makeElement('img', {
      width: '100%',
      borderRadius: '2px'
    })
    img.src = content
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
      const onNodeCreated = nodeType.prototype.onNodeCreated
      nodeType.prototype.onNodeCreated = function (...args) {
        this.options = {}
        const r = onNodeCreated ? onNodeCreated.apply(this, args) : undefined
        this.addInput('anything_1', '*')
        return r
      }

      const onConnectionsChange = nodeType.prototype.onConnectionsChange
      /**
       * @param {OnConnectionsChangeParams} args
       */
      nodeType.prototype.onConnectionsChange = function (...args) {
        const [_type, index, connected, link_info, ioSlot] = args
        const r = onConnectionsChange
          ? onConnectionsChange.apply(this, args)
          : undefined
        // TODO: remove all widgets on disconnect once computed
        shared.dynamic_connection(this, index, connected, 'anything_', '*', {
          link: link_info,
          ioSlot: ioSlot,
        })

        //- infer type
        if (link_info) {
          // const fromNode = this.graph._nodes.find(
          // (otherNode) => otherNode.id === link_info.origin_id,
          // )
          // const fromNode = app.graph.getNodeById(link_info.origin_id)
          const { from } = shared.nodesFromLink(this, link_info)
          if (!from || this.inputs.length === 0) return
          const type = from.outputs[link_info.origin_slot].type
          this.inputs[index].type = type
          // this.inputs[index].label = type.toLowerCase()
        }
        //- restore dynamic input
        if (!connected) {
          this.inputs[index].type = '*'
          this.inputs[index].label = `anything_${index + 1}`
        }
        return r
      }

      const onExecuted = nodeType.prototype.onExecuted
      nodeType.prototype.onExecuted = function (...args) {
        onExecuted?.apply(this, args)
        const [data, ..._rest] = args

        if (this.widgets) {
          let tgt_len = this.widgets.length
          for (let i = 0; i < this.widgets.length; i++) {
            if (
              this.widgets[i].name !== 'output_to_console' &&
              this.widgets[i].name !== 'as_detailed_types'
            ) {
              this.widgets[i].onRemove?.()
              this.widgets[i].onRemoved?.()
              tgt_len -= 1
            }
          }
          this.widgets.length = tgt_len
        }

        const inputData = {}

        const uiData = data.ui || data

        if (uiData.items) {
            uiData.items.forEach(item => {
                const inputName = item.input
                if (!inputData[inputName]) {
                    inputData[inputName] = { text: [], b64_images: [] }
                }
                if (item.text) {
                    inputData[inputName].text.push(...item.text)
                }
                if (item.b64_images) {
                    inputData[inputName].b64_images.push(...item.b64_images)
                }
            })
        }

        let widgetI = 1
        for (const [inputName, content] of Object.entries(inputData)) {
          if (content.text.length === 0 && content.b64_images.length === 0) {
            continue
          }

          const section = createDebugSection(inputName)

          if (content.text.length > 0) {
            content.text.forEach(text => {
              section.appendChild(createDebugContent(text, 'text'))
            })
          }

          if (content.b64_images.length > 0) {
            content.b64_images.forEach(img => {
              section.appendChild(createDebugContent(img, 'image'))
            })
          }

          this.addDOMWidget(
            `debug_section_${widgetI}`,
            'CUSTOM',
            section,
            {}
          )
          widgetI++
        }

        this.onRemoved = function () {
          for (const widget of this.widgets) {
            if (widget.canvas) {
              widget.canvas.remove()
            }
            widget.onRemoved?.()
            widget.onRemove?.()
          }
          shared.cleanupNode(this)
        }
      }
    }
  },
})
