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
import { MtbWidgets } from './mtb_widgets.js'
import * as mtb_ui from './mtb_ui.js'

// TODO: respect inputs order...

function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
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

        const prefix = 'anything_'

        if (this.widgets) {
          for (let i = 0; i < this.widgets.length; i++) {
            if (this.widgets[i].name !== 'output_to_console') {
              this.widgets[i].onRemove?.()
              this.widgets[i].onRemoved?.()
            }
          }
          this.widgets.length = 1
        }
        let widgetI = 1
        // console.log(message)
        if (data.text) {
          for (const txt of data.text) {
            const textDom = mtb_ui.makeElement('p', { fontFamily: 'monospace' })
            textDom.innerHTML = txt

            this.addDOMWidget(
              `${prefix}_${widgetI}`,
              'CUSTOM_TEXT',
              textDom,
              {},
            )
            widgetI++
          }
        }
        if (data.b64_images) {
          for (const img of data.b64_images) {
            const imgDom = mtb_ui.makeElement('img', { width: '100%' })
            imgDom.src = img

            this.addDOMWidget(
              `${prefix}_${widgetI}`,
              'CUSTOM_IMG_B64',
              mtb_ui.wrapElement(imgDom, {
                overflow: 'hidden',
              }),
              {},
            )

            widgetI++
          }
        }

        // this.setSize(this.computeSize())

        this.onRemoved = function () {
          // When removing this node we need to remove the input from the DOM
          for (const y in this.widgets) {
            if (this.widgets[y].canvas) {
              this.widgets[y].canvas.remove()
            }
            shared.cleanupNode(this)
            this.widgets[y].onRemoved?.()
            this.widgets[y].onRemove?.()
          }
        }
      }
    }
  },
})
