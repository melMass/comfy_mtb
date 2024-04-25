import { app } from '../../scripts/app.js'
import * as shared from './comfy_shared.js'
import { MtbWidgets } from './mtb_widgets.js'

export class Constant extends LiteGraph.LGraphNode {
  constructor() {
    super()
    this.uuid = shared.makeUUID()
    this.collapsable = true

    // this avoid serializing the node when converting to prompt
    this.isVirtualNode = true

    this.shape = LiteGraph.BOX_SHAPE
    this.serialize_widgets = true

    // Properties
    this.addProperty('type', 'number')
    this.addProperty('value', 0)

    // Inputs and outputs
    this.addOutput('Output', '*')

    // Widget for selecting the type
    this.addWidget(
      'combo',
      'Type',
      this.properties.type,
      (value) => {
        this.properties.type = value
        this.updateWidgets()
        this.updateOutputType()
      },
      {
        values: ['number', 'string', 'vector2', 'vector3', 'vector4', 'color'],
      },
    )

    // Initialize the node
    this.updateWidgets()
    this.updateOutputType()
  }
  // NOTE: this is called onPrompt
  applyToGraph() {
    this.updateTargetWidgets()
  }

  configure(info) {
    super.configure(info)
    this.properties.type = info.properties.type
    this.properties.value = info.properties.value

    shared.infoLogger('Configure Constant', { info, node: this })
    this.updateWidgets()
    this.updateOutputType()
  }

  updateWidgets() {
    // Remove existing widgets
    for (let i = 1; i < this.widgets.length; i++) {
      const element = this.widgets[i]
      if (element.onRemove) {
        element.onRemove()
      }
      // element?.onRemove()
    }

    this.widgets.splice(1)
    switch (this.properties.type) {
      case 'color': {
        if (typeof this.properties.value !== 'string') {
          this.properties.value = '#ffffff'
        }
        const col_widget = this.addCustomWidget(
          MtbWidgets.COLOR('Value', this.properties.value || '#ff0000'),
        )
        col_widget.callback = (col) => {
          this.properties.value = col
          this.updateOutput()
        }
        break
      }
      case 'number':
        if (typeof this.properties.value !== 'number') {
          this.properties.value = 0.0
        }
        this.addWidget('number', 'Value', this.properties.value, (value) => {
          this.properties.value = value
          this.updateOutput()
        })
        this.addWidget(
          'toggle',
          'Convert to Integer',
          this.properties.force_int,
          (value) => {
            this.properties.force_int = value
            this.updateOutputType()
          },
        )
        break
      case 'string': {
        if (typeof this.properties.value !== 'string') {
          this.properties.value = `${this.properties.value}`
        }
        shared.addMultilineWidget(
          this,
          'Value',
          {
            defaultVal: this.properties.value,
          },
          (v) => {
            this.properties.value = v
            this.updateOutput()
          },
        )
        break
      }
      case 'vector2':
      case 'vector3':
      case 'vector4': {
        const numInputs = Number.parseInt(this.properties.type.charAt(6))

        if (['string', 'number'].includes(typeof this.properties.value)) {
          this.properties.value = Array.from({ length: numInputs }, () => 0.0)
        } else if (this.properties.value.length !== numInputs) {
          if (this.properties.value.length > numInputs) {
            this.properties.value = this.properties.value.slice(0, numInputs)
          } else {
            this.properties.value = this.properties.value.concat(
              new Array(numInputs - this.properties.value.length).fill(0.0),
            )
          }
        }
        for (let i = 0; i < numInputs; i++) {
          this.addWidget(
            'number',
            `Value ${i + 1}`,
            this.properties.value[i] || 0,
            (value) => {
              this.properties.value[i] = value
              this.updateOutput()
            },
          )
        }
        break
      }
      default:
        break
    }
  }
  onConnectionsChange(type, slotIndex, isConnected, link, ioSlot) {
    // super.onConnectionsChange(type, slotIndex, isConnected, link, ioSlot)
    if (isConnected) {
      this.updateTargetWidgets([link.id])
    }
  }
  updateOutputType() {
    const cur_type = this.outputs[0].type
    const rm_if_mismatch = (type) => {
      if (cur_type !== type) {
        for (let i = 0; i < this.outputs.length; i++) {
          this.removeOutput(i)
        }
        this.addOutput('output', type)
      }
    }
    switch (this.properties.type) {
      case 'color':
        rm_if_mismatch('COLOR')
        break
      case 'number':
        if (this.properties.force_int) {
          rm_if_mismatch('INT')
        } else {
          rm_if_mismatch('FLOAT')
        }
        break
      case 'string':
        rm_if_mismatch('STRING')
        break
      case 'vector2':
        rm_if_mismatch('VECTOR2')
        break
      case 'vector3':
        rm_if_mismatch('VECTOR3')
        break
      case 'vector4':
        rm_if_mismatch('VECTOR4')
        break
      default:
        break
    }
    this.updateOutput()
  }

  /**
   * NOTE: This feels hacky but seems to work fine
   * since Constant is a virtual node.
   */
  updateTargetWidgets(u_links) {
    if (!app.graph.links) return
    const links = u_links || this.outputs[0].links
    if (!links) return
    for (let i = 0; i < links.length; i++) {
      const link = app.graph.links[links[i]]
      const tgt_node = app.graph.getNodeById(link.target_id)
      if (!tgt_node || !tgt_node.inputs) return
      const tgt_input = tgt_node.inputs[link.target_slot]
      if (!tgt_input) return
      const tgt_widget = tgt_node.widgets.filter(
        (w) => w.name === tgt_input.name,
      )
      if (!tgt_widget) return
      tgt_widget[0].value = this.properties.value
    }
  }

  updateOutput() {
    const value = this.properties.value

    switch (this.properties.type) {
      case 'color':
        this.setOutputData(0, value)
        break
      case 'number':
        this.setOutputData(0, Number.parseFloat(value))
        break
      case 'string':
        this.setOutputData(0, value.toString())
        break
      case 'vector2':
        if (value.length >= 2) {
          this.setOutputData(0, value.slice(0, 2))
        }
        break
      case 'vector3':
        if (value.length >= 3) {
          this.setOutputData(0, value.slice(0, 3))
        }
        break
      case 'vector4':
        if (value.length >= 4) {
          this.setOutputData(0, value.slice(0, 4))
        }
        break
      default:
        break
    }

    this.updateTargetWidgets()
  }
}

// app.registerExtension({
//   name: 'mtb.constant',
//   registerCustomNodes() {
//     LiteGraph.registerNodeType('Constant (mtb)', Constant)
//
//     Constant.category = 'mtb/utils'
//     Constant.title = 'Constant (mtb)'
//   },
// })
