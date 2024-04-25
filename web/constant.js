import { app } from '../../scripts/app.js'
import * as shared from './comfy_shared.js'
import { infoLogger } from './comfy_shared.js'
import { MtbWidgets } from './mtb_widgets.js'
import { ComfyWidgets } from '../../scripts/widgets.js'
import * as mtb_widgets from './mtb_widgets.js'

/**
 * @typedef {'number'|'string'|'vector2'|'vector3'|'vector4'|'color'} ConstantType
 * @typedef {import ("../../../web/types/litegraph.d.ts").LGraphNode} Node
 * @typedef {{x:number,y:number,z?:number,w?:number}} VectorValue
 * @typedef {}
 *
 */

/**
 * @param {number} size - The number of axis of the vector (2,3 or 4)
 * @param {number} val - The default scalar value to fill the vector with
 * @returns {VectorValue} vector
 * */
const initVector = (size, val = 0.0) => {
  const res = {}
  for (let i = 0; i < size; i++) {
    const axis = mtb_widgets.VECTOR_AXIS[i]
    res[axis] = val
  }
  return res
}

/**
 *
 * @extends {Node}
 * @classdesc Wrapper for the python node
 */
export class ConstantJs {
  constructor(python_node) {
    // this.uuid = shared.makeUUID()
    const wrapper = this

    python_node.shape = LiteGraph.BOX_SHAPE
    python_node.serialize_widgets = true

    const onNodeCreated = python_node.prototype.onNodeCreated
    python_node.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this) : undefined

      this.addProperty('type', 'number')
      this.addProperty('value', 0)

      this.removeInput(0)
      this.removeOutput(0)

      this.addOutput('Output', '*')

      // bind our wrapper
      this.configure = wrapper.configure.bind(this)
      // this.applyToGraph = wrapper.applyToGraph.bind(this)
      this.updateWidgets = wrapper.updateWidgets.bind(this)
      this.convertValue = wrapper.convertValue.bind(this)
      // this.updateOutput = wrapper.updateOutput.bind(this)
      this.updateOutputType = wrapper.updateOutputType.bind(this)
      // this.updateTargetWidgets = wrapper.updateTargetWidgets.bind(this)

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
          values: [
            // 'number',
            'float',
            'int',
            'string',
            'vector2',
            'vector3',
            'vector4',
            'color',
          ],
        },
      )
      this.updateWidgets()
      this.updateOutputType()

      for (let n = 0; n < this.inputs.length; n++) {
        this.removeInput(n)
      }
      this.inputs = []
      return r
    }
    return
  }

  // NOTE: this is called onPrompt
  // applyToGraph() {
  // infoLogger('Updating values for backend')
  // this.updateTargetWidgets()
  // }

  // NOTE: deserialization happens here
  configure(info) {
    // super.configure(info)
    infoLogger('Configure Constant', { info, node: this })

    this.properties.type = info.properties.type
    this.properties.value = info.properties.value

    this.pos = info.pos
    this.order = info.order

    this.updateWidgets()
    this.updateOutputType()
  }

  /**
   * Convert the old value type to the new one, falling back to some default
   * @param {ConstantType} propType - The target type
   */
  convertValue(propType) {
    switch (propType) {
      case 'color': {
        if (typeof this.properties.value !== 'string') {
          this.properties.value = '#ffffff'
        } else if (this.properties.value[0] !== '#') {
          this.properties.value = '#ff0000'
        }
        break
      }
      case 'int': {
        if (typeof this.properties.value === 'object') {
          this.properties.value = Number.parseInt(this.properties.value.x)
        } else {
          this.properties.value = Number.parseInt(this.properties.value) || 0
        }
        break
      }
      case 'float': {
        if (typeof this.properties.value === 'object') {
          this.properties.value = Number.parseFloat(this.properties.value.x)
        } else {
          this.properties.value =
            Number.parseFloat(this.properties.value) || 0.0
        }
        break
      }
      case 'string': {
        if (typeof this.properties.value !== 'string') {
          this.properties.value = JSON.stringify(this.properties.value)
        }
        break
      }
      case 'vector2':
      case 'vector3':
      case 'vector4': {
        const numInputs = Number.parseInt(propType.charAt(6))
        if (!this.properties.value) {
          this.properties.value = initVector(numInputs) // Array.from({ length: numInputs }, () => 0.0)
        } else if (typeof this.properties.value === 'string') {
          try {
            const parsed = JSON.parse(this.properties.value)
            const newVec = {}
            for (
              let i = 0;
              i < Object.keys(mtb_widgets.VECTOR_AXIS).length;
              i++
            ) {
              const axis = mtb_widgets.VECTOR_AXIS[i]
              if (Object.keys(parsed).includes(axis)) {
                newVec[axis] = parsed[axis]
              }
            }
            this.properties.value = newVec
          } catch (e) {
            shared.errorLogger(e)
            infoLogger(
              `Couldn't parse string to vec (${this.properties.value})`,
            )
            this.properties.value = initVector(numInputs)
          }
        } else if (typeof this.properties.value === 'number') {
          const newVec = initVector(numInputs)
          newVec.x = Number.parseFloat(this.properties.value)
          this.properties.value = newVec
        }

        if (
          typeof this.properties.value === 'object' &&
          Object.keys(this.properties.value).length !== numInputs
        ) {
          const current = Object.keys(this.properties.value)
          if (current.length < numInputs) {
            infoLogger('current value smaller than target, adjusting')
            for (let index = current.length; index < numInputs; index++) {
              this.properties.value[mtb_widgets.VECTOR_AXIS[index]] = 0.0
            }
          } else {
            infoLogger('current value greater than target, adjusting')
            const newVal = {}
            for (let index = 0; index < numInputs; index++) {
              newVal[mtb_widgets.VECTOR_AXIS[index]] =
                this.properties.value[mtb_widgets.VECTOR_AXIS[index]]
            }
            this.properties.value = newVal
          }
        }
        break
      }
      default:
        break
    }
  }

  /**
   * Remove all widgets but the comboBox for selecting the type
   * then recreate the appropriate widget from scratch
   */
  updateWidgets() {
    // NOTE: Remove existing widgets
    for (let i = 1; i < this.widgets.length; i++) {
      const element = this.widgets[i]
      if (element.onRemove) {
        element.onRemove()
      }
      // element?.onRemove()
    }

    this.widgets.splice(1)
    this.widgets[0].value = this.properties.type

    this.convertValue(this.properties.type)

    switch (this.properties.type) {
      case 'color': {
        const col_widget = this.addCustomWidget(
          MtbWidgets.COLOR('Value', this.properties.value),
        )
        col_widget.callback = (col) => {
          this.properties.value = col
          // this.updateOutput()
        }
        break
      }
      case 'int': {
        const f_widget = this.addCustomWidget(
          ComfyWidgets.INT(
            this,
            'Value',
            [
              '',
              {
                default: this.properties.value,
                callback: (val) => console.log('VALUE', val),
              },
            ],
            app,
          ),
        )

        f_widget.widget.callback = (val) => {
          this.properties.value = val
        }

        break
      }
      case 'float': {
        this.addWidget('number', 'Value', this.properties.value, (val) => {
          this.properties.value = val
        })
        break
      }
      case 'string': {
        mtb_widgets.addMultilineWidget(
          this,
          'Value',
          {
            defaultVal: this.properties.value,
          },
          (v) => {
            this.properties.value = v
            // this.updateOutput()
          },
        )
        break
      }
      case 'vector2':
      case 'vector3':
      case 'vector4': {
        const numInputs = Number.parseInt(this.properties.type.charAt(6))
        const node = this
        const v_widget = mtb_widgets.addVectorWidget(
          this,
          'Value',
          this.properties.value, // value
          numInputs, // vector_size
          function (v) {
            node.properties.value = v
            // this.updateOutput()
          },
        )
        break
      }

      // NOTE: this is not reached anymore, kept for reference
      case 'number': {
        if (typeof this.properties.value !== 'number') {
          this.properties.value = 0.0
        }
        const n_widget = this.addWidget(
          'number',
          'Value',
          this.properties.force_int
            ? Number.parseInt(this.properties.value)
            : this.properties.value,
          (value) => {
            this.properties.value = this.properties.force_int
              ? Number.parseInt(value)
              : value
            // this.updateOutput()
          },
        )
        //override the callback
        const origCallback = n_widget.callback
        const node = this
        n_widget.callback = function (val) {
          const r = origCallback ? origCallback.apply(this, [val]) : undefined
          if (node.properties.force_int) {
            // TODO: rework this, a it makes it harder to manipulate
            this.value = Number.parseInt(this.value)
            node.properties.value = Number.parseInt(this.value)
          }
          infoLogger('NEW NUMBER', this.value)
          return r
        }

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
    infoLogger('Updating output type')
    const rm_if_mismatch = (type) => {
      if (this.outputs[0].type !== type) {
        for (let i = 0; i < this.outputs.length; i++) {
          this.removeOutput(i)
        }
        this.addOutput('output', type)
        // this.setOutputDataType(0, type)
      }
    }
    switch (this.properties.type) {
      case 'color':
        rm_if_mismatch('COLOR')
        break
      case 'float':
        rm_if_mismatch('FLOAT')
        break
      case 'int':
        rm_if_mismatch('INT')
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
      // case 'vector2':
      // case 'vector3':
      // case 'vector4':
      //   rm_if_mismatch('FLOAT')
      //   break
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
    // this.updateOutput()
  }

  /**
   * NOTE: This feels hacky but seems to work fine
   * since Constant is a virtual node.
   */
  updateTargetWidgets(u_links) {
    infoLogger('Updating target widgets')
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
      // infoLogger('Constant Target Node', tgt_node)
      // infoLogger('Constant Target Input', tgt_input)
      if (!tgt_widget || tgt_widget.length === 0) return

      tgt_widget[0].value = this.properties.value
    }
  }

  updateOutput() {
    infoLogger('Updating output value')
    const value = this.properties.value

    switch (this.properties.type) {
      case 'color':
        this.setOutputData(0, value)
        break
      case 'number':
        if (this.properties.force_int) {
          this.setOutputData(0, Number.parseInt(value))
        } else {
          this.setOutputData(0, Number.parseFloat(value))
        }
        break
      case 'string':
        this.setOutputData(0, value.toString())
        break
      case 'vector2':
      case 'vector3':
      case 'vector4':
        this.setOutputData(0, value)
        break

      // case 'vector2':
      //   this.setOutputData(0, value.slice(0, 2))
      //   break
      // case 'vector3':
      //   this.setOutputData(0, value.slice(0, 3))
      //   break
      // case 'vector4':
      //   this.setOutputData(0, value.slice(0, 4))
      //   break
      default:
        break
    }

    infoLogger('New Value', this.value)

    this.updateTargetWidgets()
  }
}
app.registerExtension({
  name: 'mtb.constant',

  async beforeRegisterNodeDef(nodeType, nodeData, _app) {
    if (nodeData.name === 'Constant (mtb)') {
      new ConstantJs(nodeType)
    }
  },
  // NOTE: old js only registration
  //
  // registerCustomNodes() {
  //   LiteGraph.registerNodeType('Constant (mtb)', Constant)
  //
  //   Constant.category = 'mtb/utils'
  //   Constant.title = 'Constant (mtb)'
  // },
})
